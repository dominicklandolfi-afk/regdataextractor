"""Product category taxonomy and category-driven regulatory defaults.

Replaces the dosage-form-only inert defaults with a richer classifier that
considers dosage form, active ingredients, generic name, product name, and
route hints to assign each product to one of ten regulatory categories.
Each category has a known profile: flash point class, transport class,
RCRA classification, etc. Python is authoritative for the categories where
the answer is deterministic (oral solids, alcohol topicals, transdermal
patches, etc.); Perplexity is the cross-check.

Built to be testable without hitting any live API: classify_product takes
a DailyMedData and a list of active ingredients and returns a category.
The profile lookup is a pure dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .schema import DailyMedData


class ProductCategory(str, Enum):
    """Regulatory classification, not a marketing classification."""

    ORAL_SOLID = "ORAL_SOLID"
    ORAL_LIQUID_AQUEOUS = "ORAL_LIQUID_AQUEOUS"
    ORAL_LIQUID_ALCOHOL = "ORAL_LIQUID_ALCOHOL"
    TOPICAL_NONFLAM = "TOPICAL_NONFLAM"
    TOPICAL_ALCOHOL = "TOPICAL_ALCOHOL"
    AEROSOL_PROPELLED = "AEROSOL_PROPELLED"
    DRY_POWDER_INHALER = "DRY_POWDER_INHALER"
    TRANSDERMAL_PATCH = "TRANSDERMAL_PATCH"
    LOZENGE = "LOZENGE"
    OPHTHALMIC_OTIC = "OPHTHALMIC_OTIC"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FieldDefault:
    """A known-correct answer for a regulatory field within a category.

    override_below: if Perplexity returned this field with confidence
        strictly below this number, replace with `value` at `confidence`.
        Set above 100 to always override; set to 0 to never override.
    """

    value: str
    confidence: int
    override_below: int = 70


@dataclass(frozen=True)
class CategoryProfile:
    """The regulatory profile for a product category.

    defaults: per-field deterministic answers. Each field that does not
        appear here is left to Perplexity / PubChem entirely.
    canonical_state: the secondary_physical_state value to force when
        Perplexity picks something inconsistent with the dosage form.
        None means "let the existing dosage-form mapping handle it."
    """

    category: ProductCategory
    defaults: dict[str, FieldDefault]
    canonical_state: Optional[str] = None


# ---------------------------------------------------------------------------
# Flammability detection helpers
# ---------------------------------------------------------------------------

_FLAMMABLE_TOKENS: tuple[str, ...] = (
    "ethanol", "ethyl alcohol", "isopropanol", "isopropyl alcohol",
    "acetone", "methanol", "denatured alcohol", "sd alcohol",
    "grain alcohol", "hand sanitizer", "hand rub", "sanitizing",
    "antiseptic",
)

_NONFLAMMABLE_ALCOHOL_PREFIXES: tuple[str, ...] = (
    "cetyl alcohol", "stearyl alcohol", "cetostearyl alcohol",
    "cetearyl alcohol", "benzyl alcohol", "polyvinyl alcohol",
    "lanolin alcohol", "wool alcohol", "lauryl alcohol",
    "myristyl alcohol", "behenyl alcohol", "oleyl alcohol",
)


def has_flammable_solvent(strings: list[str]) -> bool:
    """True when any input string indicates a flammable solvent.

    Handles the bare word 'alcohol' as flammable EXCEPT when it appears
    as a fatty alcohol (cetyl, stearyl, benzyl, polyvinyl, etc.). Also
    triggers on 'hand sanitizer'/'hand rub'/'sanitizing'/'antiseptic'
    product-name keywords as a defense-in-depth signal."""
    blob = " ".join(s for s in strings if s).lower()
    if any(token in blob for token in _FLAMMABLE_TOKENS):
        return True
    if "alcohol" in blob:
        return not any(prefix in blob for prefix in _NONFLAMMABLE_ALCOHOL_PREFIXES)
    return False


def flammability_strings(
    dm: Optional[DailyMedData],
    active_ingredients: list[str],
) -> list[str]:
    """Every DailyMed string that might indicate a flammable solvent."""
    bits: list[str] = list(active_ingredients)
    if dm is not None:
        if dm.generic_name:
            bits.append(dm.generic_name)
        if dm.product_name:
            bits.append(dm.product_name)
    return bits


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_AEROSOL_KEYWORDS: tuple[str, ...] = (
    "AEROSOL", "METERED-DOSE", "METERED DOSE", "PROPELLANT",
)

_INHALER_KEYWORDS: tuple[str, ...] = ("INHALER", "INHALANT", "INHALATION")

_PATCH_KEYWORDS: tuple[str, ...] = ("PATCH", "TRANSDERMAL")

_LOZENGE_KEYWORDS: tuple[str, ...] = ("LOZENGE", "TROCHE")

_OPHTHALMIC_KEYWORDS: tuple[str, ...] = (
    "OPHTHALMIC", "OTIC", "EYE DROP", "EYE DROPS", "EAR DROP",
    "EAR DROPS", "OCULAR", "AURICULAR",
)

# Eye/ear/nose-drop actives - if the only active is one of these, the
# product is overwhelmingly likely to be an eye or ear drop even when
# DailyMed doesn't surface the route explicitly in the product_name.
_OPHTHALMIC_ACTIVE_HINTS: tuple[str, ...] = (
    "tetrahydrozoline", "naphazoline", "oxymetazoline ophthalmic",
    "ketotifen", "olopatadine", "azelastine ophthalmic",
    "brimonidine", "timolol", "latanoprost", "travoprost",
    "dorzolamide", "brinzolamide", "carboxymethylcellulose",
    "hypromellose", "polyethylene glycol 400",
    "carbamide peroxide",  # ear wax removal
)

# Topical product-name hints. Used to disambiguate alcohol-bearing
# 'LIQUID' or 'GEL' products: NyQuil is oral, Purell is topical. The
# product name carries the topical signal in most cases.
_TOPICAL_PRODUCT_NAME_HINTS: tuple[str, ...] = (
    "HAND SANITIZER", "HAND RUB", "HAND GEL", "SANITIZING",
    "ANTISEPTIC", "TOPICAL", "RUB", "WIPE", "SCRUB", "FOR EXTERNAL USE",
    "SKIN CLEANSER",
)

_TOPICAL_DOSAGE_KEYWORDS: tuple[str, ...] = (
    "CREAM", "LOTION", "OINTMENT", "PASTE", "FOAM",
)

_GEL_KEYWORDS: tuple[str, ...] = ("GEL",)

_GEL_EXCLUSIONS: tuple[str, ...] = ("SOFTGEL", "GELCAP")

_SPRAY_KEYWORDS: tuple[str, ...] = ("SPRAY",)

_ORAL_SOLID_KEYWORDS: tuple[str, ...] = (
    "TABLET", "CAPSULE", "CAPLET", "SOFTGEL", "GELCAP",
    "POWDER", "GRANULES", "PELLET",
)

_ORAL_LIQUID_KEYWORDS: tuple[str, ...] = (
    "SOLUTION", "SUSPENSION", "SYRUP", "ELIXIR", "DROPS", "LIQUID",
)

_DENTIFRICE_KEYWORDS: tuple[str, ...] = ("DENTIFRICE", "MOUTHWASH", "RINSE")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def _is_ophthalmic_or_otic(dm: DailyMedData, active_ingredients: list[str]) -> bool:
    """True when DailyMed indicates an eye/ear product. Route info lives
    in the product_name or dosage_form. Falls back to a list of known
    ophthalmic/otic actives when the route is not explicit (Visine doesn't
    say 'EYE' anywhere but tetrahydrozoline is exclusively an eye drop)."""
    haystack = " ".join(
        s.upper() for s in (dm.product_name, dm.dosage_form, dm.generic_name) if s
    )
    if _contains_any(haystack, _OPHTHALMIC_KEYWORDS):
        return True
    actives_blob = " ".join(active_ingredients).lower()
    return any(hint in actives_blob for hint in _OPHTHALMIC_ACTIVE_HINTS)


def _is_topical_by_name(dm: DailyMedData) -> bool:
    """True when product_name contains a topical-route signal. Used to
    distinguish topical alcohol (Purell) from oral alcohol (NyQuil) when
    dosage_form alone is generic ('LIQUID', 'GEL', 'FOAM')."""
    pn = (dm.product_name or "").upper()
    return any(hint in pn for hint in _TOPICAL_PRODUCT_NAME_HINTS)


def classify_product(
    dm: Optional[DailyMedData],
    active_ingredients: list[str],
) -> ProductCategory:
    """Decision tree mapping DailyMed signals to a regulatory category.

    Order matters: more-specific categories fire before generic catches.
    Aerosols and inhalers are checked first because 'AEROSOL FOAM' would
    otherwise match TOPICAL_NONFLAM via the FOAM keyword. Ophthalmic/otic
    is checked next because eye-drop products often have dosage_form
    'SOLUTION' that would otherwise route to ORAL_LIQUID_AQUEOUS.
    Topicals are checked before generic LIQUID so 'CREAM' wins. Oral
    liquids fall through last because LIQUID is the broadest keyword.
    """
    if dm is None or not dm.dosage_form:
        return ProductCategory.UNKNOWN

    dosage = dm.dosage_form.upper()
    flammable = has_flammable_solvent(flammability_strings(dm, active_ingredients))

    # Aerosols / propelled forms (HFA inhalers, propellant sprays)
    if _contains_any(dosage, _AEROSOL_KEYWORDS):
        return ProductCategory.AEROSOL_PROPELLED

    # Inhalers — dry powder vs aerosol distinction
    if _contains_any(dosage, _INHALER_KEYWORDS):
        if "POWDER" in dosage:
            return ProductCategory.DRY_POWDER_INHALER
        return ProductCategory.AEROSOL_PROPELLED

    # Transdermal patches
    if _contains_any(dosage, _PATCH_KEYWORDS):
        return ProductCategory.TRANSDERMAL_PATCH

    # Lozenges / troches
    if _contains_any(dosage, _LOZENGE_KEYWORDS):
        return ProductCategory.LOZENGE

    # Ophthalmic / otic — eye drops, eye ointments, ear drops. Checked
    # before topicals because some eye ointments use dosage_form 'OINTMENT'
    # with route only signaled in the product name or active ingredient.
    if _is_ophthalmic_or_otic(dm, active_ingredients):
        return ProductCategory.OPHTHALMIC_OTIC

    # Topicals: creams, lotions, ointments, pastes, foams
    if _contains_any(dosage, _TOPICAL_DOSAGE_KEYWORDS):
        return ProductCategory.TOPICAL_ALCOHOL if flammable else ProductCategory.TOPICAL_NONFLAM

    # Gels (excluding softgel/gelcap capsules)
    if _contains_any(dosage, _GEL_KEYWORDS) and not _contains_any(dosage, _GEL_EXCLUSIONS):
        return ProductCategory.TOPICAL_ALCOHOL if flammable else ProductCategory.TOPICAL_NONFLAM

    # Non-propellant sprays (nasal saline, pump sprays)
    if _contains_any(dosage, _SPRAY_KEYWORDS):
        return ProductCategory.TOPICAL_ALCOHOL if flammable else ProductCategory.TOPICAL_NONFLAM

    # Dentifrices, mouthwashes (topical for SDS purposes)
    if _contains_any(dosage, _DENTIFRICE_KEYWORDS):
        return ProductCategory.TOPICAL_ALCOHOL if flammable else ProductCategory.TOPICAL_NONFLAM

    # Oral solids
    if _contains_any(dosage, _ORAL_SOLID_KEYWORDS):
        return ProductCategory.ORAL_SOLID

    # Oral liquids — last because LIQUID matches everything liquidy.
    # For flammable liquids, disambiguate topical from oral using the
    # product name (Purell HAND SANITIZER vs NyQuil COLD AND FLU).
    if _contains_any(dosage, _ORAL_LIQUID_KEYWORDS):
        if flammable and _is_topical_by_name(dm):
            return ProductCategory.TOPICAL_ALCOHOL
        return ProductCategory.ORAL_LIQUID_ALCOHOL if flammable else ProductCategory.ORAL_LIQUID_AQUEOUS

    return ProductCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Category profiles: per-category deterministic answers
# ---------------------------------------------------------------------------

# Common building blocks used by multiple non-hazmat categories.
_NONHAZMAT_SDS_DEFAULTS: dict[str, FieldDefault] = {
    "flash_point_c": FieldDefault("None, No Flash Point", 95, 70),
    "flash_point_method": FieldDefault("Not applicable/available", 95, 70),
    "boiling_point_c": FieldDefault("Not tested/Unknown", 85, 70),
    "ph_value": FieldDefault("Not tested/Unknown", 80, 70),
    "ph_mixture_extreme": FieldDefault("No", 90, 70),
    "water_solubility": FieldDefault("No data available", 70, 70),
    "transport_regulated": FieldDefault("No, not regulated", 95, 70),
    "un_number": FieldDefault("", 95, 70),
    "proper_shipping_name": FieldDefault("", 95, 70),
    "hazard_class": FieldDefault("Not Applicable", 95, 70),
    "packing_group": FieldDefault("Not Applicable", 95, 70),
    "rcra_classification": FieldDefault(
        "Not classified as D001 or D003 Hazardous Waste under RCRA", 95, 70,
    ),
}


def _with_overrides(base: dict[str, FieldDefault], overrides: dict[str, FieldDefault]) -> dict[str, FieldDefault]:
    out = dict(base)
    out.update(overrides)
    return out


CATEGORY_PROFILES: dict[ProductCategory, CategoryProfile] = {
    ProductCategory.ORAL_SOLID: CategoryProfile(
        category=ProductCategory.ORAL_SOLID,
        defaults=_NONHAZMAT_SDS_DEFAULTS,
        canonical_state="Capsule/Tablet",
    ),
    ProductCategory.ORAL_LIQUID_AQUEOUS: CategoryProfile(
        category=ProductCategory.ORAL_LIQUID_AQUEOUS,
        # Non-alcohol oral liquids: same non-hazmat profile, but pH and
        # water solubility may have real measured values; threshold them
        # lower so Perplexity's SDS quote wins more easily.
        defaults=_with_overrides(_NONHAZMAT_SDS_DEFAULTS, {
            "ph_value": FieldDefault("Not tested/Unknown", 70, 50),
            "water_solubility": FieldDefault("Miscible in water", 75, 60),
        }),
        canonical_state="Liquid",
    ),
    ProductCategory.ORAL_LIQUID_ALCOHOL: CategoryProfile(
        # Alcohol-bearing oral liquid (NyQuil, alcoholic cough syrups).
        # Flash point depends on alcohol percentage; let Perplexity drive
        # the safety fields entirely. Only fix RCRA when Perplexity is
        # clueless.
        category=ProductCategory.ORAL_LIQUID_ALCOHOL,
        defaults={
            "ph_mixture_extreme": FieldDefault("No", 80, 60),
        },
        canonical_state="Liquid",
    ),
    ProductCategory.TOPICAL_NONFLAM: CategoryProfile(
        category=ProductCategory.TOPICAL_NONFLAM,
        # Topical creams/ointments/lotions: same non-hazmat profile;
        # pH is sometimes measured for skin products, so leave it more
        # open to Perplexity than for solid tablets.
        defaults=_with_overrides(_NONHAZMAT_SDS_DEFAULTS, {
            "ph_value": FieldDefault("Not tested/Unknown", 65, 50),
        }),
        canonical_state=None,  # depends on dosage form (Cream/Ointment/etc.)
    ),
    ProductCategory.TOPICAL_ALCOHOL: CategoryProfile(
        # Hand sanitizer, alcohol-based topical gel/foam. Class IB
        # flammable liquid (ethanol 60-95%). UN1170 covers ethanol;
        # UN1219 covers isopropanol; UN1987 is the n.o.s. fallback.
        # Default to UN1170 since US hand sanitizer is overwhelmingly
        # ethanol-based.
        # Override threshold is 95 on safety-critical fields because
        # Perplexity will sometimes confidently misread the SDS - it has
        # been observed to return '>60C and <93C' for 70% ethanol with
        # confidence 80. The category answer is more reliable than the
        # SDS quote for the flammability triad.
        category=ProductCategory.TOPICAL_ALCOHOL,
        defaults={
            "flash_point_c": FieldDefault("<23C", 95, 95),
            "flash_point_method": FieldDefault("Closed cup method", 80, 60),
            "boiling_point_c": FieldDefault(">37.7C (99.9F)", 80, 60),
            "ph_value": FieldDefault("4 - 6.9", 65, 50),
            "ph_mixture_extreme": FieldDefault("No", 90, 70),
            "water_solubility": FieldDefault("Miscible in water", 90, 70),
            "transport_regulated": FieldDefault("Yes, Agree", 95, 95),
            "un_number": FieldDefault("UN1170", 80, 70),
            "proper_shipping_name": FieldDefault("Ethanol solution", 75, 60),
            "hazard_class": FieldDefault("3", 95, 90),
            "packing_group": FieldDefault("II", 75, 60),
            "rcra_classification": FieldDefault(
                "D001 Hazardous Waste under RCRA", 90, 80,
            ),
        },
        canonical_state=None,  # liquid hand sanitizer = Liquid; foam = Foam; gel = Gel
    ),
    ProductCategory.AEROSOL_PROPELLED: CategoryProfile(
        # HFA inhalers + propellant aerosols. UN1950 covers aerosols
        # broadly. Hazard class depends on propellant (2.1 flammable
        # vs 2.2 non-flammable HFA), so don't force it - Perplexity
        # and DOT lookup decide.
        category=ProductCategory.AEROSOL_PROPELLED,
        defaults={
            "transport_regulated": FieldDefault("Yes, Agree", 90, 80),
            "un_number": FieldDefault("UN1950", 85, 70),
            "proper_shipping_name": FieldDefault("Aerosols", 85, 70),
            "ph_mixture_extreme": FieldDefault("No", 75, 60),
        },
        canonical_state="Aerosol",
    ),
    ProductCategory.DRY_POWDER_INHALER: CategoryProfile(
        # DPIs (Advair Diskus, Spiriva HandiHaler, etc.) are non-pressurized
        # powder devices. Non-hazmat for transport. Same profile as oral
        # solid for SDS purposes.
        category=ProductCategory.DRY_POWDER_INHALER,
        defaults=_NONHAZMAT_SDS_DEFAULTS,
        canonical_state="Inhaler",
    ),
    ProductCategory.TRANSDERMAL_PATCH: CategoryProfile(
        category=ProductCategory.TRANSDERMAL_PATCH,
        defaults=_NONHAZMAT_SDS_DEFAULTS,
        canonical_state="Solid",
    ),
    ProductCategory.LOZENGE: CategoryProfile(
        category=ProductCategory.LOZENGE,
        defaults=_NONHAZMAT_SDS_DEFAULTS,
        canonical_state="Solid",
    ),
    ProductCategory.OPHTHALMIC_OTIC: CategoryProfile(
        # Sterile aqueous solutions for the eye or ear. Non-flammable,
        # non-regulated for transport. pH is measured and is typically
        # near neutral - leave open.
        category=ProductCategory.OPHTHALMIC_OTIC,
        defaults=_with_overrides(_NONHAZMAT_SDS_DEFAULTS, {
            "ph_value": FieldDefault("7 (Neutral)", 65, 50),
        }),
        canonical_state="Liquid (Sterile Diluent)",
    ),
    ProductCategory.UNKNOWN: CategoryProfile(
        # No deterministic answers known. Let Perplexity / PubChem drive
        # everything.
        category=ProductCategory.UNKNOWN,
        defaults={},
        canonical_state=None,
    ),
}


def profile_for(category: ProductCategory) -> CategoryProfile:
    return CATEGORY_PROFILES[category]
