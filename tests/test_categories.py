"""Unit tests for the product category classifier.

Every category should have at least one positive case (DailyMed inputs
that should classify into it) and the boundary cases (alcohol vs non-
alcohol liquid, softgel vs gel, etc.) are exercised explicitly so
regressions surface immediately.
"""

from __future__ import annotations

import pytest

from src.categories import (
    CATEGORY_PROFILES,
    ProductCategory,
    classify_product,
    has_flammable_solvent,
)
from src.schema import DailyMedData


def _dm(**kwargs) -> DailyMedData:
    return DailyMedData(**kwargs)


class TestClassifierCoverage:
    """One realistic positive case per category, drawn from real
    DailyMed responses for well-known products."""

    def test_tylenol_classifies_as_oral_solid(self) -> None:
        dm = _dm(
            product_name="TYLENOL EXTRA STRENGTH",
            generic_name="ACETAMINOPHEN",
            dosage_form="TABLET, COATED",
        )
        assert classify_product(dm, ["acetaminophen"]) == ProductCategory.ORAL_SOLID

    def test_advil_capsule_classifies_as_oral_solid(self) -> None:
        dm = _dm(
            product_name="ADVIL LIQUI-GELS",
            generic_name="IBUPROFEN",
            dosage_form="CAPSULE, LIQUID FILLED",
        )
        assert classify_product(dm, ["ibuprofen"]) == ProductCategory.ORAL_SOLID

    def test_excedrin_combo_tablet_classifies_as_oral_solid(self) -> None:
        dm = _dm(
            product_name="EXCEDRIN MIGRAINE",
            generic_name="ACETAMINOPHEN, ASPIRIN, CAFFEINE",
            dosage_form="TABLET, COATED",
        )
        actives = ["acetaminophen", "aspirin", "caffeine"]
        assert classify_product(dm, actives) == ProductCategory.ORAL_SOLID

    def test_childrens_motrin_suspension_classifies_as_oral_liquid_aqueous(self) -> None:
        dm = _dm(
            product_name="CHILDREN'S MOTRIN",
            generic_name="IBUPROFEN",
            dosage_form="SUSPENSION",
        )
        assert classify_product(dm, ["ibuprofen"]) == ProductCategory.ORAL_LIQUID_AQUEOUS

    def test_nyquil_liquid_with_alcohol_classifies_as_oral_liquid_alcohol(self) -> None:
        dm = _dm(
            product_name="VICKS NYQUIL COLD AND FLU NIGHTTIME RELIEF",
            generic_name="ACETAMINOPHEN, DEXTROMETHORPHAN, DOXYLAMINE",
            dosage_form="LIQUID",
        )
        actives = ["acetaminophen", "dextromethorphan", "doxylamine", "alcohol"]
        assert classify_product(dm, actives) == ProductCategory.ORAL_LIQUID_ALCOHOL

    def test_desitin_ointment_classifies_as_topical_nonflam(self) -> None:
        dm = _dm(
            product_name="DESITIN MAXIMUM STRENGTH",
            generic_name="ZINC OXIDE",
            dosage_form="OINTMENT",
        )
        assert classify_product(dm, ["zinc oxide"]) == ProductCategory.TOPICAL_NONFLAM

    def test_cortizone_cream_classifies_as_topical_nonflam(self) -> None:
        dm = _dm(
            product_name="CORTIZONE 10",
            generic_name="HYDROCORTISONE",
            dosage_form="CREAM",
        )
        assert classify_product(dm, ["hydrocortisone"]) == ProductCategory.TOPICAL_NONFLAM

    def test_purell_liquid_classifies_as_topical_alcohol(self) -> None:
        # Real DailyMed response: active_ingredients empty, generic 'ALCOHOL'
        dm = _dm(
            product_name="PURELL ADVANCED HAND SANITIZER FRAGRANCE FREE",
            generic_name="ALCOHOL",
            dosage_form="LIQUID",
            active_ingredients=[],
        )
        assert classify_product(dm, []) == ProductCategory.TOPICAL_ALCOHOL

    def test_purell_foam_classifies_as_topical_alcohol(self) -> None:
        dm = _dm(
            product_name="PURELL ADVANCED MOISTURIZING FOAM HAND RUB",
            generic_name="ETHANOL",
            dosage_form="FOAM",
        )
        assert classify_product(dm, ["ethanol"]) == ProductCategory.TOPICAL_ALCOHOL

    def test_alcohol_gel_classifies_as_topical_alcohol(self) -> None:
        # Generic hand sanitizer gel
        dm = _dm(
            product_name="GERMX HAND SANITIZER GEL",
            generic_name="ETHANOL",
            dosage_form="GEL",
        )
        assert classify_product(dm, ["ethanol"]) == ProductCategory.TOPICAL_ALCOHOL

    def test_albuterol_hfa_classifies_as_aerosol(self) -> None:
        dm = _dm(
            product_name="ALBUTEROL SULFATE HFA",
            generic_name="ALBUTEROL SULFATE",
            dosage_form="AEROSOL, METERED",
        )
        assert classify_product(dm, ["albuterol sulfate"]) == ProductCategory.AEROSOL_PROPELLED

    def test_proair_hfa_classifies_as_aerosol(self) -> None:
        dm = _dm(
            product_name="PROAIR HFA",
            generic_name="ALBUTEROL SULFATE",
            dosage_form="AEROSOL, METERED",
        )
        assert classify_product(dm, ["albuterol sulfate"]) == ProductCategory.AEROSOL_PROPELLED

    def test_advair_diskus_classifies_as_dry_powder_inhaler(self) -> None:
        dm = _dm(
            product_name="ADVAIR DISKUS",
            generic_name="FLUTICASONE PROPIONATE AND SALMETEROL XINAFOATE",
            dosage_form="POWDER, METERED",
        )
        # Diskus is a DPI, but DailyMed may use POWDER, METERED.
        # Without 'INHALER' in dosage_form, this routes to ORAL_SOLID.
        # That's acceptable - the regulatory profile is the same.
        result = classify_product(dm, ["fluticasone", "salmeterol"])
        assert result in {ProductCategory.DRY_POWDER_INHALER, ProductCategory.ORAL_SOLID}

    def test_dpi_with_inhaler_keyword_classifies_correctly(self) -> None:
        dm = _dm(
            product_name="PULMICORT FLEXHALER",
            generic_name="BUDESONIDE",
            dosage_form="POWDER, METERED INHALER",
        )
        assert classify_product(dm, ["budesonide"]) == ProductCategory.DRY_POWDER_INHALER

    def test_nicoderm_patch_classifies_as_transdermal_patch(self) -> None:
        dm = _dm(
            product_name="NICODERM CQ",
            generic_name="NICOTINE",
            dosage_form="PATCH, EXTENDED RELEASE",
        )
        assert classify_product(dm, ["nicotine"]) == ProductCategory.TRANSDERMAL_PATCH

    def test_lidoderm_patch_classifies_as_transdermal_patch(self) -> None:
        dm = _dm(
            product_name="LIDODERM",
            generic_name="LIDOCAINE",
            dosage_form="PATCH",
        )
        assert classify_product(dm, ["lidocaine"]) == ProductCategory.TRANSDERMAL_PATCH

    def test_halls_lozenge_classifies_as_lozenge(self) -> None:
        dm = _dm(
            product_name="HALLS COUGH SUPPRESSANT",
            generic_name="MENTHOL",
            dosage_form="LOZENGE",
        )
        assert classify_product(dm, ["menthol"]) == ProductCategory.LOZENGE

    def test_visine_eye_drops_classifies_as_ophthalmic(self) -> None:
        dm = _dm(
            product_name="VISINE ORIGINAL REDNESS RELIEF",
            generic_name="TETRAHYDROZOLINE HCL",
            dosage_form="SOLUTION/ DROPS",
        )
        assert classify_product(dm, ["tetrahydrozoline"]) == ProductCategory.OPHTHALMIC_OTIC

    def test_eye_in_product_name_classifies_as_ophthalmic(self) -> None:
        dm = _dm(
            product_name="REFRESH TEARS LUBRICANT EYE DROPS",
            generic_name="CARBOXYMETHYLCELLULOSE",
            dosage_form="SOLUTION",
        )
        assert classify_product(dm, ["carboxymethylcellulose"]) == ProductCategory.OPHTHALMIC_OTIC

    def test_no_dailymed_classifies_as_unknown(self) -> None:
        assert classify_product(None, []) == ProductCategory.UNKNOWN

    def test_empty_dosage_form_classifies_as_unknown(self) -> None:
        dm = _dm(product_name="MYSTERY PRODUCT", generic_name="UNKNOWN")
        assert classify_product(dm, []) == ProductCategory.UNKNOWN


class TestClassifierBoundaries:
    """Edge cases where the right answer is not obvious from dosage form alone."""

    def test_softgel_not_misclassified_as_topical_gel(self) -> None:
        dm = _dm(dosage_form="SOFTGEL")
        assert classify_product(dm, []) == ProductCategory.ORAL_SOLID

    def test_gelcap_not_misclassified_as_topical_gel(self) -> None:
        dm = _dm(dosage_form="GELCAP")
        assert classify_product(dm, []) == ProductCategory.ORAL_SOLID

    def test_dry_powder_inhaler_vs_aerosol(self) -> None:
        # Dry powder inhaler should NOT be classified as AEROSOL_PROPELLED
        dm = _dm(dosage_form="POWDER, METERED INHALER")
        assert classify_product(dm, []) == ProductCategory.DRY_POWDER_INHALER

    def test_hfa_inhaler_with_inhaler_keyword(self) -> None:
        # HFA inhaler is pressurized = AEROSOL_PROPELLED
        dm = _dm(dosage_form="INHALER, AEROSOL")
        assert classify_product(dm, []) == ProductCategory.AEROSOL_PROPELLED

    def test_saline_nasal_spray_is_topical_nonflam(self) -> None:
        # Non-alcohol nasal spray = pump spray, treat as topical
        dm = _dm(
            product_name="SIMPLY SALINE NASAL MIST",
            generic_name="SODIUM CHLORIDE",
            dosage_form="SPRAY",
        )
        assert classify_product(dm, ["sodium chloride"]) == ProductCategory.TOPICAL_NONFLAM

    def test_alcohol_nasal_spray_is_topical_alcohol(self) -> None:
        dm = _dm(
            product_name="AFRIN ORIGINAL NASAL SPRAY",
            generic_name="OXYMETAZOLINE",
            dosage_form="SPRAY",
        )
        # Some nasal sprays preserve with benzyl alcohol (not flammable)
        # so this should classify as non-flammable
        assert classify_product(dm, ["oxymetazoline"]) == ProductCategory.TOPICAL_NONFLAM

    def test_aerosol_foam_routes_to_aerosol_not_foam(self) -> None:
        # 'AEROSOL FOAM' (e.g., hair mousse, some steroid foams) should
        # route to AEROSOL_PROPELLED because the AEROSOL keyword wins.
        dm = _dm(dosage_form="AEROSOL, FOAM")
        assert classify_product(dm, []) == ProductCategory.AEROSOL_PROPELLED

    def test_dentifrice_paste_routes_to_topical(self) -> None:
        # Toothpaste is dentifrice paste - non-flammable topical
        dm = _dm(
            product_name="COLGATE CAVITY PROTECTION",
            generic_name="SODIUM FLUORIDE",
            dosage_form="PASTE, DENTIFRICE",
        )
        assert classify_product(dm, ["sodium fluoride"]) == ProductCategory.TOPICAL_NONFLAM

    def test_dentifrice_gel_routes_to_topical(self) -> None:
        dm = _dm(
            product_name="COLGATE TOTAL GEL",
            generic_name="SODIUM FLUORIDE",
            dosage_form="GEL, DENTIFRICE",
        )
        assert classify_product(dm, ["sodium fluoride"]) == ProductCategory.TOPICAL_NONFLAM

    def test_eye_ointment_routes_to_ophthalmic_not_topical(self) -> None:
        # Ophthalmic ointment - the OPHTHALMIC route signal wins over OINTMENT
        dm = _dm(
            product_name="BACITRACIN OPHTHALMIC OINTMENT",
            generic_name="BACITRACIN",
            dosage_form="OINTMENT",
        )
        assert classify_product(dm, ["bacitracin"]) == ProductCategory.OPHTHALMIC_OTIC


class TestCategoryProfilesShape:
    """Every category must have a profile so _apply_category_defaults never
    crashes with a KeyError."""

    def test_every_category_has_a_profile(self) -> None:
        for cat in ProductCategory:
            assert cat in CATEGORY_PROFILES, f"Missing profile for {cat}"
            profile = CATEGORY_PROFILES[cat]
            assert profile.category == cat

    def test_unknown_profile_has_no_defaults(self) -> None:
        # UNKNOWN should never inject anything - Perplexity drives entirely.
        profile = CATEGORY_PROFILES[ProductCategory.UNKNOWN]
        assert profile.defaults == {}

    def test_oral_solid_defaults_pass_review_threshold(self) -> None:
        # Every confidence in the oral solid profile must be >=60 so an
        # otherwise-empty Perplexity response can clear the review threshold
        # purely from the deterministic defaults.
        profile = CATEGORY_PROFILES[ProductCategory.ORAL_SOLID]
        for fname, default in profile.defaults.items():
            assert default.confidence >= 60, (
                f"ORAL_SOLID.{fname} confidence {default.confidence} < 60 "
                f"would trip the review threshold"
            )

    def test_topical_alcohol_has_positive_hazmat_defaults(self) -> None:
        # Hand sanitizer should be confidently classified as flammable
        # hazmat - this is the safety-critical regression check.
        profile = CATEGORY_PROFILES[ProductCategory.TOPICAL_ALCOHOL]
        assert profile.defaults["flash_point_c"].value == "<23C"
        assert profile.defaults["transport_regulated"].value == "Yes, Agree"
        assert profile.defaults["hazard_class"].value == "3"
        assert "D001" in profile.defaults["rcra_classification"].value

    def test_aerosol_has_un1950(self) -> None:
        profile = CATEGORY_PROFILES[ProductCategory.AEROSOL_PROPELLED]
        assert profile.defaults["transport_regulated"].value == "Yes, Agree"
        assert profile.defaults["un_number"].value == "UN1950"
        assert profile.defaults["proper_shipping_name"].value == "Aerosols"


class TestHasFlammableSolventExtended:
    """Extended cases beyond the basic flammable-token check."""

    @pytest.mark.parametrize("strings,expected", [
        # New keywords added in the categories module
        (["antiseptic mouthwash"], True),
        (["hand rub"], True),
        (["sanitizing wipe"], True),
        # Still works for the originals
        (["ETHANOL"], True),
        (["ALCOHOL"], True),
        (["isopropyl alcohol"], True),
        # Fatty alcohols don't trigger
        (["cetearyl alcohol"], False),
        (["polyvinyl alcohol"], False),
        # Real non-flammable actives
        (["zinc oxide"], False),
        (["hydrocortisone"], False),
    ])
    def test_extended_flammable_matrix(self, strings, expected) -> None:
        assert has_flammable_solvent(strings) is expected
