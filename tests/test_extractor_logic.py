"""Deterministic tests for the extractor's cross-source logic.

Covers the defensive checks that catch systemic Perplexity failures:

- Active-ingredient parsing for combination products (Excedrin)
- Dosage-form vs picked-physical-state consistency check
- Non-hazmat normalization with the contradiction guard
- Multi-ingredient PubChem evidence aggregation
"""

from __future__ import annotations

import pytest

from src.extractor import (
    CRITICAL_FIELDS,
    TRANSPORT_DETAIL_FIELDS,
    TRANSPORT_NOT_REGULATED,
    _apply_inert_form_defaults,
    _canonical_physical_state,
    _check_dosage_form_consistency,
    _enrich_pubchem,
    _has_flammable_active,
    _is_inert_dose_form,
    _looks_regulated,
    _normalize_non_hazmat,
    _physical_state_matches_dosage,
    _pick_active_ingredients,
    is_non_hazmat,
)
from src.pubchem import PubChemData
from src.schema import DailyMedData, ExtractedField, SDSExtraction


def _blank_sds(**overrides) -> SDSExtraction:
    """Build a fully-populated SDSExtraction with sensible non-hazmat
    tablet defaults; tests override individual fields."""
    base = dict(
        product_type=ExtractedField(value="Prescription Pharmaceutical (Solid)", confidence=80),
        secondary_physical_state=ExtractedField(value="Capsule/Tablet", confidence=80),
        ph_mixture_extreme=ExtractedField(value="No", confidence=80),
        ph_value=ExtractedField(value="Not tested/Unknown", confidence=80),
        water_solubility=ExtractedField(value="Soluble in water", confidence=80),
        boiling_point_c=ExtractedField(value="Not tested/Unknown", confidence=80),
        flash_point_c=ExtractedField(value="None, No Flash Point", confidence=80),
        flash_point_method=ExtractedField(value="Not applicable/available", confidence=80),
        storage_temp_range=ExtractedField(value="Controlled room temperature of 20C to 25C", confidence=80),
        transport_regulated=ExtractedField(value="No, not regulated", confidence=80),
        un_number=ExtractedField(value="", confidence=80),
        proper_shipping_name=ExtractedField(value="", confidence=80),
        hazard_class=ExtractedField(value="", confidence=80),
        packing_group=ExtractedField(value="", confidence=80),
        rcra_classification=ExtractedField(value="Not classified as D001 or D003 Hazardous Waste under RCRA", confidence=80),
    )
    base.update(overrides)
    return SDSExtraction(**base)


class TestPickActiveIngredients:
    """Combination products like Excedrin must split correctly so PubChem
    queries every ingredient."""

    def test_excedrin_three_actives_with_oxford_comma(self) -> None:
        dm = DailyMedData(generic_name="acetaminophen, aspirin, and caffeine")
        actives = _pick_active_ingredients(dm)
        assert actives == ["acetaminophen", "aspirin", "caffeine"]

    def test_excedrin_three_actives_no_oxford(self) -> None:
        dm = DailyMedData(generic_name="acetaminophen, aspirin and caffeine")
        actives = _pick_active_ingredients(dm)
        assert actives == ["acetaminophen", "aspirin", "caffeine"]

    def test_semicolon_separator(self) -> None:
        dm = DailyMedData(generic_name="acetaminophen; aspirin; caffeine")
        assert _pick_active_ingredients(dm) == ["acetaminophen", "aspirin", "caffeine"]

    def test_explicit_active_ingredients_list_wins(self) -> None:
        dm = DailyMedData(
            active_ingredients=["IBUPROFEN", "PSEUDOEPHEDRINE HCL"],
            generic_name="should be ignored",
        )
        assert _pick_active_ingredients(dm) == ["IBUPROFEN", "PSEUDOEPHEDRINE HCL"]

    def test_dedupes_case_insensitively(self) -> None:
        dm = DailyMedData(generic_name="aspirin, Aspirin, ASPIRIN")
        assert _pick_active_ingredients(dm) == ["aspirin"]

    def test_empty_dailymed_returns_empty_list(self) -> None:
        assert _pick_active_ingredients(DailyMedData()) == []

    def test_single_active(self) -> None:
        dm = DailyMedData(generic_name="atorvastatin calcium")
        assert _pick_active_ingredients(dm) == ["atorvastatin calcium"]


class TestPhysicalStateMatchesDosage:
    """The hard-coded compatibility table between DailyMed dosage forms
    and SDSExtraction.secondary_physical_state enum values."""

    @pytest.mark.parametrize("dosage,picked,expected", [
        ("OINTMENT", "Ointment", True),
        ("OINTMENT", "Capsule/Tablet", False),
        ("CREAM", "Cream/Lotion", True),
        ("CREAM", "Solid", False),
        ("CREAM", "Ointment", False),
        ("LOTION", "Cream/Lotion", True),
        ("PASTE", "Paste", True),
        ("GEL", "Gel", True),
        ("FOAM", "Foam", True),
        ("SOLUTION", "Liquid", True),
        ("SOLUTION", "Capsule/Tablet", False),
        ("SUSPENSION", "Suspension", True),
        ("AEROSOL, METERED", "Aerosol", True),
        ("AEROSOL, METERED", "Capsule/Tablet", False),
        ("INHALANT", "Inhaler", True),
        ("TABLET, FILM COATED", "Capsule/Tablet", True),
        ("CAPSULE", "Capsule/Tablet", True),
        ("CAPSULE", "Solid containing liquid", True),
        ("POWDER", "Powder(s)", True),
        ("PATCH", "Solid", True),
    ])
    def test_dosage_form_to_physical_state_matrix(self, dosage, picked, expected) -> None:
        assert _physical_state_matches_dosage(dosage, picked) is expected

    def test_unknown_dosage_returns_none(self) -> None:
        assert _physical_state_matches_dosage("MYSTERY_FORM", "Liquid") is None

    def test_blank_inputs_return_none(self) -> None:
        assert _physical_state_matches_dosage("", "Liquid") is None
        assert _physical_state_matches_dosage("OINTMENT", "") is None


class TestDosageConsistencyCheck:
    """End-to-end check that mismatches surface as review reasons and
    demote confidence."""

    def test_desitin_ointment_picked_as_tablet_auto_corrects(self) -> None:
        # Wrong pick on a known dosage form is now auto-corrected to the
        # canonical enum value (Ointment) at confidence 90 so the row
        # doesn't flag for a problem we just solved. Evidence quote
        # records the correction.
        dm = DailyMedData(dosage_form="OINTMENT")
        sds = _blank_sds(
            secondary_physical_state=ExtractedField(value="Capsule/Tablet", confidence=80)
        )
        reasons = _check_dosage_form_consistency(sds, dm)
        assert reasons == []
        assert sds.secondary_physical_state.value == "Ointment"
        assert sds.secondary_physical_state.confidence == 90
        assert "Corrected" in (sds.secondary_physical_state.evidence_quote or "")
        assert "Capsule/Tablet" in (sds.secondary_physical_state.evidence_quote or "")
        assert "OINTMENT" in (sds.secondary_physical_state.evidence_quote or "")

    def test_correct_classification_passes_silently(self) -> None:
        dm = DailyMedData(dosage_form="OINTMENT")
        sds = _blank_sds(
            secondary_physical_state=ExtractedField(value="Ointment", confidence=85)
        )
        reasons = _check_dosage_form_consistency(sds, dm)
        assert reasons == []
        assert sds.secondary_physical_state.confidence == 85

    def test_no_dailymed_data_skips_check(self) -> None:
        sds = _blank_sds()
        assert _check_dosage_form_consistency(sds, None) == []
        assert _check_dosage_form_consistency(sds, DailyMedData()) == []


class TestIsNonHazmat:
    def test_explicit_not_regulated(self) -> None:
        assert is_non_hazmat("No, not regulated") is True

    def test_exception_path(self) -> None:
        assert is_non_hazmat(
            "No, not regulated for transportation due to an exception"
        ) is True

    def test_yes_agree_is_hazmat(self) -> None:
        assert is_non_hazmat("Yes, Agree") is False

    def test_empty_is_not_hazmat(self) -> None:
        assert is_non_hazmat("") is False
        assert is_non_hazmat(None) is False

    def test_constant_includes_both_no_paths(self) -> None:
        assert "No, not regulated" in TRANSPORT_NOT_REGULATED
        assert (
            "No, not regulated for transportation due to an exception"
            in TRANSPORT_NOT_REGULATED
        )


class TestNonHazmatNormalization:
    """The fill-with-Not-Applicable behavior for non-hazmat products."""

    def test_oral_tablet_normalizes_to_not_applicable(self) -> None:
        sds = _blank_sds()
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        reasons = _normalize_non_hazmat(sds, sources)
        assert reasons == []
        for fname in TRANSPORT_DETAIL_FIELDS:
            assert getattr(sds, fname).value == "Not Applicable"
            assert getattr(sds, fname).confidence >= 90
            assert "derived" in sources[fname]

    def test_aerosol_un_present_blocks_normalization(self) -> None:
        # Perplexity wrongly says non-regulated but also gave UN1950
        sds = _blank_sds(
            transport_regulated=ExtractedField(value="No, not regulated", confidence=80),
            un_number=ExtractedField(value="UN1950", confidence=85),
            hazard_class=ExtractedField(value="2.1", confidence=85),
        )
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        reasons = _normalize_non_hazmat(sds, sources)
        assert len(reasons) == 1
        assert "regulated" in reasons[0].lower()
        # UN value preserved
        assert sds.un_number.value == "UN1950"

    def test_aerosol_dosage_form_blocks_normalization(self) -> None:
        sds = _blank_sds(
            transport_regulated=ExtractedField(value="No, not regulated", confidence=80),
        )
        dm = DailyMedData(dosage_form="AEROSOL, METERED")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        reasons = _normalize_non_hazmat(sds, sources, dm)
        assert len(reasons) == 1
        # Fields stay empty (not normalized)
        assert sds.un_number.value == ""

    def test_yes_agree_skips_normalization_entirely(self) -> None:
        sds = _blank_sds(
            transport_regulated=ExtractedField(value="Yes, Agree", confidence=85),
            un_number=ExtractedField(value="UN1950", confidence=85),
        )
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        reasons = _normalize_non_hazmat(sds, sources)
        assert reasons == []
        assert sds.un_number.value == "UN1950"


class TestLooksRegulated:
    """The gate that prevents normalization of misclassified aerosols."""

    def test_un_number_present(self) -> None:
        sds = _blank_sds(un_number=ExtractedField(value="UN1950", confidence=85))
        assert _looks_regulated(sds, None) is True

    def test_aerosol_dosage_form(self) -> None:
        sds = _blank_sds()
        dm = DailyMedData(dosage_form="AEROSOL, METERED")
        assert _looks_regulated(sds, dm) is True

    def test_inhaler_dosage_form(self) -> None:
        sds = _blank_sds()
        dm = DailyMedData(dosage_form="INHALER")
        assert _looks_regulated(sds, dm) is True

    def test_oral_tablet_not_regulated(self) -> None:
        sds = _blank_sds()
        dm = DailyMedData(dosage_form="TABLET, FILM COATED")
        assert _looks_regulated(sds, dm) is False

    def test_topical_ointment_not_regulated(self) -> None:
        sds = _blank_sds()
        dm = DailyMedData(dosage_form="OINTMENT")
        assert _looks_regulated(sds, dm) is False


class TestEnrichPubchem:
    """Multi-ingredient PubChem aggregation produces the right evidence
    and confidence shifts."""

    def test_all_ingredients_supportive_bumps_confidence(self) -> None:
        sds = _blank_sds(
            flash_point_c=ExtractedField(value="None, No Flash Point", confidence=70)
        )
        records = [
            PubChemData(name="acetaminophen", cid=1983, flash_point_text="169 °C", compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/1983"),
            PubChemData(name="aspirin", cid=2244, flash_point_text="482 °F", compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/2244"),
            PubChemData(name="caffeine", cid=2519, flash_point_text="178 °C", compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/2519"),
        ]
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        extra_review: list[str] = []
        _enrich_pubchem(sds, records, sources, extra_review)

        assert sds.flash_point_c.confidence == 90
        assert "pubchem" in sources["flash_point_c"]
        assert "Consistent" in (sds.flash_point_c.evidence_quote or "")
        assert extra_review == []

    def test_one_ingredient_disagrees_drops_confidence(self) -> None:
        sds = _blank_sds(
            flash_point_c=ExtractedField(value="None, No Flash Point", confidence=80)
        )
        records = [PubChemData(name="ethanol", cid=702, flash_point_text="13 °C", compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/702")]
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        extra_review: list[str] = []
        _enrich_pubchem(sds, records, sources, extra_review)

        assert sds.flash_point_c.confidence == 50
        assert "pubchem (disagrees)" in sources["flash_point_c"]
        assert any("disagrees" in r.lower() for r in extra_review)

    def test_mixed_data_coverage_note_lists_missing(self) -> None:
        # Acetaminophen and caffeine have no flash point in PubChem;
        # aspirin does. The evidence should call out which is which.
        sds = _blank_sds(
            flash_point_c=ExtractedField(value="None, No Flash Point", confidence=70)
        )
        records = [
            PubChemData(name="acetaminophen", cid=1983, flash_point_text=None, compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/1983"),
            PubChemData(name="aspirin", cid=2244, flash_point_text="482 °F", compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/2244"),
            PubChemData(name="caffeine", cid=2519, flash_point_text=None, compound_url="https://pubchem.ncbi.nlm.nih.gov/compound/2519"),
        ]
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}
        extra_review: list[str] = []
        _enrich_pubchem(sds, records, sources, extra_review)

        evidence = sds.flash_point_c.evidence_quote or ""
        assert "checked 3 active ingredients" in evidence
        assert "acetaminophen" in evidence
        assert "caffeine" in evidence
        assert "no flash point data for" in evidence


class TestInertDoseFormDetection:
    """The gate that controls whether the deterministic non-hazmat defaults
    fire. Should match common solid/liquid/topical dose forms and skip
    aerosols + alcohol-bearing products."""

    @pytest.mark.parametrize("dosage_form,expected", [
        ("TABLET, FILM COATED", True),
        ("CAPSULE", True),
        ("CAPLET", True),
        ("SOFTGEL", True),
        ("POWDER, FOR ORAL SUSPENSION", True),
        ("GRANULES", True),
        ("CREAM", True),
        ("OINTMENT", True),
        ("LOTION", True),
        ("GEL", True),
        ("PASTE, DENTIFRICE", True),
        ("SOLUTION", True),
        ("SUSPENSION", True),
        ("SYRUP", True),
        ("PATCH", True),
        ("LOZENGE", True),
        ("AEROSOL, METERED", False),
        ("INHALER", False),
        ("INHALANT", False),
        ("SPRAY", False),
        ("METERED-DOSE INHALER", False),
        ("", False),
    ])
    def test_dosage_form_matrix(self, dosage_form, expected) -> None:
        dm = DailyMedData(dosage_form=dosage_form) if dosage_form else DailyMedData()
        assert _is_inert_dose_form(dm, []) is expected

    def test_none_dailymed_returns_false(self) -> None:
        assert _is_inert_dose_form(None, []) is False

    def test_mouthwash_with_alcohol_active_is_blocked(self) -> None:
        dm = DailyMedData(dosage_form="MOUTHWASH")  # not in inert keyword list
        assert _is_inert_dose_form(dm, ["ethanol"]) is False

    def test_oral_liquid_with_alcohol_active_is_blocked(self) -> None:
        # Hand sanitizer is a "LIQUID" but contains ethanol/isopropanol
        dm = DailyMedData(dosage_form="LIQUID")
        assert _is_inert_dose_form(dm, ["isopropyl alcohol"]) is False
        assert _is_inert_dose_form(dm, ["ETHANOL"]) is False

    def test_oral_liquid_without_alcohol_is_inert(self) -> None:
        dm = DailyMedData(dosage_form="SOLUTION")
        assert _is_inert_dose_form(dm, ["acetaminophen"]) is True


class TestFlammableActiveDetection:
    """Checks the alcohol/solvent guard that protects mouthwash, hand
    sanitizer, and oral liquids with ethanol from getting wrong defaults."""

    @pytest.mark.parametrize("strings,expected", [
        # Explicit flammable tokens
        (["ethanol"], True),
        (["ETHANOL"], True),
        (["ethyl alcohol"], True),
        (["isopropanol"], True),
        (["isopropyl alcohol"], True),
        (["Acetone"], True),
        (["methanol"], True),
        (["denatured alcohol"], True),
        (["SD Alcohol 40"], True),
        (["acetaminophen", "ethanol"], True),
        # Bare 'alcohol' (e.g., DailyMed generic_name for hand sanitizer)
        (["ALCOHOL"], True),
        (["alcohol"], True),
        # Hand sanitizer / sanitizing keyword fires even without an
        # explicit alcohol token, in case DailyMed encodes the product
        # only in the product name.
        (["PURELL HAND SANITIZER"], True),
        # Fatty alcohols and polymers are NOT flammable
        (["cetyl alcohol"], False),
        (["stearyl alcohol"], False),
        (["cetostearyl alcohol"], False),
        (["benzyl alcohol"], False),
        (["polyvinyl alcohol"], False),
        (["lanolin alcohol"], False),
        (["cetearyl alcohol", "water"], False),
        # Real active ingredients with no flammable solvent
        (["acetaminophen"], False),
        (["ibuprofen"], False),
        (["aspirin", "caffeine", "acetaminophen"], False),
        (["nicotine"], False),
        ([], False),
    ])
    def test_flammable_active_detection(self, strings, expected) -> None:
        assert _has_flammable_active(strings) is expected


class TestPurellRegression:
    """Bug 1: hand sanitizer with DailyMed generic_name='ALCOHOL' and
    empty active_ingredients was incorrectly classified as inert and
    stamped with 'None, No Flash Point'. The flammable check now reads
    generic_name + product_name on top of active_ingredients."""

    def test_purell_with_alcohol_generic_name_is_not_inert(self) -> None:
        # Mirrors the actual DailyMed response for "Purell hand sanitizer"
        dm = DailyMedData(
            product_name="PURELL ADVANCED HAND SANITIZER FRAGRANCE FREE",
            generic_name="ALCOHOL",
            dosage_form="LIQUID",
            active_ingredients=[],
        )
        assert _is_inert_dose_form(dm, []) is False

    def test_listerine_with_alcohol_in_product_name(self) -> None:
        # Mouthwash where DailyMed might encode alcohol only in product
        # name (not always returned as an active)
        dm = DailyMedData(
            product_name="LISTERINE COOL MINT ANTISEPTIC MOUTHWASH",
            generic_name="EUCALYPTOL, MENTHOL, METHYL SALICYLATE, THYMOL",
            dosage_form="LIQUID",
            active_ingredients=["eucalyptol", "menthol", "methyl salicylate", "thymol"],
        )
        # Without ethanol-in-strings, this would be treated as inert.
        # In reality DailyMed returns ethanol as an active for Listerine,
        # so this test documents the "no ethanol surfaced anywhere" path.
        # An honest gap: if DailyMed truly hides the alcohol, we can't
        # catch it from name alone. The Purell case is the realistic one.
        assert _is_inert_dose_form(dm, ["eucalyptol", "menthol"]) is True

    def test_hand_sanitizer_keyword_in_product_name_blocks(self) -> None:
        # Defense-in-depth: 'hand sanitizer' or 'sanitizing' in the name
        # blocks the inert defaults even when no active is surfaced.
        dm = DailyMedData(
            product_name="GENERIC HAND SANITIZER",
            generic_name="",
            dosage_form="LIQUID",
        )
        assert _is_inert_dose_form(dm, []) is False


class TestCanonicalPhysicalState:
    """Bug 2 helper: each dosage form keyword maps to exactly one canonical
    enum value used for auto-correction."""

    @pytest.mark.parametrize("dosage_form,expected", [
        ("TABLET", "Capsule/Tablet"),
        ("TABLET, FILM COATED", "Capsule/Tablet"),
        ("CAPSULE", "Capsule/Tablet"),
        ("SOFTGEL", "Capsule/Tablet"),
        ("PATCH, EXTENDED RELEASE", "Solid"),
        ("PATCH", "Solid"),
        ("OINTMENT", "Ointment"),
        ("CREAM", "Cream/Lotion"),
        ("LOTION", "Cream/Lotion"),
        ("PASTE, DENTIFRICE", "Paste"),
        ("GEL, DENTIFRICE", "Gel"),
        ("FOAM", "Foam"),
        ("SOLUTION", "Liquid"),
        ("SUSPENSION", "Suspension"),
        ("SYRUP", "Liquid"),
        ("AEROSOL, METERED", "Aerosol"),
        ("INHALER", "Inhaler"),
        ("POWDER", "Powder(s)"),
        ("LOZENGE", "Solid"),
    ])
    def test_canonical_lookup(self, dosage_form, expected) -> None:
        assert _canonical_physical_state(dosage_form) == expected

    def test_unknown_dosage_returns_none(self) -> None:
        assert _canonical_physical_state("MYSTERY_FORM") is None
        assert _canonical_physical_state("") is None


class TestNicodermRegression:
    """Bug 2: Perplexity picked 'Capsule/Tablet' for Nicoderm CQ's PATCH
    dosage form. The consistency check now auto-corrects to 'Solid' at
    confidence 90 instead of demoting to 50 and flagging."""

    def test_patch_picked_as_tablet_auto_corrects_to_solid(self) -> None:
        dm = DailyMedData(
            product_name="NICODERM CQ",
            generic_name="NICOTINE",
            dosage_form="PATCH, EXTENDED RELEASE",
        )
        sds = _blank_sds(
            secondary_physical_state=ExtractedField(value="Capsule/Tablet", confidence=50)
        )
        reasons = _check_dosage_form_consistency(sds, dm)

        # No review reason because we corrected it
        assert reasons == []
        # Value replaced with the canonical for PATCH
        assert sds.secondary_physical_state.value == "Solid"
        # Confidence raised above the review threshold
        assert sds.secondary_physical_state.confidence == 90

    def test_aerosol_picked_as_tablet_auto_corrects_to_aerosol(self) -> None:
        # Same fix protects albuterol-style mistakes
        dm = DailyMedData(dosage_form="AEROSOL, METERED")
        sds = _blank_sds(
            secondary_physical_state=ExtractedField(value="Capsule/Tablet", confidence=50)
        )
        reasons = _check_dosage_form_consistency(sds, dm)
        assert reasons == []
        assert sds.secondary_physical_state.value == "Aerosol"
        assert sds.secondary_physical_state.confidence == 90

    def test_correct_value_passes_through_unchanged(self) -> None:
        dm = DailyMedData(dosage_form="PATCH, EXTENDED RELEASE")
        sds = _blank_sds(
            secondary_physical_state=ExtractedField(value="Solid", confidence=85)
        )
        reasons = _check_dosage_form_consistency(sds, dm)
        assert reasons == []
        # Untouched
        assert sds.secondary_physical_state.value == "Solid"
        assert sds.secondary_physical_state.confidence == 85


class TestApplyInertFormDefaults:
    """The fix for the false-flag problem: tablets/capsules/non-alcohol
    liquids/topicals should get deterministic high-confidence answers on
    the eight SDS Section 9 / transport / RCRA fields, regardless of how
    sparse the SDS is."""

    @staticmethod
    def _low_conf_sds() -> SDSExtraction:
        """Simulates what Perplexity returns when no SDS data is found:
        defaulted values at confidence 50, below the review threshold."""
        return SDSExtraction(
            product_type=ExtractedField(value="Prescription Pharmaceutical (Solid)", confidence=80),
            secondary_physical_state=ExtractedField(value="Capsule/Tablet", confidence=80),
            ph_mixture_extreme=ExtractedField(value="No", confidence=50),
            ph_value=ExtractedField(value=None, confidence=40),
            water_solubility=ExtractedField(value=None, confidence=40),
            boiling_point_c=ExtractedField(value=None, confidence=40),
            flash_point_c=ExtractedField(value="None, No Flash Point", confidence=50),
            flash_point_method=ExtractedField(value=None, confidence=40),
            storage_temp_range=ExtractedField(value="Controlled room temperature of 20C to 25C", confidence=80),
            transport_regulated=ExtractedField(value="No, not regulated", confidence=50),
            un_number=ExtractedField(value="", confidence=80),
            proper_shipping_name=ExtractedField(value="", confidence=80),
            hazard_class=ExtractedField(value="", confidence=80),
            packing_group=ExtractedField(value="", confidence=80),
            rcra_classification=ExtractedField(
                value="Not classified as D001 or D003 Hazardous Waste under RCRA",
                confidence=50,
            ),
        )

    def test_tylenol_tablet_gets_high_confidence_defaults(self) -> None:
        sds = self._low_conf_sds()
        dm = DailyMedData(dosage_form="TABLET, FILM COATED")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["acetaminophen"], sources)

        # All eight fields now meet or exceed the review threshold of 60
        assert sds.flash_point_c.value == "None, No Flash Point"
        assert sds.flash_point_c.confidence == 95
        assert sds.flash_point_method.value == "Not applicable/available"
        assert sds.flash_point_method.confidence == 95
        assert sds.boiling_point_c.value == "Not tested/Unknown"
        assert sds.boiling_point_c.confidence == 90
        assert sds.ph_value.value == "Not tested/Unknown"
        assert sds.ph_value.confidence == 85
        assert sds.ph_mixture_extreme.value == "No"
        assert sds.ph_mixture_extreme.confidence == 90
        assert sds.water_solubility.value == "No data available"
        assert sds.water_solubility.confidence == 70
        assert sds.transport_regulated.value == "No, not regulated"
        assert sds.transport_regulated.confidence == 95
        assert sds.rcra_classification.confidence == 95
        # Each overridden field is marked as derived in the source map
        for fname in (
            "flash_point_c", "flash_point_method", "boiling_point_c",
            "ph_value", "ph_mixture_extreme", "water_solubility",
            "transport_regulated", "rcra_classification",
        ):
            assert "derived" in sources[fname], f"{fname} missing derived source"

    def test_ibuprofen_tablet_gets_defaults(self) -> None:
        sds = self._low_conf_sds()
        dm = DailyMedData(dosage_form="TABLET, COATED")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["ibuprofen"], sources)

        assert sds.flash_point_c.confidence == 95
        assert sds.transport_regulated.confidence == 95

    def test_advil_capsule_gets_defaults(self) -> None:
        sds = self._low_conf_sds()
        dm = DailyMedData(dosage_form="CAPSULE, LIQUID FILLED")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["ibuprofen"], sources)

        assert sds.flash_point_c.confidence == 95
        assert sds.rcra_classification.confidence == 95

    def test_high_confidence_perplexity_value_is_preserved(self) -> None:
        # Hypothetical: Perplexity found a real SDS for a tablet and quoted
        # a non-default flash point at confidence 90. The override must NOT
        # clobber that real datum.
        sds = self._low_conf_sds()
        sds.flash_point_c = ExtractedField(
            value=">=93C and <=815C", confidence=92,
            evidence_quote="SDS Section 9: Flash point 200°C",
        )
        dm = DailyMedData(dosage_form="TABLET")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["acetaminophen"], sources)

        assert sds.flash_point_c.value == ">=93C and <=815C"
        assert sds.flash_point_c.confidence == 92
        assert "derived" not in sources["flash_point_c"]
        # Other low-conf fields still get defaulted
        assert sds.transport_regulated.confidence == 95

    def test_aerosol_inhaler_skips_defaults(self) -> None:
        # Albuterol HFA is regulated for transport (UN1950) and we must
        # not stamp it as "not regulated".
        sds = self._low_conf_sds()
        dm = DailyMedData(dosage_form="AEROSOL, METERED")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["albuterol"], sources)

        # Confidence stays at the original 50; no derived sources added
        assert sds.transport_regulated.confidence == 50
        assert "derived" not in sources["transport_regulated"]
        assert sds.flash_point_c.confidence == 50

    def test_alcohol_mouthwash_skips_defaults(self) -> None:
        # Listerine-style mouthwash with ethanol must keep SDS-driven
        # answers so the actual flash point is preserved.
        sds = self._low_conf_sds()
        dm = DailyMedData(dosage_form="LIQUID")
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, dm, ["eucalyptol", "ethanol"], sources)

        assert sds.flash_point_c.confidence == 50
        assert "derived" not in sources["flash_point_c"]

    def test_no_dailymed_skips_defaults(self) -> None:
        sds = self._low_conf_sds()
        sources = {f: ["perplexity"] for f in SDSExtraction.model_fields.keys()}

        _apply_inert_form_defaults(sds, None, [], sources)
        _apply_inert_form_defaults(sds, DailyMedData(), [], sources)

        assert sds.flash_point_c.confidence == 50
        assert "derived" not in sources["flash_point_c"]


class TestCriticalFieldsConstant:
    """Make sure CRITICAL_FIELDS only includes the five UL portal-critical
    fields. Adding a new field here changes flagging behavior across the
    whole app, so we lock it down with a test."""

    def test_critical_fields_exact_set(self) -> None:
        assert set(CRITICAL_FIELDS) == {
            "product_type",
            "secondary_physical_state",
            "flash_point_c",
            "transport_regulated",
            "rcra_classification",
        }
