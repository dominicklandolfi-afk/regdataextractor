"""Deterministic tests for DailyMed title parsing and hit selection.

These run without network access and cover the failure modes we have
hit in production:

- "X KIT (active) KIT" titles producing duplicated 'KIT KIT' dosage form
- DailyMed returning unrelated SPLs as the first hit when the brand
  name is buried in a kit or gift set
- Brand-name searches with dosage-form filler words ("NyQuil pill form
  medicine") not resolving
"""

from __future__ import annotations

import pytest

from src.dailymed import (
    _build_search_candidates,
    _is_ndc,
    _sanitize_query,
    _score_hit,
    _split_title,
)


class TestSplitTitle:
    """Verify SPL title parsing produces the right product name and dosage form."""

    def test_lipitor_film_coated_tablet(self) -> None:
        parts = _split_title(
            "LIPITOR (atorvastatin calcium) tablet, film coated [PFIZER]"
        )
        assert parts["product_name"] == "LIPITOR"
        assert parts["generic_name"] == "atorvastatin calcium"
        assert parts["dosage_form"] == "tablet, film coated"
        assert parts["manufacturer"] == "PFIZER"

    def test_excedrin_combination_tablet(self) -> None:
        parts = _split_title(
            "EXCEDRIN MIGRAINE (ACETAMINOPHEN, ASPIRIN, AND CAFFEINE) "
            "TABLET, FILM COATED [NAVAJO MANUFACTURING COMPANY INC.]"
        )
        assert parts["product_name"] == "EXCEDRIN MIGRAINE"
        assert parts["generic_name"] == "ACETAMINOPHEN, ASPIRIN, AND CAFFEINE"
        assert parts["dosage_form"] == "TABLET, FILM COATED"

    def test_desitin_ointment(self) -> None:
        parts = _split_title(
            "DESITIN MULTI-PURPOSE HEALING (PETROLATUM) OINTMENT [KENVUE BRANDS LLC]"
        )
        assert parts["product_name"] == "DESITIN MULTI-PURPOSE HEALING"
        assert parts["generic_name"] == "PETROLATUM"
        assert parts["dosage_form"] == "OINTMENT"

    def test_kit_in_product_name_does_not_dup(self) -> None:
        # The bug case: KIT appears in the product name AND as the form.
        # Before the rfind fix this returned dosage_form="KIT  KIT".
        parts = _split_title(
            "GENTLE CARE KIT (ZINC OXIDE) KIT [KENVUE BRANDS LLC]"
        )
        assert parts["product_name"] == "GENTLE CARE KIT"
        assert parts["dosage_form"] == "KIT"

    def test_johnsons_gift_set_kit(self) -> None:
        parts = _split_title(
            "JOHNSONS BABY CARE ESSENTIALS GIFT SET (ZINC OXIDE) KIT [KENVUE BRANDS LLC]"
        )
        assert parts["product_name"] == "JOHNSONS BABY CARE ESSENTIALS GIFT SET"
        assert parts["dosage_form"] == "KIT"

    def test_aerosol_metered_inhaler(self) -> None:
        parts = _split_title(
            "PROAIR HFA (albuterol sulfate) aerosol, metered [TEVA]"
        )
        assert parts["product_name"] == "PROAIR HFA"
        assert parts["dosage_form"] == "aerosol, metered"

    def test_no_dosage_form_keyword(self) -> None:
        parts = _split_title("UNKNOWN PRODUCT (some active) [SomeMfr]")
        assert parts["product_name"] == "UNKNOWN PRODUCT"
        assert parts["dosage_form"] is None
        assert parts["manufacturer"] == "SomeMfr"

    def test_empty_title(self) -> None:
        parts = _split_title("")
        assert parts == {
            "product_name": None,
            "generic_name": None,
            "dosage_form": None,
            "manufacturer": None,
        }


class TestScoreHit:
    """Verify the hit-scoring picks SPLs that actually match the user's query."""

    def test_query_in_title_outscores_unrelated(self) -> None:
        a = _score_hit(
            "Desitin",
            "DESITIN MULTI-PURPOSE HEALING (PETROLATUM) OINTMENT [KENVUE]",
        )
        b = _score_hit(
            "Desitin",
            "GENTLE CARE KIT (ZINC OXIDE) KIT [KENVUE BRANDS LLC]",
        )
        assert a > b
        assert b == 0

    def test_first_token_match_scores_high(self) -> None:
        # 'Excedrin' matches as the leading token of an EXCEDRIN MIGRAINE SPL
        score = _score_hit("Excedrin", "EXCEDRIN MIGRAINE (...) TABLET [...]")
        assert score >= 100

    def test_empty_query_scores_zero(self) -> None:
        assert _score_hit("", "ANY TITLE") == 0

    def test_empty_title_scores_zero(self) -> None:
        assert _score_hit("anything", "") == 0


class TestQuerySanitization:
    """Verify common filler words are stripped before DailyMed search."""

    def test_strips_pill_form_medicine(self) -> None:
        assert _sanitize_query("NyQuil pill form medicine") == "NyQuil"

    def test_strips_dosage_form_words(self) -> None:
        for noisy in [
            "NyQuil tablets",
            "NyQuil LiquiCaps",
            "ibuprofen tablet",
            "atorvastatin capsules",
        ]:
            cleaned = _sanitize_query(noisy)
            assert "tablet" not in cleaned.lower()
            assert "capsule" not in cleaned.lower()
            assert "liquicap" not in cleaned.lower()

    def test_preserves_strength_modifiers(self) -> None:
        # 'Extra Strength' carries real signal for DailyMed search
        assert _sanitize_query("Tylenol Extra Strength") == "Tylenol Extra Strength"

    def test_preserves_multi_word_brand(self) -> None:
        assert _sanitize_query("Excedrin Migraine") == "Excedrin Migraine"

    def test_returns_original_when_sanitization_empties_query(self) -> None:
        # Don't strip everything down to nothing
        assert _sanitize_query("pill") == "pill"


class TestSearchCandidates:
    """The candidate chain: original query, sanitized, first token."""

    def test_nyquil_filler_words_chain(self) -> None:
        candidates = _build_search_candidates("NyQuil pill form medicine")
        assert "NyQuil pill form medicine" in candidates
        assert "NyQuil" in candidates

    def test_clean_brand_only_returns_one_candidate(self) -> None:
        candidates = _build_search_candidates("atorvastatin")
        assert candidates == ["atorvastatin"]

    def test_short_first_token_skipped(self) -> None:
        # First token < 3 chars is too generic to fall back to
        candidates = _build_search_candidates("Rx tablet")
        assert "Rx" not in candidates


class TestNDCDetection:
    def test_dashed_ndc(self) -> None:
        assert _is_ndc("0067-1086-30") is True

    def test_undashed_ndc(self) -> None:
        assert _is_ndc("0067108630") is True

    def test_drug_name_is_not_ndc(self) -> None:
        assert _is_ndc("atorvastatin") is False

    def test_short_number_is_not_ndc(self) -> None:
        assert _is_ndc("123") is False
