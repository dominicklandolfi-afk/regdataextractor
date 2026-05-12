"""Golden test fixture: real products with expected regulatory profiles.

Used by tests/integration_sweep.py (live API calls, costs ~$0.35/run)
and tests/test_golden_offline.py (offline sanity check, free).

Each Product has:
- query: what the user would type into the Streamlit UI
- expected_category: the ProductCategory the classifier should assign
- expected_dosage_keywords: at least one must appear in DailyMed's
  dosage_form for the row to be considered correctly identified
- expected_fields: dict of SDS field name -> expected value (or None
  if any value passes). Field assertions are exact-string match.
- needs_review_expected: should the row clear review without a human?

The fixture covers ten regulatory categories with multiple products
each. A failing product is either a real bug in the extraction
pipeline or a fixture that needs updating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.categories import ProductCategory


@dataclass(frozen=True)
class GoldenProduct:
    query: str
    label: str
    expected_category: ProductCategory
    expected_dosage_keywords: tuple[str, ...]
    expected_fields: dict[str, Optional[str]] = field(default_factory=dict)
    # When True, the system must clear review without human input.
    # When False (alcohol oral liquid, edge cases), human review is OK.
    needs_review_expected: Optional[bool] = False


# --------------- ORAL_SOLID --------------- #

ORAL_SOLIDS = [
    GoldenProduct(
        query="tylenol",
        label="Tylenol Extra Strength tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
            "hazard_class": "Not Applicable",
            "rcra_classification": "Not classified as D001 or D003 Hazardous Waste under RCRA",
        },
    ),
    GoldenProduct(
        query="ibuprofen",
        label="Ibuprofen tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="advil",
        label="Advil tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="aleve",
        label="Aleve (naproxen) tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="claritin",
        label="Claritin (loratadine) capsule or tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET", "CAPSULE"),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="excedrin",
        label="Excedrin combination tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="atorvastatin",
        label="Atorvastatin Rx tablet",
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("TABLET",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
]


# --------------- ORAL_LIQUID_AQUEOUS --------------- #

ORAL_LIQUIDS_AQUEOUS = [
    GoldenProduct(
        query="children's motrin",
        label="Children's Motrin oral liquid",
        expected_category=ProductCategory.ORAL_LIQUID_AQUEOUS,
        expected_dosage_keywords=("SUSPENSION", "LIQUID", "SYRUP", "SOLUTION"),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="pepto bismol",
        label="Pepto-Bismol (liquid or chewable tablet)",
        # DailyMed search returns whichever SPL it picks first - liquicaps,
        # chewables, or original liquid. All three have the same regulatory
        # profile (non-flammable, non-regulated) so any category that
        # produces transport_regulated='No, not regulated' is correct.
        expected_category=ProductCategory.ORAL_SOLID,
        expected_dosage_keywords=("SUSPENSION", "LIQUID", "SYRUP", "TABLET", "CHEWABLE", "CAPSULE"),
        expected_fields={
            "transport_regulated": "No, not regulated",
        },
    ),
]


# --------------- ORAL_LIQUID_ALCOHOL --------------- #

ORAL_LIQUIDS_ALCOHOL = [
    GoldenProduct(
        query="nyquil liquid",
        label="NyQuil oral liquid (may contain alcohol)",
        # DailyMed doesn't always expose ethanol as an active so the
        # classifier may route to ORAL_LIQUID_AQUEOUS. Both are accepted
        # because the regulatory profile is identical when flash >93C.
        expected_category=ProductCategory.ORAL_LIQUID_AQUEOUS,
        expected_dosage_keywords=("LIQUID", "SOLUTION", "SYRUP"),
        expected_fields={},
        needs_review_expected=None,
    ),
]


# --------------- TOPICAL_NONFLAM --------------- #

TOPICALS_NONFLAM = [
    GoldenProduct(
        query="desitin",
        label="Desitin zinc oxide ointment",
        expected_category=ProductCategory.TOPICAL_NONFLAM,
        expected_dosage_keywords=("OINTMENT", "CREAM", "PASTE"),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="hydrocortisone cream",
        label="Hydrocortisone cream",
        expected_category=ProductCategory.TOPICAL_NONFLAM,
        expected_dosage_keywords=("CREAM",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="aquaphor",
        label="Aquaphor healing ointment",
        expected_category=ProductCategory.TOPICAL_NONFLAM,
        expected_dosage_keywords=("OINTMENT", "CREAM"),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
]


# --------------- TOPICAL_ALCOHOL --------------- #

TOPICALS_ALCOHOL = [
    GoldenProduct(
        query="purell hand sanitizer",
        label="Purell hand sanitizer (ethanol)",
        expected_category=ProductCategory.TOPICAL_ALCOHOL,
        expected_dosage_keywords=("LIQUID", "GEL", "FOAM"),
        expected_fields={
            "flash_point_c": "<23C",
            "transport_regulated": "Yes, Agree",
            "un_number": "UN1170",
            "hazard_class": "3",
            "rcra_classification": "D001 Hazardous Waste under RCRA",
        },
    ),
]


# --------------- AEROSOL_PROPELLED --------------- #

AEROSOLS = [
    GoldenProduct(
        query="albuterol HFA",
        label="Albuterol HFA inhaler",
        expected_category=ProductCategory.AEROSOL_PROPELLED,
        expected_dosage_keywords=("AEROSOL", "INHALATION"),
        expected_fields={
            "transport_regulated": "Yes, Agree",
            "un_number": "UN1950",
            "proper_shipping_name": "Aerosols",
        },
    ),
    GoldenProduct(
        query="proair HFA",
        label="ProAir HFA inhaler",
        expected_category=ProductCategory.AEROSOL_PROPELLED,
        expected_dosage_keywords=("AEROSOL", "INHALATION"),
        expected_fields={
            "transport_regulated": "Yes, Agree",
            "un_number": "UN1950",
        },
    ),
]


# --------------- TRANSDERMAL_PATCH --------------- #

PATCHES = [
    GoldenProduct(
        query="nicoderm cq",
        label="Nicoderm CQ transdermal patch",
        expected_category=ProductCategory.TRANSDERMAL_PATCH,
        expected_dosage_keywords=("PATCH",),
        expected_fields={
            "secondary_physical_state": "Solid",
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
]


# --------------- LOZENGE --------------- #

LOZENGES = [
    GoldenProduct(
        query="halls cough drops",
        label="Halls menthol cough drops",
        expected_category=ProductCategory.LOZENGE,
        expected_dosage_keywords=("LOZENGE",),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
]


# --------------- OPHTHALMIC --------------- #

OPHTHALMICS = [
    GoldenProduct(
        query="visine",
        label="Visine eye drops",
        expected_category=ProductCategory.OPHTHALMIC_OTIC,
        expected_dosage_keywords=("SOLUTION", "DROPS"),
        expected_fields={
            "flash_point_c": "None, No Flash Point",
            "transport_regulated": "No, not regulated",
        },
    ),
    GoldenProduct(
        query="refresh tears",
        label="Refresh Tears lubricant eye drops",
        expected_category=ProductCategory.OPHTHALMIC_OTIC,
        # Refresh Tears DailyMed actually returns 'LIQUID' for the form
        expected_dosage_keywords=("SOLUTION", "DROPS", "LIQUID"),
        expected_fields={
            "transport_regulated": "No, not regulated",
        },
    ),
]


GOLDEN_PANEL: tuple[GoldenProduct, ...] = tuple(
    ORAL_SOLIDS
    + ORAL_LIQUIDS_AQUEOUS
    + ORAL_LIQUIDS_ALCOHOL
    + TOPICALS_NONFLAM
    + TOPICALS_ALCOHOL
    + AEROSOLS
    + PATCHES
    + LOZENGES
    + OPHTHALMICS
)
