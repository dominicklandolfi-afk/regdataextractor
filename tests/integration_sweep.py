"""End-to-end integration sweep across diverse product types.

Runs the full extraction pipeline (DailyMed + Perplexity + PubChem +
DOT) on a panel of products that span the major dosage forms and
hazmat profiles, then asserts the results against expected properties.
Costs roughly $0.01 per product in Perplexity API calls.

Run with:
    python -m tests.integration_sweep

Exits with a non-zero status if any product fails its expectations.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.extractor import (  # noqa: E402
    TRANSPORT_NOT_REGULATED,
    _physical_state_matches_dosage,
    extract_product,
)
from src.schema import ProductRecord  # noqa: E402


@dataclass(frozen=True)
class Expectation:
    """Lightweight ground-truth: what we expect for this product."""

    query: str
    label: str
    expected_dosage_keyword: tuple[str, ...]  # at least one must appear in DailyMed dosage_form
    expected_product_type: Optional[str] = None  # exact product_type if specified
    is_hazmat: Optional[bool] = None  # True for aerosols/flammables, False for oral solids/topicals
    notes: str = ""


PANEL: tuple[Expectation, ...] = (
    Expectation(
        query="atorvastatin",
        label="Rx tablet (Lipitor / generic)",
        expected_dosage_keyword=("TABLET",),
        expected_product_type="Prescription Pharmaceutical (Solid)",
        is_hazmat=False,
        notes="Generic Rx solid oral tablet; non-flammable, non-regulated.",
    ),
    Expectation(
        query="Excedrin",
        label="OTC combination tablet",
        expected_dosage_keyword=("TABLET",),
        expected_product_type="Prescription Pharmaceutical (Solid)",
        is_hazmat=False,
        notes="Combination of acetaminophen, aspirin, caffeine. Non-flammable.",
    ),
    Expectation(
        query="Tylenol Extra Strength",
        label="OTC tablet (acetaminophen)",
        expected_dosage_keyword=("TABLET",),
        expected_product_type="Prescription Pharmaceutical (Solid)",
        is_hazmat=False,
    ),
    Expectation(
        query="NyQuil pill form medicine",
        label="OTC liquid (filler-word query)",
        expected_dosage_keyword=("LIQUID", "SOLUTION", "SYRUP", "CAPSULE", "TABLET"),
        is_hazmat=None,  # NyQuil contains alcohol; transport may go either way depending on SPL
        notes="Tests query sanitization. Result depends on which Vicks NyQuil SPL DailyMed picks.",
    ),
    Expectation(
        query="Desitin",
        label="OTC topical ointment",
        expected_dosage_keyword=("OINTMENT", "PASTE", "CREAM"),
        expected_product_type="Prescription Pharmaceutical (Liquid)",
        is_hazmat=False,
        notes="Zinc oxide / petrolatum diaper rash topical. Non-flammable.",
    ),
    Expectation(
        query="hydrocortisone cream",
        label="OTC topical cream",
        expected_dosage_keyword=("CREAM", "OINTMENT", "LOTION"),
        expected_product_type="Prescription Pharmaceutical (Liquid)",
        is_hazmat=False,
    ),
    Expectation(
        query="ProAir HFA",
        label="Rx aerosol inhaler",
        expected_dosage_keyword=("AEROSOL", "INHALANT", "INHALATION", "SPRAY"),
        expected_product_type="Prescription Pharmaceutical (Aerosol)",
        is_hazmat=True,
        notes="Albuterol HFA metered-dose inhaler. UN1950 aerosol, regulated for transport.",
    ),
    Expectation(
        query="Listerine",
        label="OTC mouthwash (alcohol-containing)",
        expected_dosage_keyword=("LIQUID", "SOLUTION", "MOUTHWASH", "RINSE"),
        expected_product_type="Prescription Pharmaceutical (Liquid)",
        is_hazmat=None,  # depends on alcohol percentage in the specific SPL
        notes="Tests handling of flammable liquid OTC products.",
    ),
)


@dataclass
class Result:
    expectation: Expectation
    record: Optional[ProductRecord]
    failures: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.failures and self.record is not None


def _check(rec: ProductRecord, exp: Expectation) -> list[str]:
    failures: list[str] = []

    dm = rec.dailymed
    if not dm or not dm.product_name:
        failures.append(
            f"DailyMed lookup failed (no product_name) for query={exp.query!r}"
        )
        return failures

    dosage_form = (dm.dosage_form or "").upper()
    if not dosage_form:
        failures.append("DailyMed returned no dosage_form")
    else:
        if not any(kw in dosage_form for kw in exp.expected_dosage_keyword):
            failures.append(
                f"dosage_form {dosage_form!r} does not match expected "
                f"keywords {exp.expected_dosage_keyword}"
            )

    sds = rec.sds
    if sds is None:
        failures.append("Perplexity SDS extraction returned no result")
        return failures

    pt_value = (sds.product_type.value or "").strip()
    if exp.expected_product_type and pt_value != exp.expected_product_type:
        failures.append(
            f"product_type {pt_value!r} does not match expected "
            f"{exp.expected_product_type!r}"
        )

    sps_value = (sds.secondary_physical_state.value or "").strip()
    consistency = _physical_state_matches_dosage(dosage_form, sps_value)
    if consistency is False:
        failures.append(
            f"secondary_physical_state {sps_value!r} is inconsistent with "
            f"dosage form {dosage_form!r}"
        )

    transport_value = (sds.transport_regulated.value or "").strip()
    if exp.is_hazmat is True and transport_value in TRANSPORT_NOT_REGULATED:
        failures.append(
            f"transport_regulated={transport_value!r} but product is expected "
            f"to be regulated (hazmat). UN/hazard class likely missing too."
        )
    if exp.is_hazmat is False and transport_value not in TRANSPORT_NOT_REGULATED:
        failures.append(
            f"transport_regulated={transport_value!r} but product is expected "
            f"to be NON-regulated."
        )

    return failures


def _format_result(r: Result) -> str:
    status = "PASS" if r.passed else "FAIL"
    head = f"[{status}] {r.expectation.label} (query={r.expectation.query!r}) [{r.elapsed:.1f}s]"
    if r.passed:
        rec = r.record
        dm = rec.dailymed
        sds = rec.sds
        bits = [
            head,
            f"   product_name: {dm.product_name}",
            f"   dosage_form:  {dm.dosage_form}",
            f"   product_type: {sds.product_type.value if sds else '(none)'}",
            f"   physical:     {sds.secondary_physical_state.value if sds else '(none)'}",
            f"   transport:    {sds.transport_regulated.value if sds else '(none)'}",
            f"   needs_review: {rec.needs_review}",
        ]
        return "\n".join(bits)
    bits = [head]
    if r.record:
        rec = r.record
        dm = rec.dailymed
        sds = rec.sds
        bits.append(f"   product_name: {dm.product_name}")
        bits.append(f"   dosage_form:  {dm.dosage_form}")
        if sds:
            bits.append(f"   product_type: {sds.product_type.value}")
            bits.append(f"   physical:     {sds.secondary_physical_state.value}")
            bits.append(f"   transport:    {sds.transport_regulated.value}")
    for f in r.failures:
        bits.append(f"   - {f}")
    return "\n".join(bits)


def main() -> int:
    print(f"Running integration sweep across {len(PANEL)} products.")
    print(f"Estimated cost: ~${len(PANEL) * 0.012:.2f} in Perplexity API calls.")
    print()

    results: list[Result] = []
    for exp in PANEL:
        t0 = time.time()
        try:
            rec = extract_product(exp.query)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {exp.label}: {e}")
            results.append(Result(exp, None, [f"Exception: {e}"], time.time() - t0))
            continue
        elapsed = time.time() - t0
        failures = _check(rec, exp)
        result = Result(exp, rec, failures, elapsed)
        results.append(result)
        print(_format_result(result))
        print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"=== Summary: {passed}/{total} passed ===")
    if passed < total:
        print()
        print("Failures:")
        for r in results:
            if not r.passed:
                print(f"  - {r.expectation.label}: {len(r.failures)} issue(s)")
                for f in r.failures:
                    print(f"      * {f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
