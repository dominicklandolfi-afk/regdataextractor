"""Live integration sweep across the golden product panel.

Runs the full extraction pipeline against every product in the golden
fixture, then reports per-field pass/fail and an overall accuracy
percentage. Use this to drive systematic improvements: when a new bug
surfaces in production, add the product to golden_products.py and
rerun this sweep to verify the fix.

Costs roughly $0.01-0.02 per product in Perplexity API calls.

Run with:
    python -m tests.integration_sweep_v2

Exits non-zero if any product fails its category or any expected field
mismatches.
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

from src.categories import ProductCategory  # noqa: E402
from src.extractor import extract_product  # noqa: E402
from src.schema import ProductRecord  # noqa: E402
from tests.golden_products import GOLDEN_PANEL, GoldenProduct  # noqa: E402


@dataclass
class FieldResult:
    name: str
    expected: Optional[str]
    actual: Optional[str]
    confidence: Optional[int]
    passed: bool


@dataclass
class ProductResult:
    product: GoldenProduct
    record: Optional[ProductRecord]
    category_match: bool
    dosage_match: bool
    review_match: bool
    field_results: list[FieldResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def all_fields_pass(self) -> bool:
        return all(fr.passed for fr in self.field_results)

    @property
    def passed(self) -> bool:
        return (
            self.record is not None
            and self.category_match
            and self.dosage_match
            and self.review_match
            and self.all_fields_pass
            and not self.error
        )


def _check_product(p: GoldenProduct) -> ProductResult:
    t0 = time.time()
    try:
        rec = extract_product(p.query)
    except Exception as exc:  # noqa: BLE001
        return ProductResult(
            product=p,
            record=None,
            category_match=False,
            dosage_match=False,
            review_match=False,
            elapsed_s=time.time() - t0,
            error=f"Exception: {exc}",
        )

    elapsed = time.time() - t0
    dm = rec.dailymed
    sds = rec.sds

    category_match = rec.category == p.expected_category.value
    dosage_form = (dm.dosage_form or "").upper() if dm else ""
    dosage_match = any(kw in dosage_form for kw in p.expected_dosage_keywords)
    review_match = (
        p.needs_review_expected is None or rec.needs_review == p.needs_review_expected
    )

    field_results: list[FieldResult] = []
    if sds is not None:
        for fname, expected in p.expected_fields.items():
            field_obj = getattr(sds, fname)
            actual = field_obj.value
            passed = expected is None or actual == expected
            field_results.append(FieldResult(
                name=fname,
                expected=expected,
                actual=actual,
                confidence=field_obj.confidence,
                passed=passed,
            ))

    return ProductResult(
        product=p,
        record=rec,
        category_match=category_match,
        dosage_match=dosage_match,
        review_match=review_match,
        field_results=field_results,
        elapsed_s=elapsed,
    )


def _format_result(r: ProductResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    head = f"[{status}] {r.product.label} (query={r.product.query!r}) [{r.elapsed_s:.1f}s]"
    bits = [head]
    if r.error:
        bits.append(f"   ERROR: {r.error}")
        return "\n".join(bits)
    if r.record is None:
        bits.append("   No record returned")
        return "\n".join(bits)

    dm = r.record.dailymed
    bits.append(f"   product:      {dm.product_name or '(none)'}")
    bits.append(f"   dosage_form:  {dm.dosage_form or '(none)'}")
    bits.append(
        f"   category:     {r.record.category} "
        f"({'matches' if r.category_match else f'expected {r.product.expected_category.value}'})"
    )
    bits.append(f"   needs_review: {r.record.needs_review} "
                f"({'ok' if r.review_match else 'unexpected'})")
    if not r.dosage_match:
        bits.append(
            f"   dosage_form mismatch: got {dm.dosage_form!r}, "
            f"expected one of {r.product.expected_dosage_keywords}"
        )
    for fr in r.field_results:
        marker = "OK " if fr.passed else "X  "
        bits.append(
            f"   {marker} {fr.name:25s} expected={fr.expected!r:35s} "
            f"actual={fr.actual!r} conf={fr.confidence}"
        )
    return "\n".join(bits)


def _summary(results: list[ProductResult]) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    category_passes = sum(1 for r in results if r.category_match)
    dosage_passes = sum(1 for r in results if r.dosage_match)
    review_passes = sum(1 for r in results if r.review_match)
    total_fields = sum(len(r.field_results) for r in results)
    field_passes = sum(
        1 for r in results for fr in r.field_results if fr.passed
    )

    lines = [
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"  Products:      {passed:>3}/{total} passed ({100*passed/total:.0f}%)",
        f"  Categories:    {category_passes:>3}/{total} correct",
        f"  Dosage forms:  {dosage_passes:>3}/{total} correct",
        f"  Review flags:  {review_passes:>3}/{total} as expected",
    ]
    if total_fields:
        lines.append(
            f"  Field values:  {field_passes:>3}/{total_fields} match expected "
            f"({100*field_passes/total_fields:.0f}%)"
        )

    # Per-category breakdown
    by_cat: dict[str, list[ProductResult]] = {}
    for r in results:
        by_cat.setdefault(r.product.expected_category.value, []).append(r)
    lines.append("")
    lines.append("Per-category accuracy:")
    for cat, items in sorted(by_cat.items()):
        ok = sum(1 for r in items if r.passed)
        lines.append(f"  {cat:25s} {ok}/{len(items)}")

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for r in failures:
            issues: list[str] = []
            if not r.category_match:
                issues.append(f"category={r.record.category if r.record else '?'}")
            if not r.dosage_match:
                issues.append(f"dosage_form={r.record.dailymed.dosage_form if r.record else '?'!r}")
            if not r.review_match:
                issues.append(f"needs_review={r.record.needs_review if r.record else '?'}")
            for fr in r.field_results:
                if not fr.passed:
                    issues.append(f"{fr.name}={fr.actual!r}")
            lines.append(f"  - {r.product.label}: {'; '.join(issues)}")

    return "\n".join(lines)


def main() -> int:
    n = len(GOLDEN_PANEL)
    est_cost = n * 0.015
    print(f"Running golden sweep across {n} products.")
    print(f"Estimated cost: ~${est_cost:.2f} in Perplexity API calls.")
    print()

    results: list[ProductResult] = []
    for p in GOLDEN_PANEL:
        result = _check_product(p)
        results.append(result)
        print(_format_result(result))
        print()

    print(_summary(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
