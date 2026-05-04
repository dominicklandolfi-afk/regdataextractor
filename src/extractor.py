"""Orchestrator: combines DailyMed + Perplexity into one ProductRecord.

Computes needs_review based on confidence threshold and missing fields.
Exposes a single function `extract_product` plus a batch helper.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from . import dailymed, perplexity
from .schema import ProductRecord, SDSExtraction

CONFIDENCE_THRESHOLD = 60  # Below this, flag for human review.

CRITICAL_FIELDS = (
    "product_type",
    "secondary_physical_state",
    "flash_point_c",
    "transport_regulated",
    "rcra_classification",
)


def _evaluate_review(sds: Optional[SDSExtraction]) -> tuple[bool, list[str]]:
    if sds is None:
        return True, ["SDS extraction failed or returned no data"]
    reasons: list[str] = []
    for fname in CRITICAL_FIELDS:
        field = getattr(sds, fname)
        if field.value in (None, ""):
            reasons.append(f"{fname}: missing value")
        elif field.confidence < CONFIDENCE_THRESHOLD:
            reasons.append(f"{fname}: low confidence ({field.confidence})")
    return len(reasons) > 0, reasons


def extract_product(query: str) -> ProductRecord:
    """Run the full pipeline for a single product query."""
    dm = dailymed.lookup(query)

    sds: Optional[SDSExtraction] = None
    try:
        parsed, _citations, _raw = perplexity.extract_sds_fields(dm, query)
        sds = parsed
    except Exception as exc:  # network failure, auth failure, etc.
        sds = None
        review_reasons = [f"Perplexity error: {exc}"]
        return ProductRecord(
            input_query=query, dailymed=dm, sds=None,
            needs_review=True, review_reasons=review_reasons,
        )

    needs_review, reasons = _evaluate_review(sds)
    return ProductRecord(
        input_query=query, dailymed=dm, sds=sds,
        needs_review=needs_review, review_reasons=reasons,
    )


def extract_batch(
    queries: Iterable[str],
    on_progress: Optional[Callable[[int, int, ProductRecord], None]] = None,
) -> list[ProductRecord]:
    queries = list(queries)
    total = len(queries)
    results: list[ProductRecord] = []
    for idx, q in enumerate(queries, start=1):
        rec = extract_product(q)
        results.append(rec)
        if on_progress:
            on_progress(idx, total, rec)
    return results


def to_flat_dict(rec: ProductRecord) -> dict:
    """Flatten a ProductRecord into a single row dict for the spreadsheet."""
    row: dict = {
        "Input Query": rec.input_query,
        "Needs Review": "YES" if rec.needs_review else "no",
        "Review Reasons": "; ".join(rec.review_reasons),
        "NDC": rec.dailymed.ndc or "",
        "Product Name": rec.dailymed.product_name or "",
        "Generic Name": rec.dailymed.generic_name or "",
        "Manufacturer": rec.dailymed.manufacturer or "",
        "Dosage Form": rec.dailymed.dosage_form or "",
        "DailyMed URL": rec.dailymed.spl_url or "",
    }
    if rec.sds is None:
        return row
    for fname, field in rec.sds.model_dump().items():
        label = fname.replace("_", " ").title()
        row[f"{label}"] = field.get("value") or ""
        row[f"{label} Confidence"] = field.get("confidence") if field.get("confidence") is not None else ""
        row[f"{label} Evidence"] = field.get("evidence_quote") or ""
        row[f"{label} Source"] = field.get("source_url") or ""
    return row
