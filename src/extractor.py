"""Orchestrator: combines DailyMed + Perplexity + PubChem + DOT into one ProductRecord.

Pipeline:
  1. DailyMed lookup (identity, active ingredients, dosage form).
  2. In parallel: Perplexity SDS extraction + PubChem physical properties
     for every active ingredient (combination products like Excedrin
     hit PubChem once per ingredient).
  3. DOT 49 CFR 172.101 lookup keyed on the UN number Perplexity returned
     (or the proper shipping name as a fallback).
  4. Cross-source enrichment: bumps confidence on agreement, replaces
     transport values when DOT contradicts Perplexity, flags PubChem
     conflicts on flash point (the safety-critical case where any active
     ingredient strongly contradicts a non-flammable claim).
  5. Transport normalization: when Perplexity says the product is not
     regulated for transport, fill UN number / shipping name / hazard
     class / packing group with explicit "Not Applicable" answers so an
     oral tablet does not look like a missing-data row.

PubChem returns properties for the *pure compound*, not the formulation,
so for boiling, water solubility, and pH it is informational only. Flash
point is the one place we treat a PubChem-vs-Perplexity disagreement as a
real review signal: if the SDS says "non-flammable" but PubChem reports a
low flash point for any active ingredient, a human should verify.
"""

from __future__ import annotations

import concurrent.futures
import re
from typing import Callable, Iterable, Optional

from . import dailymed, dot_hazmat, perplexity, pubchem
from .schema import DailyMedData, ProductRecord, SDSExtraction

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


_INGREDIENT_SPLIT = re.compile(r"\s*(?:,|;|\band\b|&|/)\s*", re.IGNORECASE)


def _pick_active_ingredients(dm: DailyMedData) -> list[str]:
    """Return every active ingredient name DailyMed surfaced. For combination
    products (Excedrin = acetaminophen + aspirin + caffeine) DailyMed often
    encodes the list inside the generic_name parentheses, e.g.,
    'acetaminophen, aspirin and caffeine'. Splits on commas, semicolons,
    'and', '&', and '/'. Deduplicates case-insensitively while preserving
    the first-seen casing."""
    candidates: list[str] = []
    if dm.active_ingredients:
        candidates.extend(dm.active_ingredients)
    elif dm.generic_name:
        candidates.extend(_INGREDIENT_SPLIT.split(dm.generic_name))
    seen: set[str] = set()
    out: list[str] = []
    for name in candidates:
        clean = name.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _append_evidence(existing: Optional[str], addition: str) -> str:
    if not existing:
        return addition
    return f"{existing} | {addition}"


def _values_agree(field_name: str, ppx: str, dot: str) -> bool:
    if not ppx and dot:
        return False
    if ppx == dot:
        return True
    if field_name == "proper_shipping_name":
        if ppx and dot and (ppx.lower() in dot.lower() or dot.lower() in ppx.lower()):
            return True
    if field_name == "packing_group":
        if {ppx, dot}.issubset({"None", "Not Applicable"}):
            return True
    return False


TRANSPORT_NOT_REGULATED = (
    "No, not regulated",
    "No, not regulated for transportation due to an exception",
)

TRANSPORT_DETAIL_FIELDS: tuple[str, ...] = (
    "un_number",
    "proper_shipping_name",
    "hazard_class",
    "packing_group",
)

NON_HAZMAT_DEFAULTS = {fname: "Not Applicable" for fname in TRANSPORT_DETAIL_FIELDS}


def is_non_hazmat(transport_regulated_value: Optional[str]) -> bool:
    """True when the SDS says the product is not regulated for transport.
    Used downstream to suppress 'missing data' flags on UN number, proper
    shipping name, hazard class, and packing group: those fields have no
    meaningful value for an oral tablet or capsule."""
    if not transport_regulated_value:
        return False
    return transport_regulated_value.strip() in TRANSPORT_NOT_REGULATED


def _enrich_pubchem(
    sds: SDSExtraction,
    pubchem_records: list[pubchem.PubChemData],
    sources: dict[str, list[str]],
    extra_review: list[str],
) -> None:
    """Cross-reference each active ingredient against the SDS values."""
    informative = [r for r in pubchem_records if r.has_any()]
    if not informative:
        return

    # Flash point: aggregate across ingredients. Any False (clear conflict)
    # demotes confidence and flags the row. All True means cross-confirmed.
    flash_evidence: list[str] = []
    flash_decisions: list[Optional[bool]] = []
    for rec in informative:
        if not rec.flash_point_text:
            continue
        decision = pubchem.flash_point_agreement(
            sds.flash_point_c.value, rec.flash_point_text
        )
        flash_decisions.append(decision)
        flash_evidence.append(
            f"{rec.name}: {rec.flash_point_text}"
            + (f" ({rec.compound_url})" if rec.compound_url else "")
        )
    if flash_evidence:
        joined = "; ".join(flash_evidence)
        if any(d is False for d in flash_decisions):
            sources["flash_point_c"].append("pubchem (disagrees)")
            sds.flash_point_c.confidence = min(sds.flash_point_c.confidence, 50)
            sds.flash_point_c.evidence_quote = _append_evidence(
                sds.flash_point_c.evidence_quote,
                f"PubChem flash points for active ingredients ({joined}) "
                f"disagree with the SDS value. Verify on the manufacturer SDS.",
            )
            extra_review.append(
                "flash_point_c: PubChem disagrees with SDS for at least one active ingredient"
            )
        elif any(d is True for d in flash_decisions):
            sources["flash_point_c"].append("pubchem")
            sds.flash_point_c.confidence = max(sds.flash_point_c.confidence, 90)
            sds.flash_point_c.evidence_quote = _append_evidence(
                sds.flash_point_c.evidence_quote,
                f"PubChem flash points for active ingredients ({joined}) "
                f"are consistent with the SDS value.",
            )
        else:
            sds.flash_point_c.evidence_quote = _append_evidence(
                sds.flash_point_c.evidence_quote,
                f"PubChem flash points (informational, may differ from "
                f"formulation): {joined}.",
            )

    # Boiling, solubility, pH: informational citations per ingredient.
    for fname, attr in (
        ("boiling_point_c", "boiling_point_text"),
        ("water_solubility", "water_solubility_text"),
        ("ph_value", "ph_text"),
    ):
        bits: list[str] = []
        for rec in informative:
            value = getattr(rec, attr)
            if value:
                cite = f" ({rec.compound_url})" if rec.compound_url else ""
                bits.append(f"{rec.name}: {value}{cite}")
        if not bits:
            continue
        sources[fname].append("pubchem (informational)")
        field = getattr(sds, fname)
        field.evidence_quote = _append_evidence(
            field.evidence_quote,
            f"PubChem (per active ingredient, may differ from formulation): "
            + "; ".join(bits) + ".",
        )
        if fname == "boiling_point_c":
            for rec in informative:
                if pubchem.boiling_point_agreement(
                    sds.boiling_point_c.value, rec.boiling_point_text
                ):
                    sds.boiling_point_c.confidence = max(
                        sds.boiling_point_c.confidence, 85
                    )
                    break


def _enrich_dot(
    sds: SDSExtraction,
    dot_entry: dot_hazmat.DOTEntry,
    sources: dict[str, list[str]],
    extra_review: list[str],
) -> None:
    """Apply DOT 49 CFR 172.101 corrections / confirmations."""
    transport_fields = (
        ("un_number", dot_entry.un_number),
        ("proper_shipping_name", dot_entry.proper_shipping_name),
        ("hazard_class", dot_entry.hazard_class),
        ("packing_group", dot_entry.packing_group),
    )
    for fname, dot_value in transport_fields:
        field = getattr(sds, fname)
        ppx_value = (field.value or "").strip()
        if _values_agree(fname, ppx_value, dot_value):
            sources[fname] = ["perplexity", "dot"]
            field.confidence = max(field.confidence, 95)
            field.evidence_quote = _append_evidence(
                field.evidence_quote,
                f"Confirmed by {dot_hazmat.CITATION}.",
            )
        else:
            sources[fname] = ["dot (corrected from perplexity)"]
            old = ppx_value or "(empty)"
            field.value = dot_value
            field.confidence = 95
            field.evidence_quote = _append_evidence(
                field.evidence_quote,
                f"Replaced by {dot_hazmat.CITATION}: was '{old}', "
                f"now '{dot_value}'.",
            )
            extra_review.append(
                f"{fname}: corrected from '{old}' to '{dot_value}' "
                f"using {dot_hazmat.CITATION}"
            )


_AEROSOL_DOSAGE_HINTS: tuple[str, ...] = (
    "AEROSOL", "INHALANT", "INHALATION", "SPRAY",
    "INHALER", "PROPELLANT",
)


def _looks_regulated(sds: SDSExtraction, dm: Optional[DailyMedData]) -> bool:
    """Defensive: detect contradictions between transport_regulated and
    other signals so we don't normalize an aerosol to 'Not Applicable'.
    Returns True when the product looks regulated regardless of what the
    transport_regulated dropdown says.
    """
    un_value = (sds.un_number.value or "").strip().lower()
    if un_value and un_value not in {"", "none", "n/a", "na", "not applicable"}:
        return True
    shipping = (sds.proper_shipping_name.value or "").strip().lower()
    if shipping and shipping not in {"", "none", "n/a", "na", "not applicable"}:
        return True
    hazard = (sds.hazard_class.value or "").strip()
    if hazard and hazard not in {"", "Not Applicable"}:
        return True
    if dm and dm.dosage_form:
        dosage = dm.dosage_form.upper()
        if any(hint in dosage for hint in _AEROSOL_DOSAGE_HINTS):
            return True
    return False


def _normalize_non_hazmat(
    sds: SDSExtraction,
    sources: dict[str, list[str]],
    dm: Optional[DailyMedData] = None,
) -> list[str]:
    """When Perplexity determined the product is not regulated for transport,
    fill the four transport detail fields with an explicit 'Not Applicable'
    answer instead of leaving them empty. An oral tablet or capsule should
    read as a definitive negative, not a missing-data row.

    Defensive: if any other signal contradicts the non-regulated claim
    (a UN number is present, dosage form is an aerosol, etc.), skip
    normalization and return a review reason so the human can resolve
    the conflict.
    """
    transport_value = (sds.transport_regulated.value or "").strip()
    if transport_value not in TRANSPORT_NOT_REGULATED:
        return []

    if _looks_regulated(sds, dm):
        return [
            "transport_regulated says 'not regulated' but other signals "
            "(UN number, hazard class, or aerosol dosage form) suggest the "
            "product IS regulated for transport - verify transport status"
        ]

    for fname, default_value in NON_HAZMAT_DEFAULTS.items():
        field = getattr(sds, fname)
        current = (field.value or "").strip()
        if current == "" or current.lower() in {"none", "n/a", "na"}:
            field.value = default_value
            field.confidence = max(field.confidence, 90)
            field.evidence_quote = _append_evidence(
                field.evidence_quote,
                f"Derived from transport_regulated = '{transport_value}': "
                f"non-hazmat products have no UN number, shipping name, "
                f"hazard class, or packing group.",
            )
            sources[fname] = list(dict.fromkeys(sources.get(fname, []) + ["derived"]))
    return []


def _enrich(
    sds: SDSExtraction,
    pubchem_records: list[pubchem.PubChemData],
    dot_entry: Optional[dot_hazmat.DOTEntry],
    dm: Optional[DailyMedData] = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Mutate sds with cross-source enrichment. Return (sources_map, extra_review)."""
    sources: dict[str, list[str]] = {
        fname: ["perplexity"] for fname in SDSExtraction.model_fields.keys()
    }
    extra_review: list[str] = []

    _enrich_pubchem(sds, pubchem_records, sources, extra_review)

    if dot_entry is not None:
        _enrich_dot(sds, dot_entry, sources, extra_review)

    contradiction_reasons = _normalize_non_hazmat(sds, sources, dm)
    extra_review.extend(contradiction_reasons)

    return sources, extra_review


def extract_product(query: str) -> ProductRecord:
    """Run the full pipeline for a single product query."""
    dm = dailymed.lookup(query)
    active_ingredients = _pick_active_ingredients(dm)

    parsed: Optional[SDSExtraction] = None
    perplexity_error: Optional[Exception] = None
    pubchem_records: list[pubchem.PubChemData] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        ppx_future = pool.submit(perplexity.extract_sds_fields, dm, query)
        pc_future = pool.submit(pubchem.lookup_multi, active_ingredients)
        try:
            parsed, _citations, _raw = ppx_future.result()
        except Exception as exc:  # network failure, auth failure, etc.
            perplexity_error = exc
        try:
            pubchem_records = pc_future.result()
        except Exception:
            pubchem_records = []

    if perplexity_error is not None:
        return ProductRecord(
            input_query=query, dailymed=dm, sds=None,
            needs_review=True,
            review_reasons=[f"Perplexity error: {perplexity_error}"],
        )
    if parsed is None:
        return ProductRecord(
            input_query=query, dailymed=dm, sds=None,
            needs_review=True,
            review_reasons=["Perplexity returned an unparseable response"],
        )

    sds = parsed

    un = (sds.un_number.value or "").strip()
    hazard_hint = (sds.hazard_class.value or "").strip()
    shipping_name = (sds.proper_shipping_name.value or "").strip()
    dot_entry = dot_hazmat.lookup(
        un_number=un or None,
        shipping_name=shipping_name or None,
        hazard_class_hint=hazard_hint or None,
    )

    sources_map, extra_review = _enrich(sds, pubchem_records, dot_entry, dm)

    needs_review, reasons = _evaluate_review(sds)
    if extra_review:
        needs_review = True
        reasons.extend(extra_review)

    return ProductRecord(
        input_query=query,
        dailymed=dm,
        sds=sds,
        needs_review=needs_review,
        review_reasons=reasons,
        sources=sources_map,
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
        row[f"{label} Sources"] = ", ".join(rec.sources.get(fname, []))
    return row
