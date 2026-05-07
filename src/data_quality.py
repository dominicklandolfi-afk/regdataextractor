"""Data Quality dashboard.

Aggregate health view of every saved extraction. Surfaces the things
that should drive a manual audit:

- How many records flag for review
- Confidence distribution on the five critical fields
- Records with no DailyMed match (extraction failed at step 1)
- Records where Perplexity's secondary_physical_state contradicts
  DailyMed's dosage form (the Desitin failure mode)
- Records where transport status looks wrong relative to dosage form

The point: glance at this page once a week and know whether the system
is producing trustworthy output, without auditing individual rows.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import storage
from .extractor import (
    CONFIDENCE_THRESHOLD,
    CRITICAL_FIELDS,
    TRANSPORT_NOT_REGULATED,
    _physical_state_matches_dosage,
    is_non_hazmat,
)


def _critical_confidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-critical-field stats: median confidence, low-confidence count,
    missing count."""
    rows = []
    for fname in CRITICAL_FIELDS:
        conf_col = f"{fname}_confidence"
        if fname not in df.columns:
            continue
        values = df[fname].astype(str).str.strip()
        conf = pd.to_numeric(df[conf_col], errors="coerce") if conf_col in df.columns else pd.Series([], dtype=float)
        missing = int((values == "") .sum() + values.isna().sum())
        low = int((conf < CONFIDENCE_THRESHOLD).sum()) if not conf.empty else 0
        median_conf = float(conf.median()) if not conf.empty else 0.0
        rows.append({
            "Field": fname,
            "Records": len(df),
            "Missing": missing,
            f"Below {CONFIDENCE_THRESHOLD}": low,
            "Median confidence": round(median_conf, 1),
        })
    return pd.DataFrame(rows)


def _dosage_mismatch_records(df: pd.DataFrame) -> pd.DataFrame:
    """Records where Perplexity's secondary_physical_state doesn't match
    DailyMed's dosage form (the Desitin/cream-as-tablet failure mode)."""
    if "dosage_form" not in df.columns or "secondary_physical_state" not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        dosage = str(r.get("dosage_form") or "").strip()
        picked = str(r.get("secondary_physical_state") or "").strip()
        if not dosage or not picked:
            continue
        match = _physical_state_matches_dosage(dosage, picked)
        if match is False:
            rows.append({
                "label": r.get("label", ""),
                "product_name": r.get("product_name", ""),
                "dosage_form": dosage,
                "picked physical state": picked,
                "saved_at": r.get("saved_at", ""),
            })
    return pd.DataFrame(rows)


def _transport_inconsistency_records(df: pd.DataFrame) -> pd.DataFrame:
    """Records where transport_regulated says non-regulated but the
    dosage form / hazard class suggest regulated. Catches Perplexity
    misclassifications that slipped past the in-pipeline guard."""
    rows = []
    aerosol_hints = ("AEROSOL", "INHALANT", "INHALATION", "PROPELLANT")
    for _, r in df.iterrows():
        transport = str(r.get("transport_regulated") or "").strip()
        if not is_non_hazmat(transport):
            continue
        dosage = str(r.get("dosage_form") or "").upper()
        un = str(r.get("un_number") or "").strip().lower()
        hazard = str(r.get("hazard_class") or "").strip()
        flagged = False
        reasons: list[str] = []
        if any(hint in dosage for hint in aerosol_hints):
            flagged = True
            reasons.append(f"dosage form {dosage} suggests regulated")
        if un and un not in {"", "none", "n/a", "na", "not applicable"}:
            flagged = True
            reasons.append(f"un_number = {un}")
        if hazard and hazard not in {"", "Not Applicable"}:
            flagged = True
            reasons.append(f"hazard_class = {hazard}")
        if flagged:
            rows.append({
                "label": r.get("label", ""),
                "product_name": r.get("product_name", ""),
                "transport_regulated": transport,
                "dosage_form": dosage,
                "concern": "; ".join(reasons),
            })
    return pd.DataFrame(rows)


def _no_dailymed_match(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[
        df["product_name"].fillna("").astype(str).str.strip().eq("")
        & df["ndc"].fillna("").astype(str).str.strip().eq("")
    ]
    return rows[["label", "input_query", "saved_at"]] if not rows.empty else pd.DataFrame()


def render() -> None:
    st.title("Data Quality")
    st.caption(
        "A glance at the aggregate health of saved extractions. Use this "
        "page to spot systemic issues without auditing individual records. "
        "Each section below shows a specific failure mode the pipeline "
        "tries to catch automatically; non-empty tables here are records "
        "that need a human look."
    )

    df = storage.list_records()
    if df.empty:
        st.info("No saved records yet. Run an extraction on the Extract page first.")
        return

    total = len(df)
    flagged = int((df["needs_review"].astype(str) == "YES").sum())
    pct = (flagged / total * 100) if total else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", total)
    col2.metric("Flagged for review", flagged, f"{pct:.0f}% of total")

    no_dm = _no_dailymed_match(df)
    col3.metric("No DailyMed match", len(no_dm))

    st.divider()
    st.subheader("Confidence on critical fields")
    st.caption(
        "These five fields drive the review flag. A field with high median "
        "confidence and no missing values is in good shape. Many missing or "
        "low-confidence entries mean the prompt or pipeline needs work."
    )
    summary = _critical_confidence_summary(df)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dosage form vs picked physical state mismatches")
    st.caption(
        "Records where DailyMed says (e.g.) OINTMENT but Perplexity picked "
        "Capsule/Tablet. The pipeline already demotes confidence and adds "
        "a review reason for these; this list lets you see them all in one "
        "place. Should be empty or near-empty in normal operation."
    )
    mismatches = _dosage_mismatch_records(df)
    if mismatches.empty:
        st.success("No dosage-form mismatches detected.")
    else:
        st.warning(f"{len(mismatches)} record(s) have a mismatch.")
        st.dataframe(mismatches, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Transport classification inconsistencies")
    st.caption(
        "Records where transport_regulated says 'No, not regulated' but "
        "other signals (aerosol dosage form, UN number present, hazard "
        "class set) suggest the product IS regulated. Most are caught by "
        "the in-pipeline guard, but the dashboard surfaces any that "
        "slipped through historic data or edge cases."
    )
    transport_issues = _transport_inconsistency_records(df)
    if transport_issues.empty:
        st.success("No transport classification inconsistencies detected.")
    else:
        st.warning(f"{len(transport_issues)} record(s) flagged.")
        st.dataframe(transport_issues, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Records with no DailyMed match")
    st.caption(
        "If DailyMed cannot find the product, downstream extraction has "
        "almost nothing to anchor on. Re-run these with an NDC if you have "
        "one, or with a more specific product name."
    )
    if no_dm.empty:
        st.success("Every record has a DailyMed match.")
    else:
        st.warning(f"{len(no_dm)} record(s) have no DailyMed match.")
        st.dataframe(no_dm, use_container_width=True, hide_index=True)
