"""Data Manager page for the Streamlit app.

Lists every saved extraction, lets the user edit cells inline, run SQL,
delete rows, and download the full table. The Needs Review pane is
recomputed from current saved values, so it stays accurate after manual
edits.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from . import storage
from .extractor import CONFIDENCE_THRESHOLD, CRITICAL_FIELDS
from .schema import SDSExtraction


def _compute_gaps(row: pd.Series) -> tuple[list[str], list[str]]:
    """Return (critical_gaps, other_gaps) for a saved row, computed live
    from the current values rather than the stored review_reasons string.
    """
    sds_fields = list(SDSExtraction.model_fields.keys())
    critical: list[str] = []
    other: list[str] = []
    for fname in sds_fields:
        value = row.get(fname, "")
        conf = row.get(f"{fname}_confidence")
        if value in (None, ""):
            gap = f"{fname}: empty"
        elif conf is not None and not pd.isna(conf) and int(conf) < CONFIDENCE_THRESHOLD:
            gap = f"{fname}: low confidence ({int(conf)})"
        else:
            continue
        if fname in CRITICAL_FIELDS:
            critical.append(gap)
        else:
            other.append(gap)
    return critical, other


def _render_needs_review(df: pd.DataFrame) -> None:
    items = []
    for _, row in df.iterrows():
        crit, other = _compute_gaps(row)
        if crit or other:
            items.append({
                "label": row["label"],
                "product_name": row.get("product_name", "") or "",
                "critical": crit,
                "other": other,
            })
    items.sort(key=lambda x: (-len(x["critical"]), -len(x["other"])))

    st.subheader("Needs review")
    if not items:
        st.success("All saved records have complete, high-confidence values.")
        return

    st.caption(
        f"{len(items)} of {len(df)} record(s) have empty or low-confidence "
        f"fields (confidence threshold: {CONFIDENCE_THRESHOLD}). Sorted by "
        "critical-gap count."
    )
    for item in items:
        crit_n, other_n = len(item["critical"]), len(item["other"])
        title_parts = [item["label"]]
        if item["product_name"]:
            title_parts.append(f"— {item['product_name']}")
        title_parts.append(f"| {crit_n} critical, {other_n} other")
        with st.expander(" ".join(title_parts)):
            if item["critical"]:
                st.markdown("**Critical fields:**")
                for g in item["critical"]:
                    st.markdown(f"- {g}")
            if item["other"]:
                st.markdown("**Other fields:**")
                for g in item["other"]:
                    st.markdown(f"- {g}")


def render() -> None:
    st.title("Data Manager")
    st.caption(
        "Every extraction is auto-saved here. Duplicate inputs get a "
        "numeric suffix (e.g., 'ibuprofen2') so previous results are "
        "preserved. Edit cells inline, run SQL, or delete rows."
    )

    df = storage.list_records()
    st.metric("Saved records", len(df))

    if df.empty:
        st.info("No records yet. Run an extraction to populate this table.")
        return

    _render_needs_review(df)

    st.divider()
    st.subheader("SQL query")
    st.caption(
        "Table: `products`. Columns are snake_case. SELECT returns rows; "
        "UPDATE/DELETE/INSERT report row count. One statement at a time."
    )
    sql_default = st.session_state.get(
        "sql_query",
        "SELECT label, product_name, flash_point_c, transport_regulated FROM products ORDER BY saved_at DESC",
    )
    sql = st.text_area(
        "Query",
        value=sql_default,
        height=110,
        key="sql_query",
    )
    if st.button("Run query"):
        result_df, rowcount, err = storage.run_sql(sql)
        if err:
            st.error(err)
        elif result_df is not None:
            st.success(f"{rowcount} row(s) returned.")
            st.dataframe(result_df, use_container_width=True, height=320)
        else:
            st.success(f"{rowcount} row(s) affected.")

    st.divider()
    st.subheader("All saved records")
    st.caption("Edit any cell, then click 'Save edits' to write back to the database. The 'label' column is the primary key — keep it unique.")

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        height=420,
        key="data_manager_editor",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Save edits", type="primary"):
            try:
                count = storage.replace_table(edited)
                st.success(f"Saved {count} record(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    with col2:
        labels_to_delete = st.multiselect(
            "Delete labels",
            options=df["label"].tolist(),
            key="delete_labels",
        )
        if st.button("Delete selected", disabled=not labels_to_delete):
            removed = storage.delete_records(labels_to_delete)
            st.success(f"Deleted {removed} record(s).")
            st.rerun()

    with col3:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            edited.to_excel(writer, sheet_name="Saved Records", index=False)
        buf.seek(0)
        st.download_button(
            label="Download xlsx",
            data=buf,
            file_name=f"saved_extractions_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        csv = edited.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download csv",
            data=csv,
            file_name=f"saved_extractions_{ts}.csv",
            mime="text/csv",
        )
