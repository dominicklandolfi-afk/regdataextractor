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


def _gap_fields(row: pd.Series) -> tuple[list[str], list[str]]:
    """Return (critical_field_names, other_field_names) with gaps,
    computed live from the current values rather than the stored
    review_reasons string.
    """
    sds_fields = list(SDSExtraction.model_fields.keys())
    critical: list[str] = []
    other: list[str] = []
    for fname in sds_fields:
        value = row.get(fname, "")
        conf = row.get(f"{fname}_confidence")
        empty = value in (None, "")
        low = (
            conf is not None
            and not pd.isna(conf)
            and int(conf) < CONFIDENCE_THRESHOLD
        )
        if empty or low:
            (critical if fname in CRITICAL_FIELDS else other).append(fname)
    return critical, other


def _build_review_table(row: pd.Series, fields: list[str], critical_set: set[str]) -> pd.DataFrame:
    rows = []
    for f in fields:
        rows.append({
            "Field": f,
            "Critical": "Yes" if f in critical_set else "",
            "Value": row.get(f, "") or "",
            "Confidence": row.get(f"{f}_confidence"),
            "Evidence": row.get(f"{f}_evidence", "") or "",
            "Source": row.get(f"{f}_source", "") or "",
        })
    return pd.DataFrame(rows)


def _apply_review_edits(label: str, original_df: pd.DataFrame, edited: pd.DataFrame) -> int:
    """Write back any value edits for the given label. Sets confidence to
    100 on edited fields (human-verified). Returns the number of fields
    actually changed.
    """
    full = storage.list_records()
    mask = full["label"] == label
    if not mask.any():
        return 0
    changed = 0
    for _, new_row in edited.iterrows():
        field = new_row["Field"]
        new_val = "" if new_row["Value"] is None else str(new_row["Value"])
        orig = original_df.loc[original_df["Field"] == field, "Value"]
        orig_val = "" if orig.empty else str(orig.iloc[0] or "")
        if new_val != orig_val:
            full.loc[mask, field] = new_val
            full.loc[mask, f"{field}_confidence"] = 100
            changed += 1
    if changed:
        storage.replace_table(full)
    return changed


def _render_needs_review(df: pd.DataFrame) -> None:
    items = []
    for _, row in df.iterrows():
        crit, other = _gap_fields(row)
        if crit or other:
            items.append({
                "label": row["label"],
                "product_name": row.get("product_name", "") or "",
                "saved_at": row.get("saved_at", "") or "",
                "row": row,
                "critical": crit,
                "other": other,
            })
    items.sort(key=lambda x: x["saved_at"], reverse=True)
    items.sort(key=lambda x: (-len(x["critical"]), -len(x["other"])))

    st.subheader("Needs review")
    if not items:
        st.success("All saved records have complete, high-confidence values.")
        return

    st.caption(
        f"{len(items)} of {len(df)} record(s) have empty or low-confidence "
        f"fields (confidence threshold: {CONFIDENCE_THRESHOLD}). Critical fields "
        "appear first in each record. Edit the Value column inline; saved "
        "edits are marked confidence 100 (human-verified)."
    )

    for item in items:
        crit_n, other_n = len(item["critical"]), len(item["other"])
        ts = ""
        if item["saved_at"]:
            try:
                dt = pd.to_datetime(item["saved_at"])
                ts = f"{dt.strftime('%b')} {dt.day}, {dt.year} {dt.strftime('%I:%M %p').lstrip('0')}"
            except (ValueError, TypeError):
                ts = str(item["saved_at"])
        title_parts = [item["label"]]
        if item["product_name"]:
            title_parts.append(f"— {item['product_name']}")
        if ts:
            title_parts.append(f"| {ts}")
        title_parts.append(f"| {crit_n} critical, {other_n} other")
        with st.expander(" ".join(title_parts)):
            ordered_fields = item["critical"] + item["other"]
            critical_set = set(item["critical"])
            review_df = _build_review_table(item["row"], ordered_fields, critical_set)

            edited = st.data_editor(
                review_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key=f"review_editor_{item['label']}",
                column_config={
                    "Field": st.column_config.TextColumn(disabled=True, width="medium"),
                    "Critical": st.column_config.TextColumn(disabled=True, width="small"),
                    "Value": st.column_config.TextColumn(
                        help="Double-click to edit. Saving sets confidence to 100.",
                        width="medium",
                    ),
                    "Confidence": st.column_config.NumberColumn(disabled=True, width="small"),
                    "Evidence": st.column_config.TextColumn(disabled=True, width="large"),
                    "Source": st.column_config.TextColumn(
                        disabled=True,
                        help="May contain multiple URLs separated by '; '.",
                        width="medium",
                    ),
                },
            )

            if st.button(
                f"Save edits to {item['label']}",
                type="primary",
                key=f"review_save_{item['label']}",
            ):
                try:
                    n = _apply_review_edits(item["label"], review_df, edited)
                    if n == 0:
                        st.info("No changes detected.")
                    else:
                        st.success(f"Saved {n} field edit(s). Refreshing.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")


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
    st.caption(
        "Browse, select, and delete records. Use the Needs Review pane "
        "above for inline edits, or the SQL pane for bulk changes. The "
        "table toolbar (top right of the table) has built-in search and "
        "column-header sort."
    )

    essentials = (
        ["label", "saved_at", "input_query", "product_name", "needs_review", "manufacturer", "dosage_form"]
        + list(SDSExtraction.model_fields.keys())
    )
    essentials = [c for c in essentials if c in df.columns]
    all_cols = list(df.columns)

    with st.expander("Add row (manual entry)"):
        with st.form("add_row_form", clear_on_submit=True):
            new_query = st.text_input(
                "Product / input query",
                placeholder="e.g. albuterol HFA 90mcg",
            )
            new_pname = st.text_input(
                "Product name (optional)",
                placeholder="ProAir HFA",
            )
            submitted = st.form_submit_button("Add row", type="primary")
            if submitted:
                if not new_query.strip():
                    st.error("Product / input query is required.")
                else:
                    new_label = storage.add_blank_record(
                        new_query.strip(), new_pname.strip()
                    )
                    st.success(
                        f"Added '{new_label}'. Open the Needs Review pane "
                        f"above to fill in the SDS fields."
                    )
                    st.rerun()

    visible = st.multiselect(
        "Show columns",
        options=all_cols,
        default=st.session_state.get("dm_visible_cols", essentials),
        key="dm_visible_cols",
    )

    if not visible:
        visible = essentials
    if "label" not in visible:
        visible = ["label"] + visible

    if st.button("Show all columns"):
        st.session_state["dm_visible_cols"] = all_cols
        st.rerun()

    st.caption(
        f"{len(df)} record(s), most recent first. "
        f"{len(visible)} of {len(all_cols)} columns shown."
    )

    view = df[visible].reset_index(drop=True)
    if "saved_at" in view.columns:
        view["saved_at"] = pd.to_datetime(view["saved_at"], errors="coerce")
    event = st.dataframe(
        view,
        use_container_width=True,
        height=420,
        hide_index=True,
        selection_mode="multi-row",
        on_select="rerun",
        key="data_manager_dataframe",
        column_config={
            "saved_at": st.column_config.DatetimeColumn(
                label="Saved",
                format="MMM D, YYYY h:mm A",
            ),
        },
    )

    selected_indices = list(getattr(event.selection, "rows", []) or [])
    selected_labels = view.iloc[selected_indices]["label"].tolist() if selected_indices else []

    action_col, download_col = st.columns([1, 2])
    with action_col:
        if selected_labels:
            if st.button(
                f"Delete {len(selected_labels)} selected row"
                + ("s" if len(selected_labels) != 1 else ""),
                type="primary",
            ):
                removed = storage.delete_records(selected_labels)
                st.success(f"Deleted {removed} record(s).")
                st.rerun()
        else:
            st.caption("Select rows in the table to enable deletion.")

    with download_col:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_df = view
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, sheet_name="Saved Records", index=False)
        buf.seek(0)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                label="Download xlsx (current view)",
                data=buf,
                file_name=f"saved_extractions_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with dcol2:
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download csv (current view)",
                data=csv,
                file_name=f"saved_extractions_{ts}.csv",
                mime="text/csv",
            )
