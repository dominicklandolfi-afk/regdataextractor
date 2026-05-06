"""Data Manager page for the Streamlit app.

Lists every saved extraction, lets the user edit cells inline, run SQL,
delete rows, and download the full table. The Needs Review pane is
recomputed from current saved values, so it stays accurate after manual
edits.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st

from . import storage
from .extractor import (
    CONFIDENCE_THRESHOLD,
    CRITICAL_FIELDS,
    TRANSPORT_DETAIL_FIELDS,
    is_non_hazmat,
)
from .schema import SDSExtraction


def _gap_fields(row: pd.Series) -> tuple[list[str], list[str]]:
    """Return (critical_field_names, other_field_names) with gaps,
    computed live from the current values rather than the stored
    review_reasons string.

    Transport detail fields are skipped when the product is non-hazmat:
    an empty UN number on an oral tablet is the correct answer, not a gap.
    """
    sds_fields = list(SDSExtraction.model_fields.keys())
    critical: list[str] = []
    other: list[str] = []
    non_hazmat = is_non_hazmat(str(row.get("transport_regulated", "") or ""))
    transport_detail = set(TRANSPORT_DETAIL_FIELDS)
    for fname in sds_fields:
        if non_hazmat and fname in transport_detail:
            continue
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
            "Sources": row.get(f"{f}_sources", "") or "",
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


def _apply_dialog_edits(label: str, edits: dict[str, str]) -> int:
    """Persist dialog edits. Each edited SDS field gets confidence 100
    (human-verified). Returns the number of fields written."""
    if not edits:
        return 0
    full = storage.list_records()
    mask = full["label"] == label
    if not mask.any():
        return 0
    sds_fields = set(SDSExtraction.model_fields.keys())
    n = 0
    for fname, new_val in edits.items():
        if fname in sds_fields:
            full.loc[mask, fname] = new_val
            full.loc[mask, f"{fname}_confidence"] = 100
            n += 1
        elif fname in {"product_name", "generic_name", "manufacturer", "dosage_form", "ndc"}:
            full.loc[mask, fname] = new_val
            n += 1
    if n:
        storage.replace_table(full)
    return n


def _close_dialog() -> None:
    st.session_state.pop("dialog_label", None)


@st.dialog("Record details", width="large")
def _record_dialog(label: str) -> None:
    """Two-column scrollable detail view for one saved record. Inspired
    by the Account Manager drawer pattern. Edits are applied on Save
    Changes; Delete removes the row; Cancel discards edits."""
    df = storage.list_records()
    sel = df[df["label"] == label]
    if sel.empty:
        st.error(f"Record `{label}` not found. It may have been deleted.")
        if st.button("Close", key=f"dlg_close_missing_{label}"):
            _close_dialog()
            st.rerun()
        return

    row = sel.iloc[0]
    title = row.get("product_name") or row.get("input_query") or label
    st.subheader(title)
    saved_at = row.get("saved_at") or "unknown"
    st.caption(f"`{label}` · saved {saved_at}")

    review_reasons = (row.get("review_reasons") or "").strip()
    if review_reasons:
        st.warning(f"**Needs review:** {review_reasons}")

    edits: dict[str, str] = {}

    st.markdown("##### Identity")
    col_l, col_r = st.columns(2)
    _render_text_field(col_l, label, row, "input_query", "Input query", disabled=True)
    _render_text_field(col_l, label, row, "ndc", "NDC", disabled=True)
    _render_text_field(col_l, label, row, "manufacturer", "Manufacturer", edits=edits)
    _render_text_field(col_r, label, row, "product_name", "Product name", edits=edits)
    _render_text_field(col_r, label, row, "generic_name", "Generic name", edits=edits)
    _render_text_field(col_r, label, row, "dosage_form", "Dosage form", edits=edits)

    dailymed_url = row.get("dailymed_url") or ""
    if dailymed_url:
        st.caption(f"DailyMed SPL: [{dailymed_url}]({dailymed_url})")

    st.divider()
    st.markdown("##### Regulatory data")
    st.caption(
        "Edit any value below. Saving marks the field confidence 100 "
        "(human-verified). Confidence, evidence, and source provenance are "
        "shown beneath each field."
    )

    sds_fields = list(SDSExtraction.model_fields.keys())
    for i in range(0, len(sds_fields), 2):
        c_left, c_right = st.columns(2)
        _render_sds_field(c_left, label, row, sds_fields[i], edits)
        if i + 1 < len(sds_fields):
            _render_sds_field(c_right, label, row, sds_fields[i + 1], edits)

    st.divider()
    btn_delete, btn_cancel, btn_save = st.columns([1, 1, 1])
    with btn_delete:
        if st.button("Delete record", key=f"dlg_del_{label}", help="Remove this record permanently."):
            storage.delete_records([label])
            _close_dialog()
            st.rerun()
    with btn_cancel:
        if st.button("Cancel", key=f"dlg_cancel_{label}"):
            _close_dialog()
            st.rerun()
    with btn_save:
        if st.button("Save Changes", type="primary", key=f"dlg_save_{label}"):
            n = _apply_dialog_edits(label, edits)
            if n == 0:
                st.info("No changes to save.")
            else:
                st.success(f"Saved {n} field(s).")
                _close_dialog()
                st.rerun()


def _render_text_field(
    column,
    label: str,
    row: pd.Series,
    fname: str,
    pretty: str,
    disabled: bool = False,
    edits: Optional[dict[str, str]] = None,
) -> None:
    current = row.get(fname, "") or ""
    new = column.text_input(
        pretty,
        value=str(current),
        disabled=disabled,
        key=f"dlg_{fname}_{label}",
    )
    if edits is not None and not disabled and new != str(current):
        edits[fname] = new


def _render_sds_field(
    column,
    label: str,
    row: pd.Series,
    fname: str,
    edits: dict[str, str],
) -> None:
    pretty = fname.replace("_", " ").title()
    current = row.get(fname, "") or ""
    with column:
        new = st.text_input(
            pretty,
            value=str(current),
            key=f"dlg_{fname}_{label}",
        )
        if new != str(current):
            edits[fname] = new

        meta_bits: list[str] = []
        conf = row.get(f"{fname}_confidence")
        if conf is not None and not pd.isna(conf):
            try:
                meta_bits.append(f"Confidence: {int(conf)}")
            except (TypeError, ValueError):
                pass
        sources = row.get(f"{fname}_sources") or ""
        if sources:
            meta_bits.append(f"Sources: {sources}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        evidence = row.get(f"{fname}_evidence") or ""
        if evidence:
            with st.expander("Evidence", expanded=False):
                st.caption(evidence)

        src_url = row.get(f"{fname}_source") or ""
        if src_url:
            first_url = src_url.split(";")[0].strip()
            if first_url:
                st.caption(f"Source: [{first_url}]({first_url})")


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
                    "Sources": st.column_config.TextColumn(
                        disabled=True,
                        help="Databases that contributed to this value "
                             "(e.g. 'perplexity, dot' = Perplexity confirmed "
                             "by 49 CFR 172.101).",
                        width="small",
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
    st.subheader("All saved records")
    st.caption(
        "Browse, select, and delete records. Select a single row and "
        "click **Open record** to view and edit every field in a "
        "scrollable detail panel. Use the Needs Review pane above for "
        "inline edits on records flagged for review. For cross-record "
        "reports and bulk SQL, use the **Report Builder** page in the "
        "sidebar. The table toolbar (top right of the table) has built-in "
        "search and column-header sort."
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

    st.caption(
        "Tip: select a single row, then click **Open record** to view and "
        "edit every field in a scrollable detail panel."
    )

    action_col, view_col, download_col = st.columns([1, 1, 2])
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
    with view_col:
        if len(selected_labels) == 1:
            if st.button(f"Open record: {selected_labels[0]}"):
                st.session_state["dialog_label"] = selected_labels[0]
                st.rerun()
        elif len(selected_labels) > 1:
            st.caption("Select exactly one row to open the detail panel.")
        else:
            st.caption("")

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

    dialog_label = st.session_state.get("dialog_label")
    if dialog_label:
        _record_dialog(dialog_label)
