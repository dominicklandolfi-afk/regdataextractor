"""Report Builder page: preset reports + free-form SQL.

Replaces the SQL Workspace section that used to live inside the Data
Manager page. Each preset is a one-click button that drops a templated
SQL into the editor and runs it. The free-form editor below lets the
user tweak the query or write something new from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from . import storage


@dataclass(frozen=True)
class Preset:
    label: str
    helptext: str
    sql: str


PRESETS: tuple[Preset, ...] = (
    Preset(
        label="Entries last 3 days",
        helptext="Records added in the last 72 hours, newest first.",
        sql=(
            "SELECT label, input_query, product_name, manufacturer, "
            "saved_at, needs_review "
            "FROM products "
            "WHERE saved_at >= datetime('now', '-3 days') "
            "ORDER BY saved_at DESC"
        ),
    ),
    Preset(
        label="Entries last 7 days",
        helptext="Records added in the last week.",
        sql=(
            "SELECT label, input_query, product_name, manufacturer, "
            "saved_at, needs_review "
            "FROM products "
            "WHERE saved_at >= datetime('now', '-7 days') "
            "ORDER BY saved_at DESC"
        ),
    ),
    Preset(
        label="Flagged for review",
        helptext="Records where needs_review = 'YES'.",
        sql=(
            "SELECT label, input_query, product_name, saved_at, "
            "review_reasons "
            "FROM products "
            "WHERE needs_review = 'YES' "
            "ORDER BY saved_at DESC"
        ),
    ),
    Preset(
        label="Low confidence on critical fields",
        helptext=(
            "Any of product_type, secondary_physical_state, flash_point_c, "
            "transport_regulated, or rcra_classification under confidence 60."
        ),
        sql=(
            "SELECT label, input_query, product_name, "
            "product_type_confidence AS pt_conf, "
            "secondary_physical_state_confidence AS sps_conf, "
            "flash_point_c_confidence AS flash_conf, "
            "transport_regulated_confidence AS transport_conf, "
            "rcra_classification_confidence AS rcra_conf, "
            "saved_at "
            "FROM products "
            "WHERE COALESCE(product_type_confidence, 0) < 60 "
            "OR COALESCE(secondary_physical_state_confidence, 0) < 60 "
            "OR COALESCE(flash_point_c_confidence, 0) < 60 "
            "OR COALESCE(transport_regulated_confidence, 0) < 60 "
            "OR COALESCE(rcra_classification_confidence, 0) < 60 "
            "ORDER BY saved_at DESC"
        ),
    ),
    Preset(
        label="No DailyMed match",
        helptext=(
            "Records with no product name and no NDC: DailyMed lookup likely "
            "returned nothing. Worth re-running with an NDC."
        ),
        sql=(
            "SELECT label, input_query, saved_at, needs_review "
            "FROM products "
            "WHERE COALESCE(product_name, '') = '' "
            "AND COALESCE(ndc, '') = '' "
            "ORDER BY saved_at DESC"
        ),
    ),
    Preset(
        label="All entries (full table)",
        helptext="Every saved record, most recent first.",
        sql="SELECT * FROM products ORDER BY saved_at DESC",
    ),
)


def render() -> None:
    st.title("Report Builder")
    st.caption(
        "Run preset reports against the saved records, or compose your own "
        "SQL. The Data Manager page is for record-by-record review; this "
        "page is for cross-record reporting and bulk export."
    )

    if "report_last_sql" not in st.session_state:
        st.session_state["report_last_sql"] = ""
    if "report_last_label" not in st.session_state:
        st.session_state["report_last_label"] = ""

    st.markdown("##### Preset reports")
    st.caption("Click a preset to populate the query and run it.")
    cols_per_row = 3
    for start in range(0, len(PRESETS), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, preset in zip(row_cols, PRESETS[start:start + cols_per_row]):
            with col:
                if st.button(
                    preset.label,
                    help=preset.helptext,
                    key=f"preset_{preset.label}",
                    use_container_width=True,
                ):
                    st.session_state["report_last_sql"] = preset.sql
                    st.session_state["report_last_label"] = preset.label

    st.divider()
    st.markdown("##### Free-form SQL")
    st.caption(
        "Table: `products`. Columns are snake_case. SELECT returns rows; "
        "UPDATE/DELETE/INSERT report row count. One statement at a time."
    )

    default_sql = st.session_state.get("report_last_sql") or (
        "SELECT label, product_name, flash_point_c, transport_regulated "
        "FROM products ORDER BY saved_at DESC"
    )
    sql = st.text_area(
        "Query",
        value=default_sql,
        height=140,
        key="report_sql_textarea",
    )
    if st.button("Run query", type="primary"):
        st.session_state["report_last_sql"] = sql
        st.session_state["report_last_label"] = "Custom query"

    active_sql = st.session_state.get("report_last_sql")
    if not active_sql:
        return

    st.divider()
    label = st.session_state.get("report_last_label") or ""
    if label:
        st.markdown(f"##### Results: {label}")
    else:
        st.markdown("##### Results")

    result_df, rowcount, err = storage.run_sql(active_sql)
    if err:
        st.error(err)
        return
    if result_df is None:
        st.success(f"{rowcount} row(s) affected.")
        return

    st.success(f"{rowcount} row(s) returned.")
    with st.expander("Query", expanded=False):
        st.code(active_sql, language="sql")
    st.dataframe(
        result_df,
        use_container_width=True,
        height=440,
        hide_index=True,
    )
    _download_buttons(result_df)


def _download_buttons(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Report", index=False)
        buf.seek(0)
        st.download_button(
            "Download xlsx",
            data=buf,
            file_name=f"report_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download csv",
            data=csv,
            file_name=f"report_{ts}.csv",
            mime="text/csv",
        )
