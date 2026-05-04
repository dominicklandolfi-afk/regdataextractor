"""Streamlit web UI for the RX Regulatory Data Extractor.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.extractor import extract_batch, to_flat_dict

st.set_page_config(
    page_title="RX Regulatory Data Extractor",
    page_icon=None,
    layout="wide",
)


def _get_app_password() -> str:
    pw = os.getenv("APP_PASSWORD", "")
    if pw:
        return pw
    try:
        return st.secrets.get("APP_PASSWORD", "")
    except Exception:
        return ""


def _password_gate() -> None:
    expected = _get_app_password()
    if not expected:
        return
    if st.session_state.get("auth_ok"):
        return
    st.title("RX Regulatory Data Extractor")
    st.caption("Internal demo. Enter the access password to continue.")
    pw = st.text_input("Password", type="password", key="_pw_input")
    if pw:
        if pw == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_password_gate()

st.title("RX Regulatory Data Extractor")
st.caption(
    "Looks up prescription pharmaceutical products in DailyMed, finds the "
    "Safety Data Sheet via Perplexity, and extracts the regulatory fields "
    "from the portal spec sheet. Each value is returned with a confidence "
    "score, an evidence quote, and a source URL."
)

with st.sidebar:
    st.header("Run Settings")
    st.markdown(
        "Paste one product per line. Use NDC numbers (e.g., `0067-1086-30`) "
        "or product names (e.g., `atorvastatin`). NDCs are more accurate."
    )
    sample_queries = ["atorvastatin", "sertraline", "ibuprofen"]
    if st.button("Load 3 sample products"):
        st.session_state["product_input"] = "\n".join(sample_queries)
    st.divider()
    st.markdown("**Confidence threshold for human review:** 60")
    st.markdown(
        "Rows where any critical field (transport, flash point, RCRA, "
        "physical state) has confidence below 60 are flagged."
    )

product_input = st.text_area(
    "Products to research",
    value=st.session_state.get("product_input", ""),
    height=180,
    placeholder="atorvastatin\nsertraline\nibuprofen",
    key="product_input",
)

run_clicked = st.button(
    "Extract regulatory data",
    type="primary",
    disabled=not product_input.strip(),
)

if run_clicked:
    queries = [line.strip() for line in product_input.splitlines() if line.strip()]
    if not queries:
        st.warning("No products to process.")
        st.stop()

    progress_bar = st.progress(0.0, text="Starting...")
    status_box = st.empty()
    rows: list[dict] = []

    def on_progress(idx: int, total: int, record) -> None:
        progress_bar.progress(idx / total, text=f"Processed {idx} of {total}: {record.input_query}")
        status_box.info(
            f"**{record.input_query}** -> "
            f"{record.dailymed.product_name or 'no DailyMed match'}"
            + ("  (FLAGGED FOR REVIEW)" if record.needs_review else "")
        )

    with st.spinner("Calling DailyMed and Perplexity..."):
        results = extract_batch(queries, on_progress=on_progress)

    rows = [to_flat_dict(r) for r in results]
    df = pd.DataFrame(rows)
    st.session_state["last_results_df"] = df
    progress_bar.progress(1.0, text="Done.")
    status_box.success(f"Finished {len(results)} products. {sum(r.needs_review for r in results)} flagged for review.")

if "last_results_df" in st.session_state:
    df: pd.DataFrame = st.session_state["last_results_df"]
    st.subheader("Results")
    st.caption("Edit any cell, then download. Flagged rows are highlighted.")

    def _highlight_flag(row: pd.Series) -> list[str]:
        if str(row.get("Needs Review", "")).upper() == "YES":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        height=420,
        key="results_editor",
    )

    col1, col2 = st.columns(2)
    with col1:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            edited.to_excel(writer, sheet_name="Extraction", index=False)
        buf.seek(0)
        st.download_button(
            label="Download xlsx",
            data=buf,
            file_name=f"rx_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col2:
        csv = edited.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download csv",
            data=csv,
            file_name=f"rx_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    flagged_count = int((edited.get("Needs Review", pd.Series(dtype=str)).astype(str).str.upper() == "YES").sum())
    st.metric("Rows flagged for review", flagged_count)
