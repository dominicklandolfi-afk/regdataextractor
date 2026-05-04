"""Storage for extracted product records.

Backed by either local SQLite (default, for development) or Turso / libSQL
(for production on Streamlit Cloud). Selection is automatic:

- If `TURSO_DATABASE_URL` is set (env var or st.secrets), use libSQL.
- Otherwise, write to `data/extractions.db` on the local filesystem.

Both backends speak the same SQL, so the rest of the app does not care
which one is in use. The schema is wide (one column per field) so the
Data Manager can run familiar `SELECT name, value FROM products` queries.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .extractor import to_flat_dict
from .schema import ProductRecord, SDSExtraction

LOCAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "extractions.db"

META_COLS: tuple[str, ...] = (
    "label",
    "input_query",
    "saved_at",
    "needs_review",
    "review_reasons",
    "ndc",
    "product_name",
    "generic_name",
    "manufacturer",
    "dosage_form",
    "dailymed_url",
)


def _sds_field_names() -> tuple[str, ...]:
    return tuple(SDSExtraction.model_fields.keys())


def _sds_columns() -> tuple[str, ...]:
    cols: list[str] = []
    for fname in _sds_field_names():
        cols.append(fname)
        cols.append(f"{fname}_confidence")
        cols.append(f"{fname}_evidence")
        cols.append(f"{fname}_source")
    return tuple(cols)


def all_columns() -> tuple[str, ...]:
    return META_COLS + _sds_columns()


def _turso_creds() -> tuple[str, str]:
    url = os.getenv("TURSO_DATABASE_URL", "")
    token = os.getenv("TURSO_AUTH_TOKEN", "")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("TURSO_DATABASE_URL", "") or url
            token = st.secrets.get("TURSO_AUTH_TOKEN", "") or token
        except Exception:
            pass
    return url, token


def backend() -> str:
    """Return 'turso' or 'sqlite' depending on configured credentials."""
    url, _ = _turso_creds()
    return "turso" if url else "sqlite"


def _connect():
    url, token = _turso_creds()
    if url:
        try:
            import libsql
        except ImportError as exc:
            raise RuntimeError(
                "TURSO_DATABASE_URL is set but `libsql` is not installed. "
                "Add it to requirements.txt and reinstall."
            ) from exc
        conn = libsql.connect(url, auth_token=token)
    else:
        LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(LOCAL_DB_PATH)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn) -> None:
    col_defs = []
    for col in all_columns():
        if col == "label":
            col_defs.append(f'"{col}" TEXT PRIMARY KEY')
        elif col.endswith("_confidence"):
            col_defs.append(f'"{col}" INTEGER')
        else:
            col_defs.append(f'"{col}" TEXT')
    conn.execute(f'CREATE TABLE IF NOT EXISTS products ({", ".join(col_defs)})')

    info = conn.execute("PRAGMA table_info(products)").fetchall()
    existing = {row[1] for row in info}
    for col in all_columns():
        if col not in existing:
            col_type = "INTEGER" if col.endswith("_confidence") else "TEXT"
            conn.execute(f'ALTER TABLE products ADD COLUMN "{col}" {col_type}')
    conn.commit()


_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_\- ]+")


def _slugify(text: str) -> str:
    cleaned = _LABEL_SAFE.sub("", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "record"


def _unique_label(conn, base: str) -> str:
    candidate = base
    n = 2
    while conn.execute(
        "SELECT 1 FROM products WHERE label = ?", (candidate,)
    ).fetchone() is not None:
        candidate = f"{base}{n}"
        n += 1
    return candidate


def _record_to_row(record: ProductRecord, label: str) -> dict:
    flat = to_flat_dict(record)
    row: dict = {
        "label": label,
        "input_query": record.input_query,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "needs_review": flat.get("Needs Review", ""),
        "review_reasons": flat.get("Review Reasons", ""),
        "ndc": flat.get("NDC", ""),
        "product_name": flat.get("Product Name", ""),
        "generic_name": flat.get("Generic Name", ""),
        "manufacturer": flat.get("Manufacturer", ""),
        "dosage_form": flat.get("Dosage Form", ""),
        "dailymed_url": flat.get("DailyMed URL", ""),
    }
    if record.sds is not None:
        for fname, field in record.sds.model_dump().items():
            row[fname] = field.get("value") or ""
            row[f"{fname}_confidence"] = field.get("confidence")
            row[f"{fname}_evidence"] = field.get("evidence_quote") or ""
            row[f"{fname}_source"] = field.get("source_url") or ""
    return row


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def save_record(record: ProductRecord) -> str:
    """Persist one record. Returns the assigned label."""
    base = _slugify(record.input_query)
    conn = _connect()
    label = _unique_label(conn, base)
    row = _record_to_row(record, label)
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    col_sql = ",".join(f'"{c}"' for c in cols)
    conn.execute(
        f'INSERT INTO products ({col_sql}) VALUES ({placeholders})',
        [_clean(row[c]) for c in cols],
    )
    conn.commit()
    return label


def save_records(records: Iterable[ProductRecord]) -> list[str]:
    return [save_record(r) for r in records]


def list_records() -> pd.DataFrame:
    conn = _connect()
    cur = conn.execute('SELECT * FROM products ORDER BY saved_at DESC')
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def delete_records(labels: Iterable[str]) -> int:
    labels = list(labels)
    if not labels:
        return 0
    conn = _connect()
    placeholders = ",".join(["?"] * len(labels))
    cur = conn.execute(
        f'DELETE FROM products WHERE label IN ({placeholders})', labels
    )
    conn.commit()
    return cur.rowcount


def replace_table(df: pd.DataFrame) -> int:
    """Overwrite the products table with the supplied dataframe."""
    expected = set(all_columns())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    labels = df["label"].astype(str).str.strip()
    if (labels == "").any():
        raise ValueError("Every row must have a non-empty label.")
    if labels.duplicated().any():
        dupes = labels[labels.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate labels: {dupes}")

    ordered = list(all_columns())
    df = df[ordered].copy()

    conn = _connect()
    conn.execute("DELETE FROM products")
    col_sql = ",".join(f'"{c}"' for c in ordered)
    placeholders = ",".join(["?"] * len(ordered))
    rows = [tuple(_clean(row[c]) for c in ordered) for _, row in df.iterrows()]
    conn.executemany(
        f'INSERT INTO products ({col_sql}) VALUES ({placeholders})',
        rows,
    )
    conn.commit()
    return len(df)


def run_sql(sql: str) -> tuple[Optional[pd.DataFrame], int, Optional[str]]:
    """Execute one SQL statement.

    Returns (dataframe_or_None, rowcount, error_message_or_None).
    SELECT-style statements return a dataframe. UPDATE/DELETE/INSERT
    return rowcount only. Errors are returned, not raised.
    """
    sql = sql.strip().rstrip(";")
    if not sql:
        return None, 0, "Empty statement."
    try:
        conn = _connect()
        cur = conn.execute(sql)
        if cur.description is not None:
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            conn.commit()
            return pd.DataFrame(rows, columns=cols), len(rows), None
        conn.commit()
        return None, cur.rowcount, None
    except (sqlite3.Error, Exception) as exc:
        return None, 0, str(exc)
