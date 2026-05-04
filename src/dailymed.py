"""DailyMed REST API client.

Used to verify the product exists in DailyMed's authoritative database
and pull the basic identity fields (product name, generic name, NDC,
SPL URL). Heavier extraction (SDS-derived fields) is handled by the
Perplexity client.

API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
No API key required.
"""

from __future__ import annotations

import re
from typing import Optional

import requests

from .schema import DailyMedData

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
TIMEOUT = 30


def _is_ndc(query: str) -> bool:
    cleaned = query.replace("-", "").replace(" ", "")
    return cleaned.isdigit() and len(cleaned) in (10, 11)


DOSAGE_FORM_KEYWORDS = (
    "TABLET", "CAPSULE", "SOLUTION", "INJECTION", "INJECTABLE",
    "CREAM", "OINTMENT", "PATCH", "LOTION", "AEROSOL", "SPRAY",
    "INHALANT", "INHALATION", "SUSPENSION", "POWDER", "GEL",
    "SYRUP", "LIQUID", "PASTE", "KIT", "DROPS", "EMULSION",
    "GRANULES", "LOZENGE", "IMPLANT", "PELLET", "SHAMPOO", "SOAP",
    "STICK", "SUPPOSITORY", "TINCTURE", "CONCENTRATE", "WAFER",
    "ELIXIR", "MOUTHWASH", "RINSE", "FOAM", "BAR",
)


def _split_title(title: str) -> dict:
    """Parse a DailyMed SPL title.

    Typical formats:
      "LIPITOR (atorvastatin calcium) tablet, film coated [PFIZER]"
      "ATORVASTATIN CALCIUM TABLET, FILM COATED [CARDINAL HEALTH]"
      "IBUPROFEN DYE FREE TABLET, FILM COATED [CVS PHARMACY]"
    Returns dict with keys: product_name, generic_name, dosage_form, manufacturer.
    """
    out = {"product_name": None, "generic_name": None,
           "dosage_form": None, "manufacturer": None}
    if not title:
        return out

    work = title.strip()

    mfr_match = re.search(r"\s*\[([^\]]+)\]\s*$", work)
    if mfr_match:
        out["manufacturer"] = mfr_match.group(1).strip()
        work = work[:mfr_match.start()].strip()

    paren_match = re.search(r"\(([^)]+)\)", work)
    if paren_match:
        out["generic_name"] = paren_match.group(1).strip()
        work = (work[:paren_match.start()] + work[paren_match.end():]).strip()

    upper = work.upper()
    split_at = None
    for kw in DOSAGE_FORM_KEYWORDS:
        for needle in (" " + kw, "\t" + kw):
            idx = upper.find(needle)
            if idx >= 0:
                split_at = idx + 1
                break
        if split_at is not None:
            break
        if upper.startswith(kw):
            split_at = 0
            break

    if split_at is not None:
        product = work[:split_at].strip().rstrip(",").strip()
        form = work[split_at:].strip().rstrip(",").strip()
        out["product_name"] = product or None
        out["dosage_form"] = form or None
    else:
        out["product_name"] = work or None

    if not out["generic_name"] and out["product_name"]:
        out["generic_name"] = out["product_name"]
    return out


def search_spl(query: str) -> Optional[dict]:
    """Search DailyMed by NDC or drug name. Returns the first SPL hit or None."""
    params = {"pagesize": "5"}
    if _is_ndc(query):
        params["ndc"] = query
    else:
        params["drug_name"] = query
    try:
        resp = requests.get(f"{BASE}/spls.json", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    hits = data.get("data") or []
    if not hits:
        return None
    return hits[0]


def fetch_ndcs(setid: str) -> list[str]:
    try:
        resp = requests.get(f"{BASE}/spls/{setid}/ndcs.json", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []
    payload = data.get("data") or {}
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("ndcs") or []
    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, dict):
            ndc = row.get("ndc") or row.get("ndc11") or row.get("ndc9")
            if ndc:
                out.append(ndc)
    return out


def lookup(query: str) -> DailyMedData:
    """High-level lookup: search and assemble DailyMedData."""
    record = DailyMedData()
    hit = search_spl(query)
    if not hit:
        return record

    setid = hit.get("setid")
    title = hit.get("title", "")
    parts = _split_title(title)

    record.product_name = parts["product_name"]
    record.generic_name = parts["generic_name"]
    record.dosage_form = parts["dosage_form"]
    record.manufacturer = parts["manufacturer"]
    if setid:
        record.spl_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
        ndcs = fetch_ndcs(setid)
        if ndcs:
            record.ndc = ndcs[0]
    if _is_ndc(query) and not record.ndc:
        record.ndc = query
    return record
