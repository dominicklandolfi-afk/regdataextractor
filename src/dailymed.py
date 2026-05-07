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


# Words a non-expert user is likely to type alongside a drug name that
# DailyMed's drug_name parameter does not understand. Stripping these
# turns "NyQuil pill form medicine" -> "NyQuil" and "Excedrin tablets" ->
# "Excedrin", both of which DailyMed matches. Conservative on purpose: we
# do NOT strip "extra strength", "PM", "max", etc. because those carry
# real meaning for finding the right SPL.
QUERY_NOISE_WORDS = frozenset({
    "pill", "pills", "tablet", "tablets", "capsule", "capsules",
    "softgel", "softgels", "gelcap", "gelcaps", "liquicap", "liquicaps",
    "caplet", "caplets",
    "liquid", "liquids", "syrup", "solution", "suspension", "elixir",
    "cream", "ointment", "lotion", "foam", "paste", "gel",
    "spray", "aerosol", "inhaler", "inhalation",
    "patch", "patches", "powder", "powders", "drops", "drop",
    "injection", "injectable", "suppository", "lozenge",
    "form", "forms", "medicine", "medicines", "medication",
    "medications", "drug", "drugs", "product", "products",
    "pharmaceutical", "pharmaceuticals",
    "otc", "brand", "generic", "prescription", "rx",
})


def _sanitize_query(query: str) -> str:
    """Strip filler words a user might type alongside the drug name.
    Returns the original query if sanitization would empty it."""
    if not query:
        return query
    tokens = query.split()
    kept = [t for t in tokens if t.lower().strip(".,;:!?") not in QUERY_NOISE_WORDS]
    cleaned = " ".join(kept).strip()
    return cleaned or query


def _build_search_candidates(query: str) -> list[str]:
    """Order DailyMed search candidates from most-specific to most-permissive.

    1. The raw query (what the user typed).
    2. The sanitized query with noise words stripped.
    3. The first significant token from the sanitized query.

    Duplicates and short fragments are filtered. The first hit wins."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        v = (value or "").strip()
        if not v:
            return
        key = v.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(v)

    _add(query)
    _add(_sanitize_query(query))
    sanitized = _sanitize_query(query)
    tokens = sanitized.split()
    if tokens and len(tokens[0]) >= 3:
        _add(tokens[0])
    return candidates


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
    # rfind so that titles like "GENTLE CARE KIT (ZINC OXIDE) KIT" (where
    # the dosage-form keyword also appears in the product name) split on
    # the LAST occurrence, which is reliably the actual dosage form.
    for kw in DOSAGE_FORM_KEYWORDS:
        candidate = -1
        for needle in (" " + kw, "\t" + kw):
            idx = upper.rfind(needle)
            if idx > candidate:
                candidate = idx
        if candidate >= 0:
            split_at = candidate + 1
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


def _score_hit(query: str, title: str) -> int:
    """Higher score = better match. Used to pick the best DailyMed SPL
    when the API returns multiple hits, some of which don't actually
    contain the query (e.g., 'Desitin' returning 'GENTLE CARE KIT' as
    its first hit because the kit happens to bundle Desitin)."""
    if not title:
        return 0
    q = query.strip().lower()
    t = title.lower()
    if not q:
        return 0
    score = 0
    if q in t:
        score += 100
    # Word-boundary bonus: query appears as its own token, not glued
    # inside another word.
    tokens = re.findall(r"[A-Za-z0-9]+", t)
    q_tokens = re.findall(r"[A-Za-z0-9]+", q)
    if q_tokens and all(qt in tokens for qt in q_tokens):
        score += 50
    # First-token bonus: the query is the first word of the title
    # (typical for brand-name searches like 'Excedrin').
    if t.startswith(q):
        score += 25
    return score


def search_spl(query: str) -> Optional[dict]:
    """Search DailyMed by NDC or drug name. Returns the best-matching SPL
    or None. When the API returns several hits, prefer one whose title
    actually contains the query string."""
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
    if _is_ndc(query):
        return hits[0]
    scored = [(idx, _score_hit(query, hit.get("title", "")), hit) for idx, hit in enumerate(hits)]
    # Highest score first; on tie, original API order wins.
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[0][2]


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
    """High-level lookup: search and assemble DailyMedData.

    Tries several progressively-cleaner versions of the user's query so
    that brand names typed alongside dosage forms ('NyQuil pill form
    medicine') still resolve. NDC queries skip sanitization."""
    record = DailyMedData()

    if _is_ndc(query):
        candidates = [query]
    else:
        candidates = _build_search_candidates(query)

    hit = None
    for candidate in candidates:
        hit = search_spl(candidate)
        if hit:
            break
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
