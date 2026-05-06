"""PubChem PUG REST + PUG VIEW client.

Cross-references active ingredient physical properties against the
Perplexity / SDS extraction. PubChem returns properties for the *pure
compound*, not the formulation: aspirin (the molecule) has a flash point
near 250 C, a tablet of aspirin does not. Treat values from this client
as sanity checks on the Perplexity output, not ground truth. The
orchestrator only flags disagreement when the perplexity value is in a
range the pure-compound data clearly contradicts (e.g., perplexity says
"None, No Flash Point" but PubChem reports a low-temperature flash for
the active ingredient).

API docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
No API key required. Public NIH service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import requests

REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"
TIMEOUT = 30


@dataclass
class PubChemData:
    """Physical properties for the active ingredient (pure compound)."""

    cid: Optional[int] = None
    name: Optional[str] = None
    boiling_point_text: Optional[str] = None
    flash_point_text: Optional[str] = None
    water_solubility_text: Optional[str] = None
    ph_text: Optional[str] = None
    compound_url: Optional[str] = None

    def has_any(self) -> bool:
        return any(
            [
                self.boiling_point_text,
                self.flash_point_text,
                self.water_solubility_text,
                self.ph_text,
            ]
        )


def _resolve_cid(name: str) -> Optional[int]:
    if not name:
        return None
    try:
        resp = requests.get(
            f"{REST}/compound/name/{quote(name)}/cids/JSON", timeout=TIMEOUT
        )
        if resp.status_code != 200:
            return None
        cids = resp.json().get("IdentifierList", {}).get("CID") or []
        return cids[0] if cids else None
    except (requests.RequestException, ValueError):
        return None


def _fetch_section(cid: int, heading: str) -> Optional[str]:
    """Pull the first textual value under a named heading from PUG VIEW."""
    try:
        resp = requests.get(
            f"{VIEW}/{cid}/JSON",
            params={"heading": heading},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        return _first_value(resp.json())
    except (requests.RequestException, ValueError):
        return None


def _first_value(node) -> Optional[str]:
    """Walk a PUG VIEW response tree and return the first concrete value."""
    if isinstance(node, dict):
        if "Information" in node and isinstance(node["Information"], list):
            for info in node["Information"]:
                value = info.get("Value", {}) or {}
                strs = value.get("StringWithMarkup") or []
                if strs:
                    txt = (strs[0] or {}).get("String")
                    if txt:
                        return txt
                nums = value.get("Number")
                if nums:
                    units = value.get("Unit", "") or ""
                    return f"{nums[0]} {units}".strip()
        for v in node.values():
            found = _first_value(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _first_value(item)
            if found:
                return found
    return None


def lookup(active_ingredient: Optional[str]) -> PubChemData:
    """High-level lookup: name -> CID -> physical property strings."""
    record = PubChemData(name=active_ingredient or None)
    if not active_ingredient:
        return record
    cid = _resolve_cid(active_ingredient)
    if cid is None:
        return record
    record.cid = cid
    record.compound_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    record.boiling_point_text = _fetch_section(cid, "Boiling Point")
    record.flash_point_text = _fetch_section(cid, "Flash Point")
    record.water_solubility_text = _fetch_section(cid, "Solubility")
    record.ph_text = _fetch_section(cid, "pH")
    return record


def lookup_multi(ingredients: list[str]) -> list[PubChemData]:
    """Look up every active ingredient. Combination products such as
    Excedrin (acetaminophen + aspirin + caffeine) need each ingredient
    cross-referenced separately because PubChem keys on a single CID per
    compound. Returns one PubChemData per ingredient (CID may be None
    when PubChem has no match)."""
    return [lookup(ing) for ing in ingredients if ing]


# ----------- agreement helpers used by the orchestrator -----------

# Enum range -> (lo, hi) in Celsius. None means open-ended on that side.
BOILING_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "<=20C (68F)": (None, 20.0),
    "20.1C (68.1F) - 35C (95F)": (20.1, 35.0),
    ">35C (95F) - 37.7C (99.9F)": (35.0, 37.7),
    ">37.7C (99.9F)": (37.7, None),
}

FLASH_RANGES: dict[str, tuple[Optional[float], Optional[float]]] = {
    "<23C": (None, 23.0),
    ">=23C and <38C": (23.0, 38.0),
    ">=38C and <=60C": (38.0, 60.0),
    ">60C and <93C": (60.0, 93.0),
    ">=93C and <=815C": (93.0, 815.0),
}


def parse_temp_c(text: Optional[str]) -> Optional[float]:
    """Extract first temperature in Celsius from a free-text PubChem string."""
    if not text:
        return None
    c_match = re.search(r"(-?\d+(?:\.\d+)?)\s*[°\s]*C", text)
    if c_match:
        try:
            return float(c_match.group(1))
        except ValueError:
            pass
    f_match = re.search(r"(-?\d+(?:\.\d+)?)\s*[°\s]*F", text)
    if f_match:
        try:
            return (float(f_match.group(1)) - 32.0) * 5.0 / 9.0
        except ValueError:
            pass
    return None


def _temp_in_range(temp_c: float, range_label: str, ranges: dict) -> bool:
    bounds = ranges.get(range_label)
    if not bounds:
        return False
    lo, hi = bounds
    if lo is not None and temp_c < lo:
        return False
    if hi is not None and temp_c > hi:
        return False
    return True


def flash_point_agreement(perplexity_value: Optional[str], pubchem_text: Optional[str]) -> Optional[bool]:
    """True = PubChem supports perplexity's range. False = clear conflict.
    None when undecidable (no PubChem data, unparseable, or numeric on both sides).
    """
    if not perplexity_value or not pubchem_text:
        return None
    temp = parse_temp_c(pubchem_text)
    if temp is None:
        return None
    if perplexity_value == "None, No Flash Point":
        return temp >= 93.0
    if perplexity_value == "Not Tested/Unknown":
        return None
    if perplexity_value in FLASH_RANGES:
        return _temp_in_range(temp, perplexity_value, FLASH_RANGES)
    return None


def boiling_point_agreement(perplexity_value: Optional[str], pubchem_text: Optional[str]) -> Optional[bool]:
    if not perplexity_value or not pubchem_text:
        return None
    temp = parse_temp_c(pubchem_text)
    if temp is None:
        return None
    if perplexity_value in BOILING_RANGES:
        return _temp_in_range(temp, perplexity_value, BOILING_RANGES)
    return None
