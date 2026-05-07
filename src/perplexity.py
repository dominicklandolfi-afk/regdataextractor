"""Perplexity API client for SDS-derived regulatory fields.

Uses the sonar-pro model with json_schema response format so the model
output conforms to the SDSExtraction Pydantic schema.
Each field returns value + confidence + evidence_quote + source_url.

Docs: https://docs.perplexity.ai/api-reference/chat-completions-post
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests
from pydantic import ValidationError

from .schema import DailyMedData, SDSExtraction

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
TIMEOUT = 120

SYSTEM_PROMPT = """You are a regulatory data analyst extracting safety and \
transportation classifications for pharmaceutical products. The products may \
be prescription (Rx) or over-the-counter (OTC). The product_type enum labels \
read "Prescription Pharmaceutical (...)" but those same labels also apply to \
OTC products; choose by physical form, not by Rx vs OTC status. You search \
the web for the manufacturer's most recent Safety Data Sheet (SDS) and the \
DailyMed Structured Product Label, then fill in the requested fields.

CRITICAL: choose product_type and secondary_physical_state from the DailyMed \
dosage form. Do NOT default to tablet/capsule when the dosage form clearly \
indicates a different physical form. Use this mapping:

  Dosage form keywords            -> product_type                                  -> secondary_physical_state
  TABLET, CAPLET, CAPSULE,        -> "Prescription Pharmaceutical (Solid)"        -> "Capsule/Tablet"
    SOFTGEL, GELCAP
  CREAM, LOTION                   -> "Prescription Pharmaceutical (Liquid)"       -> "Cream/Lotion"
  OINTMENT                        -> "Prescription Pharmaceutical (Liquid)"       -> "Ointment"
  PASTE                           -> "Prescription Pharmaceutical (Liquid)"       -> "Paste"
  GEL                             -> "Prescription Pharmaceutical (Liquid)"       -> "Gel"
  FOAM                            -> "Prescription Pharmaceutical (Liquid)"       -> "Foam"
  SOLUTION, SUSPENSION, SYRUP,    -> "Prescription Pharmaceutical (Liquid)"       -> "Liquid" or "Suspension"
    LIQUID, ELIXIR, DROPS
  AEROSOL, METERED-DOSE INHALER,  -> "Prescription Pharmaceutical (Aerosol)"      -> "Aerosol"
    PROPELLANT SPRAY, INHALER
  POWDER, GRANULES                -> "Prescription Pharmaceutical (Solid)"       -> "Powder(s)" or "Granular"
  PATCH                           -> "Prescription Pharmaceutical (Solid)"       -> closest enum match
  SOFTGEL with liquid fill        -> "Prescription Pharmaceutical with Liquid Core" -> "Solid containing liquid"

If the dosage form is unclear or absent, default to the form implied by the \
product name (e.g., "Desitin Diaper Rash Ointment" -> ointment).

Rules:
- Choose values EXACTLY as written in the allowed enum lists. Do not \
paraphrase or add words. If the schema field describes allowed values \
(e.g., "<23C"), use that exact string.
- Quote the source verbatim in evidence_quote (10 to 40 words). Include \
the SDS section number when relevant (e.g., "Section 9: Boiling Point: \
>200 C").
- Set source_url to the actual SDS PDF URL or DailyMed page, not a search \
results page. If multiple authoritative sources support the same value, \
separate the URLs with "; " (semicolon-space).
- Confidence is 0 to 100:
    90-100 = direct quote from a current manufacturer SDS
    70-89  = stated on DailyMed or distributor SDS
    40-69  = inferred from similar products or partial data
    0-39   = no source found, defaulted
- Default values for non-flammable, non-aerosol products (oral solids, oral \
liquids, topical creams/ointments/lotions, etc.):
    transport_regulated="No, not regulated"
    flash_point_c="None, No Flash Point"
    un_number=""
    proper_shipping_name=""
    hazard_class="Not Applicable"
    packing_group="Not Applicable"
    rcra_classification="Not classified as D001 or D003 Hazardous Waste under RCRA"
  Use confidence ~80 when the SDS confirms non-flammable, ~50 when \
defaulting without an SDS.
- For aerosols (HFA inhalers, propellant sprays): transport_regulated="Yes, Agree", \
un_number="UN1950", proper_shipping_name typically "Aerosols", hazard_class \
"2.1" (flammable propellant) or "2.2" (non-flammable propellant).
- For products containing alcohol or other flammable solvents (mouthwash, \
hand sanitizer, some oral liquids), expect flash points below 93 C and \
classify accordingly.
- If the SDS does not state a value, return value=null and write a brief \
explanation in evidence_quote ("Not stated on SDS Section 9").
"""


def _user_prompt(dm: DailyMedData, original_query: str) -> str:
    bits = [f"Product query: {original_query}"]
    if dm.product_name:
        bits.append(f"DailyMed product name: {dm.product_name}")
    if dm.generic_name and dm.generic_name != dm.product_name:
        bits.append(f"Generic name: {dm.generic_name}")
    if dm.manufacturer:
        bits.append(f"Manufacturer: {dm.manufacturer}")
    if dm.dosage_form:
        bits.append(
            f"DOSAGE FORM (use this to choose product_type and "
            f"secondary_physical_state per the system mapping table): "
            f"{dm.dosage_form}"
        )
    if dm.ndc:
        bits.append(f"NDC: {dm.ndc}")
    if dm.spl_url:
        bits.append(f"DailyMed page: {dm.spl_url}")
    bits.append("")
    bits.append(
        "Find the most recent manufacturer Safety Data Sheet for this product "
        "and extract the requested regulatory fields. Prefer the manufacturer's "
        "own PDF over third-party aggregators. If multiple SDSes are returned, "
        "use the most recently dated one. Cite the SDS URL in source_url for "
        "every field that comes from the SDS. Do NOT default product_type or "
        "secondary_physical_state to tablet/capsule values when the dosage form "
        "above indicates a cream, ointment, lotion, gel, paste, solution, "
        "aerosol, powder, or other non-tablet form."
    )
    return "\n".join(bits)


def _schema_for_response_format() -> dict:
    """Build a JSON Schema acceptable to Perplexity from SDSExtraction."""
    schema = SDSExtraction.model_json_schema()
    return schema


def extract_sds_fields(
    dm: DailyMedData,
    original_query: str,
    api_key: Optional[str] = None,
) -> tuple[Optional[SDSExtraction], list[str], Optional[str]]:
    """Call Perplexity and return (parsed_extraction, citations, raw_text).

    parsed_extraction is None if the model output failed validation.
    raw_text holds the unparsed response so the caller can debug.
    """
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set in environment.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(dm, original_query)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": _schema_for_response_format()},
        },
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    citations = data.get("citations") or data.get("search_results") or []
    if citations and isinstance(citations[0], dict):
        citations = [c.get("url") for c in citations if c.get("url")]

    content = data["choices"][0]["message"]["content"]

    try:
        parsed_dict = json.loads(content)
        parsed = SDSExtraction.model_validate(parsed_dict)
    except (json.JSONDecodeError, ValidationError):
        return None, citations, content

    return parsed, citations, content
