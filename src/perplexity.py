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
transportation classifications for prescription pharmaceutical products. \
You search the web for the manufacturer's most recent Safety Data Sheet \
(SDS) and the DailyMed Structured Product Label, then fill in the requested \
fields.

Rules:
- Choose values EXACTLY as written in the allowed enum lists. Do not \
paraphrase or add words. If the schema field describes allowed values \
(e.g., "<23C"), use that exact string.
- Quote the source verbatim in evidence_quote (10 to 40 words). Include \
the SDS section number when relevant (e.g., "Section 9: Boiling Point: \
>200 C").
- Set source_url to the actual SDS PDF URL or DailyMed page, not a search \
results page. If multiple authoritative sources support the same value \
(e.g., the SDS plus the DailyMed SPL), separate the URLs with "; " \
(semicolon-space). One URL is fine when only one source applies.
- Confidence is 0 to 100. Use:
    90-100 = direct quote from a current manufacturer SDS
    70-89  = stated on DailyMed or distributor SDS
    40-69  = inferred from similar products or partial data
    0-39   = no source found, defaulted
- For non-aerosol prescription tablets and capsules with no flammable \
ingredients, defaults are: transport_regulated="No, not regulated", \
flash_point_c="None, No Flash Point", un_number="", proper_shipping_name="", \
hazard_class="Not Applicable", packing_group="Not Applicable", \
rcra_classification="Not classified as D001 or D003 Hazardous Waste under RCRA". \
Use confidence ~80 for these defaults when the SDS confirms non-flammable, \
~50 when defaulting without an SDS.
- For aerosols, transport_regulated="Yes, Agree", un_number="UN1950", \
proper_shipping_name typically "Aerosols".
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
        bits.append(f"Dosage form: {dm.dosage_form}")
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
        "every field that comes from the SDS."
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
