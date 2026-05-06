"""DOT Hazardous Materials Table (49 CFR 172.101) lookup.

Authoritative federal source for UN number, proper shipping name, hazard
class, and packing group. The bundled subset is curated for prescription
pharmaceutical products and is limited to entries that map cleanly to the
schema's allowed hazard_class enum {2.1, 2.2, 2.3, 3, Not Applicable}.

49 CFR 172.101 is public domain federal regulation. Source of truth:
https://www.ecfr.gov/current/title-49/subtitle-B/chapter-I/subchapter-C/part-172/subpart-B/section-172.101

Adding more entries: append to TABLE below. The orchestrator keys lookups
on the UN number returned by Perplexity, with proper-shipping-name as a
fallback. When multiple entries share a UN (e.g., UN1950 flammable vs
non-flammable aerosols), the orchestrator passes a hazard_class hint to
disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CITATION = "49 CFR 172.101 Hazardous Materials Table"


@dataclass(frozen=True)
class DOTEntry:
    un_number: str
    proper_shipping_name: str
    hazard_class: str
    packing_group: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "un_number": self.un_number,
            "proper_shipping_name": self.proper_shipping_name,
            "hazard_class": self.hazard_class,
            "packing_group": self.packing_group,
            "notes": self.notes,
        }


TABLE: list[DOTEntry] = [
    DOTEntry(
        "UN1950", "Aerosols, flammable", "2.1", "None",
        "Most prescription metered-dose inhalers with HFA propellants",
    ),
    DOTEntry(
        "UN1950", "Aerosols, non-flammable", "2.2", "None",
        "Compressed-gas inhalers without flammable propellants",
    ),
    DOTEntry(
        "UN1170", "Ethanol", "3", "II",
        "Pure ethanol or solutions >70% ethanol",
    ),
    DOTEntry(
        "UN1170", "Ethanol solution", "3", "III",
        "Ethanol-based liquid formulations 24-70%",
    ),
    DOTEntry("UN1219", "Isopropanol", "3", "II", ""),
    DOTEntry("UN1090", "Acetone", "3", "II", ""),
    DOTEntry("UN1230", "Methanol", "3", "II", ""),
    DOTEntry(
        "UN3248", "Medicine, liquid, flammable, toxic, n.o.s.",
        "3", "II", "Flammable liquid prescription medicine",
    ),
    DOTEntry(
        "UN3500", "Chemical under pressure, n.o.s.",
        "2.2", "None", "Some non-flammable propellant inhalers",
    ),
    DOTEntry("UN1011", "Butane", "2.1", "None", ""),
    DOTEntry("UN1075", "Petroleum gases, liquefied", "2.1", "None", ""),
    DOTEntry("UN1978", "Propane", "2.1", "None", ""),
    DOTEntry(
        "UN1993", "Flammable liquid, n.o.s.", "3", "II",
        "Generic flammable liquid catch-all",
    ),
    DOTEntry(
        "UN1993", "Flammable liquid, n.o.s.", "3", "III",
        "Generic flammable liquid catch-all (lower hazard)",
    ),
]


def _normalize_un(un_number: str) -> str:
    target = un_number.strip().upper().replace(" ", "")
    if not target.startswith("UN") and target.isdigit():
        target = f"UN{target}"
    return target


def lookup_all_by_un(un_number: str) -> list[DOTEntry]:
    if not un_number:
        return []
    target = _normalize_un(un_number)
    return [e for e in TABLE if e.un_number == target]


def lookup_by_shipping_name(name: str) -> Optional[DOTEntry]:
    if not name:
        return None
    target = name.strip().lower()
    for entry in TABLE:
        if entry.proper_shipping_name.lower() == target:
            return entry
    for entry in TABLE:
        if target in entry.proper_shipping_name.lower():
            return entry
    return None


def lookup(
    un_number: Optional[str] = None,
    shipping_name: Optional[str] = None,
    hazard_class_hint: Optional[str] = None,
) -> Optional[DOTEntry]:
    """Best-effort lookup. UN first, then shipping name. Disambiguates
    multi-entry UN numbers using the hazard class hint when provided."""
    if un_number:
        matches = lookup_all_by_un(un_number)
        if matches:
            if hazard_class_hint:
                for m in matches:
                    if m.hazard_class == hazard_class_hint:
                        return m
            return matches[0]
    if shipping_name:
        return lookup_by_shipping_name(shipping_name)
    return None
