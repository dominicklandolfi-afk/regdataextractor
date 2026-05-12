"""Pydantic models matching the regulatory portal dropdown options.

Every Literal type below is copied verbatim from the spec sheet
"Process for AI-Driven Regulatory Data Automation" so the AI output
plugs directly into the portal without translation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

ProductType = Literal[
    "Prescription Pharmaceutical (Aerosol)",
    "Prescription Pharmaceutical (Liquid)",
    "Prescription Pharmaceutical (Solid)",
    "Prescription Pharmaceutical with Liquid Core",
]

SecondaryPhysicalState = Literal[
    "lyophilized film", "Aerosol", "Aggregate coated by liquid",
    "Bag-on-valve (BOV)", "Biosoluble Ceramic Insulation",
    "Bonded, fibrous glass web", "Capsule/Tablet", "Compressed gas",
    "Compressed liquefied gas", "Cream/Lotion", "Cryogenic Liquid",
    "Emulsion", "Fabric (no free liquid)", "Flaked", "Foam", "Frozen",
    "Gas", "Gas dissolved in liquid", "Gas only", "Gel", "Glass bead",
    "Granular", "Inhaler", "Liquid", "Liquid (Sterile Diluent)",
    "Liquid coating on stickers", "Liquid in cylinder - Gas generated",
    "Liquid spray", "Liquified gas", "monofilament",
    "No data available", "No information available", "Oil", "Ointment",
    "Paper", "Paste", "Pellets", "Powder(s)", "Pre-moistened towelette",
    "Pre-Moistened Towelette (free liquids)",
    "Pre-Moistened Towelette (no free liquids)", "Pump Spray",
    "Semi-Solid", "Solid", "Solid containing liquid",
    "Solid Gel Consistency", "Solid spray",
    "Solid, (white to off-white) powder or cake, lyophilized",
    "Solid, powder", "Solid, powder (Drug for Injection)", "Suspension",
    "Very viscous", "Viscous liquid", "Wax for candles",
]

WaterSolubility = Literal[
    "Cloth not soluble", "Completely soluble", "Dispersible",
    "Immiscible", "Insoluble in water", "Miscible", "Miscible in water",
    "Moderately soluble", "Mostly soluble", "Negligible",
    "No data available", "Partially soluble", "Soluble in water",
]

BoilingPointRange = Literal[
    "<=20C (68F)",
    "20.1C (68.1F) - 35C (95F)",
    ">35C (95F) - 37.7C (99.9F)",
    ">37.7C (99.9F)",
    "Not tested/Unknown",
]

FlashPointRange = Literal[
    "<23C",
    ">=23C and <38C",
    ">=38C and <=60C",
    ">60C and <93C",
    ">=93C and <=815C",
    "Not Tested/Unknown",
    "None, No Flash Point",
]

FlashPointMethod = Literal[
    "Closed cup method", "Open cup method", "Not applicable/available"
]

PHRange = Literal[
    "<= 2", "2.1 - 3.9", "4 - 6.9", "7 (Neutral)",
    "7.1 - 9.9", "10 - 12.4", ">=12.5", "Not tested/Unknown",
]

StorageTempRange = Literal[
    "Controlled room temperature of 20C to 25C",
    "Cool Storage of 8C to 15C",
    "Freezer Storage of -25C to -26C",
    "Kit: content temperature storage and handling requirements may vary",
    "Refrigerator Storage of 2C to 8C",
]

TransportRegulatedStatus = Literal[
    "Yes, Agree",
    "No, not regulated for transportation due to an exception",
    "No, not regulated",
]

DOTMode = Literal[
    "Limited quantity", "Consumer commodity", "Fully regulated",
    "Not Shipped via DOT",
]

IMDGMode = Literal[
    "Limited quantity", "Fully regulated", "Not Shipped via IMDG",
]

IATAMode = Literal[
    "Limited quantity", "Consumer commodity", "Fully regulated",
    "Not Shipped via IATA",
]

HazardClass = Literal["2.1", "2.2", "2.3", "3", "Not Applicable"]

PackingGroup = Literal["I", "II", "III", "None", "Not Applicable"]

RCRAClassification = Literal[
    "D001 Hazardous Waste under RCRA",
    "D003 Hazardous Waste under RCRA",
    "Not classified as D001 or D003 Hazardous Waste under RCRA",
]


class ExtractedField(BaseModel):
    """One regulatory data point with provenance."""

    value: Optional[str] = Field(
        None,
        description="The extracted value, must match the allowed enum exactly when applicable.",
    )
    confidence: int = Field(
        ge=0, le=100,
        description="0 to 100. Below 60 means the script flags the row for human review.",
    )
    evidence_quote: Optional[str] = Field(
        None,
        description="Direct quote from the SDS or source document supporting the value. "
                    "Empty if no source was found.",
    )
    source_url: Optional[str] = Field(
        None,
        description="URL of the SDS or page the value came from.",
    )


class SDSExtraction(BaseModel):
    """Fields the AI is responsible for extracting from SDS / web research.

    DailyMed-pulled fields (manufacturer, dosage form, etc.) are populated
    separately by the DailyMed client and merged in the orchestrator.
    """

    product_type: ExtractedField = Field(
        description=f"Choose one of: {ProductType.__args__}"
    )
    secondary_physical_state: ExtractedField = Field(
        description=f"Choose one of: {SecondaryPhysicalState.__args__}"
    )
    ph_mixture_extreme: ExtractedField = Field(
        description="Yes or No. Yes if mixed 1:1 with water gives pH<=2 or pH>=12.5. "
                    "Default No when SDS does not flag it."
    )
    ph_value: ExtractedField = Field(
        description="Numeric pH if the SDS states one (e.g., '7.2'), "
                    "otherwise the matching range from: "
                    f"{PHRange.__args__}"
    )
    water_solubility: ExtractedField = Field(
        description=f"Choose one of: {WaterSolubility.__args__}"
    )
    boiling_point_c: ExtractedField = Field(
        description="Numeric boiling point in Celsius if stated, "
                    f"otherwise the matching range from: {BoilingPointRange.__args__}"
    )
    flash_point_c: ExtractedField = Field(
        description="Numeric flash point in Celsius if stated, "
                    f"otherwise the matching range from: {FlashPointRange.__args__}. "
                    "Use 'None, No Flash Point' for non-flammable products."
    )
    flash_point_method: ExtractedField = Field(
        description=f"Choose one of: {FlashPointMethod.__args__}"
    )
    storage_temp_range: ExtractedField = Field(
        description=f"Choose one of: {StorageTempRange.__args__}"
    )
    transport_regulated: ExtractedField = Field(
        description=f"Choose one of: {TransportRegulatedStatus.__args__}. "
                    "Aerosols are typically 'Yes, Agree'; most liquids and "
                    "solids are 'No, not regulated'."
    )
    un_number: ExtractedField = Field(
        description="UN number, e.g., 'UN1950' for aerosols. Empty string if not regulated."
    )
    proper_shipping_name: ExtractedField = Field(
        description="e.g., 'Aerosols' or 'Aerosols, flammable, n.o.s.'. "
                    "Empty if not regulated."
    )
    hazard_class: ExtractedField = Field(
        description=f"Choose one of: {HazardClass.__args__}"
    )
    packing_group: ExtractedField = Field(
        description=f"Choose one of: {PackingGroup.__args__}"
    )
    rcra_classification: ExtractedField = Field(
        description=f"Choose one of: {RCRAClassification.__args__}. "
                    "Default to 'Not classified as D001 or D003 Hazardous Waste under RCRA' "
                    "for non-flammable products with no hazardous ingredients."
    )


class DailyMedData(BaseModel):
    """Fields populated from the DailyMed REST API. None when unavailable."""

    ndc: Optional[str] = None
    product_name: Optional[str] = None
    generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    distributor: Optional[str] = None
    dosage_form: Optional[str] = None
    dea_schedule: Optional[str] = None
    marketing_category: Optional[str] = None
    marketing_end_date: Optional[str] = None
    nda_number: Optional[str] = None
    active_ingredients: list[str] = Field(default_factory=list)
    spl_url: Optional[str] = None


class ProductRecord(BaseModel):
    """One row of the final output spreadsheet."""

    input_query: str
    dailymed: DailyMedData
    sds: Optional[SDSExtraction] = None
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    sources: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-field provenance: maps SDS field name to the list "
                    "of sources that contributed (e.g., ['perplexity', 'dot']). "
                    "Populated by the orchestrator after cross-source enrichment.",
    )
    category: Optional[str] = Field(
        default=None,
        description="Regulatory product category assigned by the classifier "
                    "(ORAL_SOLID, TOPICAL_ALCOHOL, AEROSOL_PROPELLED, etc.). "
                    "Drives the deterministic-default profile for the row.",
    )
