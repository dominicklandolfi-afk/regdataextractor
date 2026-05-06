# Regulatory Data Extractor

Automates the extraction of regulatory and safety data for pharmaceutical
products (prescription and OTC). Each product is reconciled across four
sources:

- **DailyMed** for identity and active ingredients (deterministic lookup)
- **Perplexity** for the Safety Data Sheet (transport classification,
  flash point, pH, water solubility, RCRA, etc.)
- **PubChem** for active-ingredient physical properties, cross-checked
  against the SDS values
- **49 CFR 172.101** for transport classification, used to confirm or
  correct the UN number, proper shipping name, hazard class, and packing
  group

Every extracted value comes with a 0 to 100 confidence score, a direct
evidence quote, a source URL, and the list of databases that contributed.
Source agreement raises confidence (typically to 90+); disagreement either
corrects the value (when 49 CFR contradicts Perplexity on transport) or
flags the row for review (when PubChem contradicts the SDS on flash
point for the active ingredient). Rows where any critical field has
confidence below 60 are auto-flagged for review.

## What it produces

A spreadsheet (xlsx or csv) with one row per product and roughly 70
columns: DailyMed identity fields plus 16 SDS-derived fields, each with
its own value, confidence, evidence quote, and source URL.

## First-time setup

1. Open PowerShell and run:

   ```powershell
   cd C:\Claude\projects\rx-regulatory-extractor
   ```

2. The Python virtual environment and dependencies are already installed.
   To re-install on a fresh machine:

   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Make sure `.env` contains your Perplexity API key:

   ```
   PERPLEXITY_API_KEY=pplx-...
   ```

   Get one at https://www.perplexity.ai/account/api ($5 minimum top-up
   covers a few hundred products).

## Run the web UI

```powershell
cd C:\Claude\projects\rx-regulatory-extractor
.venv\Scripts\streamlit.exe run app.py
```

A browser tab opens at `http://localhost:8501`. Paste a list of products
(one per line, NDC numbers preferred), click **Extract regulatory data**,
watch the progress bar, then review and edit the results in the table.
Download as xlsx or csv.

## Run from the command line

```powershell
.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from src.extractor import extract_product, to_flat_dict; print(to_flat_dict(extract_product('atorvastatin')))"
```

## How accuracy works

| Field source | Method | Typical confidence |
|---|---|---|
| Product Name, Generic Name, NDC, Manufacturer, Dosage Form | DailyMed REST API | n/a (deterministic) |
| Product Type, Physical State | Inferred from dosage form | 80 |
| Flash Point, Boiling Point, pH, Water Solubility | Perplexity searches manufacturer SDS | 70 to 95 if SDS found, 40 to 60 if defaulted |
| Transport Classification, UN Number, Hazard Class | Perplexity + dosage form rules | 80 for non-aerosol tablets, 90+ for aerosols with confirmed SDS |
| RCRA Classification | Perplexity + flammability check | 80 default, 90+ with SDS |

Anything below confidence 60 on a critical field flags the row for human
review. The reviewer can read the evidence quote, click the source URL,
and either accept the AI value or correct it before final import to the
portal.

## Project layout

```
rx-regulatory-extractor/
  app.py                  Streamlit UI entry point
  requirements.txt        Python dependencies
  .env                    Perplexity API key (gitignored)
  src/
    schema.py             Pydantic models with all portal dropdown enums
    dailymed.py           DailyMed REST API client
    perplexity.py         Perplexity sonar-pro client with json_schema
    pubchem.py            PubChem PUG REST + PUG VIEW client
    dot_hazmat.py         49 CFR 172.101 lookup table
    extractor.py          Orchestrator + cross-source reconciliation
    storage.py            SQLite / Turso persistence
    data_manager.py       Streamlit page: saved records and edit
    report_builder.py     Streamlit page: preset reports and SQL
  output/                 (xlsx files land here when downloaded)
```

## Adapting to new fields

The spec sheet is "Process for AI-Driven Regulatory Data Automation"
in the user's Drive. If the portal adds or changes a dropdown:

1. Open `src/schema.py`.
2. Find the matching `Literal[...]` type and update the values.
3. Update the `description=` string in `SDSExtraction` so the model
   knows the new constraint.

No other code changes are needed.

## Cost

Perplexity sonar-pro runs about $0.005 to $0.015 per product depending
on context and citation count. A batch of 100 products costs roughly
$1 to $2. DailyMed is free.

## Limitations and roadmap

- Day-one v1 writes to local xlsx. To push results to a shared Google
  Sheet, add `gspread` plus a Google Cloud service account, share the
  target sheet with the service account email, and call
  `gspread.open_by_key(...).sheet1.update(...)`. Easy to add when needed.
- For aerosol products the transport classification often pulls from
  multiple SDS sections; double-check UN# and hazard class rows before
  portal import.
- DailyMed search returns the most recent SPL match; for products with
  many manufacturers, you may want to disambiguate by passing the NDC.
