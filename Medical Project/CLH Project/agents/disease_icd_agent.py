import json
from pathlib import Path
from google.adk.agents import LlmAgent
from .config import MODEL_NAME

ICD_CODES_FILE = Path(__file__).parent.parent / "data" / "icd_codes.json"

def load_icd_lookup_table() -> dict:
    if ICD_CODES_FILE.exists():
        with open(ICD_CODES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

icd_table = load_icd_lookup_table()
icd_table_formatted = json.dumps(icd_table, indent=2)

DISEASE_ICD_INSTRUCTION = f"""You are an expert Disease Identification and ICD Code Retrieval Agent.
Your task is to analyze the extracted clinical evidence provided below and match it to the single most appropriate disease/condition and its corresponding ICD code using ONLY the provided local ICD lookup table.

Extracted Clinical Evidence from Agent 1:
{{evidence}}

Available Local ICD Lookup Table:
{icd_table_formatted}

Instructions:
1. Identify the primary medical condition described by the symptoms, diagnoses, history, treatments, and findings in the evidence.
2. Match that condition to the closest matching entry in the Available Local ICD Lookup Table above.
3. Retrieve the exact ICD code associated with that entry in the table.
4. Output a strict JSON object with keys:
   - "disease": String (The identified disease/condition name as matched in the table)
   - "icd_code": String (The exact ICD code from the table)
   - "rationale": String (A concise single-sentence explanation of why this condition and code match the evidence)

Rules:
- You MUST use the exact ICD code given in the lookup table for the matched condition.
- Return ONLY the raw JSON object without markdown code blocks or extra text.
"""

disease_icd_agent = LlmAgent(
    name="DiseaseICDAgent",
    model=MODEL_NAME,
    instruction=DISEASE_ICD_INSTRUCTION,
    description="Identifies the primary disease and retrieves its corresponding ICD code from the local lookup table.",
    output_key="coding"
)
