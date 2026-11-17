import json
from pathlib import Path
from google.adk.agents import LlmAgent
from .config import MODEL_NAME

GUIDELINES_FILE = Path(__file__).parent.parent / "data" / "coding_guidelines.json"

def load_coding_guidelines() -> dict:
    if GUIDELINES_FILE.exists():
        with open(GUIDELINES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rules": []}

guidelines_data = load_coding_guidelines()
guidelines_formatted = json.dumps(guidelines_data, indent=2)

GUIDELINES_VALIDATION_INSTRUCTION = f"""You are an expert Coding Guidelines and Validation Agent.
Your responsibility is to validate the proposed disease and ICD code provided by Agent 2 against the official project medical coding guidelines.

Proposed Coding from Agent 2:
{{coding}}

Local Coding Guidelines:
{guidelines_formatted}

Instructions:
1. Examine the proposed disease and ICD code in the proposed coding.
2. Cross-check against the Local Coding Guidelines rules above.
3. Check for:
   - Whether the proposed ICD code accurately matches the assigned disease category.
   - Any obvious coding inconsistencies (e.g. Asthma assigned a COPD code J44).
   - Conflicting diagnostic information or duplicate/invalid/empty codes.
4. Output a strict JSON object with the following fields:
   - "status": String ("valid" if coding passes all guidelines, or "needs_review" if an issue/inconsistency is found)
   - "disease": String (The disease being validated)
   - "icd_code": String (The ICD code being validated)
   - "issues": List of Strings (Empty array [] if valid, or a list of specific warning/error messages if needs_review)

Rules:
- Remain strictly objective and concise.
- Return ONLY the raw JSON object without markdown formatting or extra text.
"""

guidelines_agent = LlmAgent(
    name="CodingGuidelinesValidationAgent",
    model=MODEL_NAME,
    instruction=GUIDELINES_VALIDATION_INSTRUCTION,
    description="Validates proposed disease and ICD codes against official local coding guidelines.",
    output_key="validation"
)
