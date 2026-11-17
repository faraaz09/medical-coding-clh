from google.adk.agents import LlmAgent
from .config import MODEL_NAME

EVIDENCE_EXTRACTION_INSTRUCTION = """You are an expert Medical Evidence Extraction Agent.
Your responsibility is to carefully read the unstructured clinical note provided in the input message and extract clinically relevant information into a structured JSON object.

Focus on extracting:
1. "symptoms": List of symptoms described (e.g., frequent urination, chest pain, shortness of breath).
2. "diagnoses": List of explicit diagnoses mentioned in the note, if any.
3. "medical_history": List of relevant past medical history or pre-existing conditions.
4. "treatments": List of medications, therapies, or interventions currently prescribed or administered.
5. "relevant_findings": List of vital signs, lab results, or physical examination findings.

Rules:
- Ignore irrelevant administrative or boilerplate text.
- Do NOT determine or guess the final ICD code or diagnosis for coding.
- Return ONLY a valid JSON object matching the keys: "symptoms", "diagnoses", "medical_history", "treatments", "relevant_findings".
- Output raw JSON only.
"""

evidence_agent = LlmAgent(
    name="EvidenceExtractionAgent",
    model=MODEL_NAME,
    instruction=EVIDENCE_EXTRACTION_INSTRUCTION,
    description="Extracts structured clinical evidence from unstructured medical notes.",
    output_key="evidence"
)
