from google.adk.agents import SequentialAgent
from .evidence_agent import evidence_agent
from .disease_icd_agent import disease_icd_agent
from .guidelines_agent import guidelines_agent
from .reconciliation_agent import reconciliation_agent

root_agent = SequentialAgent(
    name="MedicalCodingPipeline",
    sub_agents=[
        evidence_agent,
        disease_icd_agent,
        guidelines_agent,
        reconciliation_agent
    ],
    description="Sequential multi-agent medical coding pipeline implementing Evidence Extraction -> Disease/ICD Retrieval -> Guidelines Validation -> Final Reconciliation."
)

agent = root_agent
