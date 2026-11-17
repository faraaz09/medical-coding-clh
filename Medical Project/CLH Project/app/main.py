import os
import sys
import json
import time
import asyncio
import argparse
from typing import Dict, Any

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.config import MODEL_NAME
from agents.root_agent import root_agent
from evaluation.evaluate import main_async as run_evaluation
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

BANNER = """
================================================================================
          MEDICAL CODING: CODE LIKE HUMANS (GOOGLE ADK & GEMINI)
   Sequential Multi-Agent Pipeline for Unstructured Clinical Note Coding
================================================================================
"""

DISCLAIMER = """
[DISCLAIMER]: This software is an educational prototype based on agentic AI research.
It is intended solely for demonstration and decision-support testing and must NOT
be used as a substitute for certified medical coding or professional clinical judgment.
"""

def print_json_block(title: str, content: Any):
    print(f"\n--- [{title}] ---")
    if isinstance(content, dict) or isinstance(content, list):
        print(json.dumps(content, indent=2))
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(content)
    else:
        print(content)

async def process_clinical_note(clinical_note: str) -> Dict[str, Any]:
    """
    Executes the 4-agent sequential ADK pipeline on a single clinical note.
    Returns session state containing all intermediate agent outputs.
    """
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="medical_coding_app", session_service=session_service)

    app_name = "medical_coding_app"
    user_id = "clinical_user"

    print("\nInitializing 4-Agent ADK Sequential Workflow...")
    print(f"Using Model: {MODEL_NAME}")
    print("\nRunning Agents sequentially:")
    print("  [Agent 1] Evidence Extraction Agent (output_key='evidence')")
    print("  [Agent 2] Disease Identification + ICD Retrieval Agent (output_key='coding')")
    print("  [Agent 3] Coding Guidelines + Validation Agent (output_key='validation')")
    print("  [Agent 4] Final Reconciliation Agent (output_key='final_result')")
    print("\nExecuting pipeline... Please wait.")

    start_t = time.time()
    max_retries = 3
    final_session_id = None

    for attempt in range(max_retries):
        try:
            session_id = f"note_session_{time.time_ns()}"
            final_session_id = session_id

            session = await session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state={"clinical_note": clinical_note}
            )

            new_msg = types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Clinical Note:\n{clinical_note}")]
            )

            async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=new_msg):
                pass
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = (attempt + 1) * 15
                print(f"\n[Rate Limit Notice] Exceeded RPM limit. Waiting {wait_time} seconds before retrying... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                raise e

    elapsed = time.time() - start_t

    updated_session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=final_session_id) if final_session_id else None
    state = updated_session.state if updated_session else {}

    print(f"\nPipeline execution completed in {elapsed:.2f} seconds.")
    return state

def display_pipeline_results(state: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("                        INTERMEDIATE AGENT OUTPUTS")
    print("=" * 80)

    print_json_block("Agent 1 Output — Evidence Extraction (output_key='evidence')", state.get("evidence", {}))
    print_json_block("Agent 2 Output — Disease ID & ICD Retrieval (output_key='coding')", state.get("coding", {}))
    print_json_block("Agent 3 Output — Guidelines & Validation (output_key='validation')", state.get("validation", {}))
    print_json_block("Agent 4 Output — Final Reconciliation (output_key='final_result')", state.get("final_result", {}))

    final_res = state.get("final_result", {})
    if isinstance(final_res, str):
        try:
            final_res = json.loads(final_res)
        except Exception:
            final_res = {}

    print("\n" + "=" * 80)
    print("                           FINAL CODING RESULT")
    print("=" * 80)
    print(f"Final Diagnosis : {final_res.get('final_diagnosis', 'N/A')}")
    print(f"Final ICD Code  : {final_res.get('final_icd_code', 'N/A')}")
    print(f"Validation Status: {final_res.get('status', 'N/A')}")
    print(f"Explanation     : {final_res.get('explanation', 'N/A')}")
    print("=" * 80)

def show_lookup_data():
    icd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/icd_codes.json"))
    guidelines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/coding_guidelines.json"))

    print("\n--- LOCAL ICD LOOKUP TABLE ---")
    if os.path.exists(icd_path):
        with open(icd_path, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print("ICD lookup file not found.")

    print("\n--- LOCAL CODING GUIDELINES ---")
    if os.path.exists(guidelines_path):
        with open(guidelines_path, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print("Coding guidelines file not found.")

async def interactive_menu():
    print(BANNER)
    print(DISCLAIMER)

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Process Single Clinical Note (Mode 1)")
        print("2. Run Local Dataset Evaluation (Mode 2)")
        print("3. View Local ICD Table & Coding Guidelines")
        print("4. Exit")

        try:
            choice = input("\nSelect an option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == "1":
            print("\nEnter or paste the clinical note below (press Enter then Ctrl+Z / Ctrl+D to finish if multi-line, or enter a single line):")
            note_lines = []
            try:
                line = input("> ").strip()
                if line:
                    note_lines.append(line)
            except (EOFError, KeyboardInterrupt):
                pass
            
            note_text = " ".join(note_lines).strip()
            if not note_text:
                print("Error: Clinical note cannot be empty.")
                continue

            state = await process_clinical_note(note_text)
            display_pipeline_results(state)

        elif choice == "2":
            try:
                limit_str = input("\nEnter number of synthetic cases to evaluate (default 10, enter 0 for all 200): ").strip()
                limit = int(limit_str) if limit_str else 10
            except ValueError:
                limit = 10
            
            await run_evaluation(limit=limit, delay=1.0)

        elif choice == "3":
            show_lookup_data()

        elif choice == "4":
            print("\nThank you for using Medical Coding: Code Like Humans!")
            break
        else:
            print("Invalid selection. Please enter 1, 2, 3, or 4.")

def main():
    parser = argparse.ArgumentParser(description="Medical Coding: Code Like Humans Application Entrypoint")
    parser.add_argument("--note", type=str, help="Process a single clinical note directly via CLI")
    parser.add_argument("--eval", action="store_true", help="Run dataset evaluation mode directly via CLI")
    parser.add_argument("--limit", type=int, default=10, help="Number of cases for evaluation mode (default 10)")
    args = parser.parse_args()

    if args.note:
        print(BANNER)
        state = asyncio.run(process_clinical_note(args.note))
        display_pipeline_results(state)
    elif args.eval:
        print(BANNER)
        asyncio.run(run_evaluation(limit=args.limit, delay=1.0))
    else:
        asyncio.run(interactive_menu())

if __name__ == "__main__":
    main()
