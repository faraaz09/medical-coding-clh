import os
import sys
import json
import time
import asyncio
import argparse
import statistics
import pandas as pd
from typing import Dict, Any, List

# Ensure line-buffered stdout for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Add parent directory to sys.path so agents module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.config import MODEL_NAME, rotate_api_key
from agents.root_agent import root_agent
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

async def run_single_case(runner: Runner, clinical_note: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    Executes the 4-agent ADK sequential pipeline for a single clinical note.
    Records pure API execution latency and extracts exact usage_metadata tokens.
    Hides ground-truth from the pipeline.
    """
    app_name = "medical_coding_eval"
    user_id = "eval_user"

    for attempt in range(max_retries):
        try:
            session_id = f"eval_session_{time.time_ns()}"
            session = await runner.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state={"clinical_note": clinical_note}
            )

            new_msg = types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Clinical Note:\n{clinical_note}")]
            )

            agent_tokens = {}
            t0 = time.perf_counter()

            async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=new_msg):
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    author = getattr(event, "author", f"agent_{len(agent_tokens)}")
                    u = event.usage_metadata
                    agent_tokens[author] = {
                        "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
                        "candidates_tokens": getattr(u, "candidates_token_count", 0) or 0,
                        "thoughts_tokens": getattr(u, "thoughts_token_count", 0) or 0,
                        "total_tokens": getattr(u, "total_token_count", 0) or 0
                    }

            api_latency = time.perf_counter() - t0

            updated_session = await runner.session_service.get_session(app_name=app_name, user_id=user_id, session_id=session.id)
            final_result = updated_session.state.get("final_result", {}) if updated_session else {}

            # Extract predicted ICD code and diagnosis
            if isinstance(final_result, str):
                try:
                    final_dict = json.loads(final_result)
                except Exception:
                    final_dict = {}
            elif isinstance(final_result, dict):
                final_dict = final_result
            else:
                final_dict = {}

            predicted_icd = final_dict.get("final_icd_code", "UNKNOWN")
            predicted_disease = final_dict.get("final_diagnosis", "UNKNOWN")
            status = final_dict.get("status", "UNKNOWN")

            # Aggregate token counts across all agents for this note
            note_prompt = sum(v["prompt_tokens"] for v in agent_tokens.values())
            note_candidates = sum(v["candidates_tokens"] for v in agent_tokens.values())
            note_thoughts = sum(v["thoughts_tokens"] for v in agent_tokens.values())
            note_total = sum(v["total_tokens"] for v in agent_tokens.values())

            return {
                "predicted_icd": predicted_icd,
                "predicted_disease": predicted_disease,
                "status": status,
                "api_latency_seconds": round(api_latency, 4),
                "tokens": {
                    "prompt_tokens": note_prompt,
                    "candidates_tokens": note_candidates,
                    "thoughts_tokens": note_thoughts,
                    "total_tokens": note_total,
                    "per_agent": agent_tokens
                },
                "raw_final_result": final_dict,
                "error": None
            }
        except Exception as e:
            error_str = str(e)
            rotate_api_key()
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                wait_time = (attempt + 1) * 3
                print(f"\n[API Throttle: {error_str[:150]}] Switched to next key. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"\n[Unexpected Error: {error_str[:150]}] (Attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    return {
                        "predicted_icd": "ERROR",
                        "predicted_disease": "ERROR",
                        "status": "error",
                        "api_latency_seconds": 0.0,
                        "tokens": {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0},
                        "raw_final_result": {},
                        "error": error_str
                    }
                await asyncio.sleep(2)

    return {
        "predicted_icd": "ERROR",
        "predicted_disease": "ERROR",
        "status": "error",
        "api_latency_seconds": 0.0,
        "tokens": {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0},
        "raw_final_result": {},
        "error": "Max retries exceeded"
    }

def calculate_comprehensive_metrics(y_true: List[str], y_pred: List[str], latencies: List[float], token_records: List[Dict[str, int]]) -> Dict[str, Any]:
    """
    Computes rigorous ground-truth evaluation metrics:
    Accuracy, Macro-Precision, Macro-Recall, Macro-F1, per-class metrics with support,
    confusion matrix, and pure API latency statistics (mean +- std).
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_score_macro": 0.0}

    # Accuracy
    correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt.strip().upper() == pred.strip().upper())
    accuracy = correct / total

    # Per-class support & metrics
    all_classes = sorted(list(set(y_true + y_pred)))
    confusion_matrix = {gt_c: {pred_c: 0 for pred_c in all_classes} for gt_c in all_classes}
    for gt, pred in zip(y_true, y_pred):
        gt_key = gt.strip()
        pred_key = pred.strip()
        if gt_key in confusion_matrix and pred_key in confusion_matrix[gt_key]:
            confusion_matrix[gt_key][pred_key] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for c in all_classes:
        c_upper = c.strip().upper()
        support = sum(1 for gt in y_true if gt.strip().upper() == c_upper)
        tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt.strip().upper() == c_upper and pred.strip().upper() == c_upper)
        fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt.strip().upper() != c_upper and pred.strip().upper() == c_upper)
        fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt.strip().upper() == c_upper and pred.strip().upper() != c_upper)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if support > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        per_class[c] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4)
        }

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Latency statistics (Pure API response time, excluding rate-limit throttle sleep)
    valid_latencies = [l for l in latencies if l > 0]
    lat_mean = statistics.mean(valid_latencies) if valid_latencies else 0.0
    lat_std = statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0.0
    lat_min = min(valid_latencies) if valid_latencies else 0.0
    lat_max = max(valid_latencies) if valid_latencies else 0.0
    lat_median = statistics.median(valid_latencies) if valid_latencies else 0.0

    # Token statistics from usage_metadata
    prompt_tokens = [t.get("prompt_tokens", 0) for t in token_records]
    cand_tokens = [t.get("candidates_tokens", 0) for t in token_records]
    thought_tokens = [t.get("thoughts_tokens", 0) for t in token_records]
    total_tokens = [t.get("total_tokens", 0) for t in token_records]

    return {
        "total_cases": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(macro_precision, 4),
        "recall_macro": round(macro_recall, 4),
        "f1_score_macro": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "latency_seconds": {
            "mean": round(lat_mean, 4),
            "std": round(lat_std, 4),
            "median": round(lat_median, 4),
            "min": round(lat_min, 4),
            "max": round(lat_max, 4),
            "unit": "seconds per clinical note (pure API execution, excluding sleep delays)"
        },
        "tokens_per_note": {
            "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else 0,
            "candidates_tokens_mean": round(statistics.mean(cand_tokens), 2) if cand_tokens else 0,
            "thoughts_tokens_mean": round(statistics.mean(thought_tokens), 2) if thought_tokens else 0,
            "total_tokens_mean": round(statistics.mean(total_tokens), 2) if total_tokens else 0,
            "total_tokens_all_notes": sum(total_tokens)
        }
    }

async def main_async(limit: int = 200, delay: float = 1.0):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/clinical_dataset.csv"))
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    total_in_file = len(df)
    if limit > 0:
        df = df.head(limit)

    print("=" * 75)
    print("MEDICAL CODING BENCHMARK: Google ADK Sequential 4-Agent System")
    print(f"Dataset Path : {dataset_path}")
    print(f"Total Notes  : {len(df)} of {total_in_file} in dataset")
    print(f"Model ID     : {MODEL_NAME}")
    print("=" * 75)

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="medical_coding_eval", session_service=session_service)

    y_true = []
    y_pred = []
    latencies = []
    token_records = []
    results_log = []

    wall_start = time.time()

    for idx, row in df.iterrows():
        rotate_api_key()
        case_id = row["id"]
        note = row["clinical_note"]
        gt_disease = row["ground_truth_disease"]
        gt_icd = str(row["ground_truth_icd"]).strip()

        print(f"\n[{idx + 1}/{len(df)}] Case ID {case_id} | GT: {gt_disease} ({gt_icd})")

        res = await run_single_case(runner, note)

        pred_icd = res["predicted_icd"].strip()
        pred_disease = res["predicted_disease"]
        is_match = bool(gt_icd.upper() == pred_icd.upper())
        call_lat = res["api_latency_seconds"]

        print(f"      Pred: {pred_disease} ({pred_icd}) -> {'[MATCH]' if is_match else '[MISMATCH]'}")
        print(f"      Pure API Latency: {call_lat:.2f}s | Tokens: {res['tokens']['total_tokens']}")

        y_true.append(str(gt_icd))
        y_pred.append(str(pred_icd))
        latencies.append(call_lat)
        token_records.append(res["tokens"])

        results_log.append({
            "id": int(case_id),
            "ground_truth_disease": str(gt_disease),
            "ground_truth_icd": str(gt_icd),
            "predicted_disease": str(pred_disease),
            "predicted_icd": str(pred_icd),
            "match": bool(is_match),
            "status": str(res["status"]),
            "api_latency_seconds": call_lat,
            "tokens": res["tokens"],
            "error": str(res["error"]) if res["error"] else None
        })

        # Checkpoint save after every case so no progress is ever lost
        try:
            interim_metrics = calculate_comprehensive_metrics(y_true, y_pred, latencies, token_records)
            report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.json"))
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": MODEL_NAME,
                    "framework": "Google ADK v2.8.0 (4-Agent Sequential)",
                    "dataset": "data/clinical_dataset.csv (Synthetic Clinical Vignettes)",
                    "in_progress": bool(idx < len(df) - 1),
                    "cases_completed": idx + 1,
                    "total_cases_planned": len(df),
                    "metrics": interim_metrics,
                    "detailed_results": results_log
                }, f, indent=2, default=str)
        except Exception as ckpt_err:
            pass

        if delay > 0 and idx < len(df) - 1:
            await asyncio.sleep(delay)

    wall_elapsed = time.time() - wall_start
    metrics = calculate_comprehensive_metrics(y_true, y_pred, latencies, token_records)

    print("\n" + "=" * 75)
    print("EMPIRICAL BENCHMARK METRICS (4-Agent ADK Sequential Pipeline)")
    print("=" * 75)
    print(f"Total Evaluated Notes  : {metrics['total_cases']}")
    print(f"Correct Predictions    : {metrics['correct_predictions']}")
    print(f"Incorrect Predictions  : {metrics['incorrect_predictions']}")
    print(f"Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision        : {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall           : {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score         : {metrics['f1_score_macro'] * 100:.2f}%")
    print(f"Mean API Latency/Note  : {metrics['latency_seconds']['mean']:.2f}s +/- {metrics['latency_seconds']['std']:.2f}s (Median: {metrics['latency_seconds']['median']:.2f}s)")
    print(f"Mean Tokens/Note       : {metrics['tokens_per_note']['total_tokens_mean']:.0f} (Prompt: {metrics['tokens_per_note']['prompt_tokens_mean']:.0f}, Candidates: {metrics['tokens_per_note']['candidates_tokens_mean']:.0f})")
    print(f"Wall Clock Run Time    : {wall_elapsed:.2f} seconds")
    print("=" * 75)

    # Save detailed evaluation report
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_results.json"))
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME,
                "framework": "Google ADK v2.8.0 (4-Agent Sequential)",
                "dataset": "data/clinical_dataset.csv (Synthetic Clinical Vignettes)",
                "wall_elapsed_seconds": round(wall_elapsed, 2),
                "metrics": metrics,
                "detailed_results": results_log
            }, f, indent=2, default=str)
        print(f"\nEmpirical benchmark report saved to {report_path}")
    except Exception as save_err:
        print(f"\nError saving evaluation report: {save_err}")

def main():
    parser = argparse.ArgumentParser(description="Medical Coding Benchmark Evaluation Script")
    parser.add_argument("--limit", type=int, default=200, help="Number of cases to evaluate (default: 200)")
    parser.add_argument("--delay", type=float, default=1.0, help="Throttle delay in seconds between cases")
    args = parser.parse_args()

    asyncio.run(main_async(limit=args.limit, delay=args.delay))

if __name__ == "__main__":
    main()
