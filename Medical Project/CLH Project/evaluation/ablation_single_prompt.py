import os
import sys
import json
import time
import argparse
import statistics
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv

# Ensure line-buffered stdout for real-time logging
sys.stdout.reconfigure(line_buffering=True)

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.config import MODEL_NAME, rotate_api_key
from google import genai

ICD_CODES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/icd_codes.json"))
GUIDELINES_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/coding_guidelines.json"))

with open(ICD_CODES_FILE, "r", encoding="utf-8") as f:
    icd_table = json.load(f)

with open(GUIDELINES_FILE, "r", encoding="utf-8") as f:
    guidelines_data = json.load(f)

SINGLE_PROMPT_SYSTEM_TEMPLATE = f"""You are an expert Certified Clinical Medical Coder.
Your task is to review the following unstructured clinical note, consult the provided official ICD lookup table and official clinical coding guidelines, and determine the single most accurate ICD-10 diagnostic code and final diagnosis.

Official ICD-10 Lookup Table:
{json.dumps(icd_table, indent=2)}

Official Clinical Coding Guidelines & Differential Rules:
{json.dumps(guidelines_data, indent=2)}

Instructions:
1. Extract relevant symptoms, diagnoses, history, treatments, and findings from the note.
2. Match the primary condition to the ICD-10 Lookup Table.
3. Validate your candidate code against the Official Coding Guidelines (e.g., ensure Asthma is not assigned COPD J44 codes, check Type 1 vs Type 2 diabetes rules, etc.).
4. Return ONLY a valid JSON object matching this schema:
{{
  "final_diagnosis": "Name of primary condition matching the lookup table",
  "final_icd_code": "Exact ICD code from the lookup table",
  "explanation": "Concise 1-2 sentence clinical rationale explaining the choice and guideline compliance."
}}

Do NOT include markdown formatting or backticks around the JSON. Output raw JSON only.
"""

def run_single_prompt_case(clinical_note: str, max_retries: int = 5) -> Dict[str, Any]:
    prompt = f"{SINGLE_PROMPT_SYSTEM_TEMPLATE}\n\nClinical Note:\n{clinical_note}"
    
    for attempt in range(max_retries):
        try:
            curr_key = rotate_api_key()
            client = genai.Client(api_key=curr_key)
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            api_latency = time.perf_counter() - t0

            raw_text = response.text.strip() if response.text else ""
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            try:
                parsed = json.loads(raw_text)
            except Exception:
                parsed = {}

            pred_icd = parsed.get("final_icd_code", "UNKNOWN")
            pred_disease = parsed.get("final_diagnosis", "UNKNOWN")

            u = response.usage_metadata
            tokens = {
                "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
                "candidates_tokens": getattr(u, "candidates_token_count", 0) or 0,
                "thoughts_tokens": getattr(u, "thoughts_token_count", 0) or 0,
                "total_tokens": getattr(u, "total_token_count", 0) or 0
            }

            return {
                "predicted_icd": pred_icd,
                "predicted_disease": pred_disease,
                "api_latency_seconds": round(api_latency, 4),
                "tokens": tokens,
                "raw_response": parsed,
                "error": None
            }
        except Exception as e:
            error_str = str(e)
            rotate_api_key()
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str:
                wait_time = (attempt + 1) * 3
                print(f"\n[Ablation Throttle / Key Rotated] Switched to next key. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    return {
                        "predicted_icd": "ERROR",
                        "predicted_disease": "ERROR",
                        "api_latency_seconds": 0.0,
                        "tokens": {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0},
                        "raw_response": {},
                        "error": error_str
                    }
                time.sleep(2)

    return {
        "predicted_icd": "ERROR",
        "predicted_disease": "ERROR",
        "api_latency_seconds": 0.0,
        "tokens": {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0},
        "raw_response": {},
        "error": "Max retries exceeded"
    }

def calculate_comprehensive_metrics(y_true: List[str], y_pred: List[str], latencies: List[float], token_records: List[Dict[str, int]]) -> Dict[str, Any]:
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_score_macro": 0.0}

    correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt.strip().upper() == pred.strip().upper())
    accuracy = correct / total

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

    valid_latencies = [l for l in latencies if l > 0]
    lat_mean = statistics.mean(valid_latencies) if valid_latencies else 0.0
    lat_std = statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0.0
    lat_median = statistics.median(valid_latencies) if valid_latencies else 0.0

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
            "unit": "seconds per clinical note (pure API execution)"
        },
        "tokens_per_note": {
            "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else 0,
            "candidates_tokens_mean": round(statistics.mean(cand_tokens), 2) if cand_tokens else 0,
            "thoughts_tokens_mean": round(statistics.mean(thought_tokens), 2) if thought_tokens else 0,
            "total_tokens_mean": round(statistics.mean(total_tokens), 2) if total_tokens else 0,
            "total_tokens_all_notes": sum(total_tokens)
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Single-Prompt Ablation Baseline Evaluation Script")
    parser.add_argument("--limit", type=int, default=200, help="Number of cases to evaluate (default: 200)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between cases")
    args = parser.parse_args()

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/clinical_dataset.csv"))
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    total_in_file = len(df)
    if args.limit > 0:
        df = df.head(args.limit)

    print("=" * 75)
    print("SINGLE-PROMPT ABLATION BASELINE: Zero-Shot Prompt with Full Knowledge Base")
    print(f"Dataset Path : {dataset_path}")
    print(f"Total Notes  : {len(df)} of {total_in_file} in dataset")
    print(f"Model ID     : {MODEL_NAME}")
    print("=" * 75)

    client = genai.Client(api_key=api_key)

    y_true = []
    y_pred = []
    latencies = []
    token_records = []
    results_log = []

    wall_start = time.time()

    for idx, row in df.iterrows():
        case_id = row["id"]
        note = row["clinical_note"]
        gt_disease = row["ground_truth_disease"]
        gt_icd = str(row["ground_truth_icd"]).strip()

        print(f"\n[Ablation {idx + 1}/{len(df)}] Case ID {case_id} | GT: {gt_disease} ({gt_icd})")

        res = run_single_prompt_case(note)

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
            "api_latency_seconds": call_lat,
            "tokens": res["tokens"],
            "error": str(res["error"]) if res["error"] else None
        })

        # Checkpoint save after every case so no progress is ever lost
        try:
            interim_metrics = calculate_comprehensive_metrics(y_true, y_pred, latencies, token_records)
            report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ablation_results.json"))
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": MODEL_NAME,
                    "framework": "Zero-Shot Single-Prompt (Full KB Access)",
                    "dataset": "data/clinical_dataset.csv (Synthetic Clinical Vignettes)",
                    "in_progress": bool(idx < len(df) - 1),
                    "cases_completed": idx + 1,
                    "total_cases_planned": len(df),
                    "metrics": interim_metrics,
                    "detailed_results": results_log
                }, f, indent=2, default=str)
        except Exception:
            pass

        if args.delay > 0 and idx < len(df) - 1:
            time.sleep(args.delay)

    wall_elapsed = time.time() - wall_start
    metrics = calculate_comprehensive_metrics(y_true, y_pred, latencies, token_records)

    print("\n" + "=" * 75)
    print("SINGLE-PROMPT ABLATION BASELINE METRICS")
    print("=" * 75)
    print(f"Total Evaluated Notes  : {metrics['total_cases']}")
    print(f"Correct Predictions    : {metrics['correct_predictions']}")
    print(f"Incorrect Predictions  : {metrics['incorrect_predictions']}")
    print(f"Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision        : {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall           : {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score         : {metrics['f1_score_macro'] * 100:.2f}%")
    print(f"Mean API Latency/Note  : {metrics['latency_seconds']['mean']:.2f}s +/- {metrics['latency_seconds']['std']:.2f}s (Median: {metrics['latency_seconds']['median']:.2f}s)")
    print(f"Mean Tokens/Note       : {metrics['tokens_per_note']['total_tokens_mean']:.0f}")
    print(f"Wall Clock Run Time    : {wall_elapsed:.2f} seconds")
    print("=" * 75)

    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ablation_results.json"))
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME,
                "framework": "Zero-Shot Single-Prompt (Full KB Access)",
                "dataset": "data/clinical_dataset.csv (Synthetic Clinical Vignettes)",
                "wall_elapsed_seconds": round(wall_elapsed, 2),
                "metrics": metrics,
                "detailed_results": results_log
            }, f, indent=2, default=str)
        print(f"\nAblation evaluation report saved to {report_path}")
    except Exception as save_err:
        print(f"\nError saving ablation report: {save_err}")

if __name__ == "__main__":
    main()
