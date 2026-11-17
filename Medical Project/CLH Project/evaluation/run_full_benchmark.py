import sys
import subprocess
import os

# Line-buffered output
sys.stdout.reconfigure(line_buffering=True)

python_bin = sys.executable

print("===========================================================================")
print("STARTING FULL REPRODUCIBLE BENCHMARK RUN (200 Notes)")
print("1. 4-Agent Google ADK Pipeline (evaluate.py)")
print("2. Single-Prompt Full-KB Baseline (ablation_single_prompt.py)")
print("===========================================================================")

print("\n>>> STEP 1: Running 4-Agent Pipeline on all 200 notes...")
res1 = subprocess.run([python_bin, "evaluation/evaluate.py", "--limit", "200", "--delay", "1.0"], cwd=os.getcwd())
if res1.returncode != 0:
    print(f"Warning: evaluate.py finished with returncode {res1.returncode}")

print("\n>>> STEP 2: Running Single-Prompt Ablation on all 200 notes...")
res2 = subprocess.run([python_bin, "evaluation/ablation_single_prompt.py", "--limit", "200", "--delay", "1.0"], cwd=os.getcwd())
if res2.returncode != 0:
    print(f"Warning: ablation_single_prompt.py finished with returncode {res2.returncode}")

print("\n===========================================================================")
print("BENCHMARK EXECUTION COMPLETE! Both evaluation_results.json and ablation_results.json updated.")
print("===========================================================================")
