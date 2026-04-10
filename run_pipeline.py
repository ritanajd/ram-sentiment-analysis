"""
Royal Air Maroc — Multilingual Sentiment Analysis Pipeline
==========================================================
Master script: runs the full pipeline end-to-end.

Usage:
    python run_pipeline.py

Steps executed:
    1. Data ingestion and harmonisation
    2. Sentiment labelling
    3. Exploratory Data Analysis (EDA) + figures
    4. Baseline models (Logistic Regression, Linear SVM)
    5. DistilBERT fine-tuning and evaluation

All outputs are written to the `outputs/` directory.
"""

import subprocess
import sys
import os

STEPS = [
    ("Step 1 — Data Ingestion",      "src/01_data_ingestion.py"),
    ("Step 2 — Labelling",           "src/02_labeling.py"),
    ("Step 3 — EDA",                 "src/03_eda.py"),
    ("Step 4 — Baseline Models",     "src/04_baselines.py"),
    ("Step 5 — DistilBERT Training", "src/05_distilbert_train.py"),
]

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"  ✅ {name} completed.")

if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    print("\n🚀 Royal Air Maroc Sentiment Analysis Pipeline")
    print("   Starting full pipeline run...\n")

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE")
    print("  All outputs saved to outputs/")
    print("="*60 + "\n")
