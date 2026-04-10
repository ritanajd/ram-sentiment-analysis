#!/usr/bin/env python3
"""
Royal Air Maroc — Multilingual Sentiment Analysis Pipeline
==========================================================
Master script: runs the full pipeline end-to-end.

Usage:
    python run_pipeline.py

Steps executed:
    1. Data ingestion and harmonisation  → outputs/merged_clean_complete.csv
    2. Sentiment labelling               → outputs/merged_labeled_3class.csv
    3. Exploratory Data Analysis (EDA)   → outputs/figures/01_*.png … 10_*.png
    4. Baseline models (LR + SVM)        → outputs/tables/, outputs/figures/
    5. DistilBERT fine-tuning            → outputs/models/, outputs/results/, outputs/figures/

All outputs are written to the `outputs/` directory.

Requirements:
    pip install -r requirements.txt
"""

import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent

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
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script)],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"  ✅ {name} completed.")


if __name__ == "__main__":
    # Pre-create output directories
    for sub in ['figures', 'results', 'models', 'tables']:
        (REPO_ROOT / 'outputs' / sub).mkdir(parents=True, exist_ok=True)

    print("\n🚀 Royal Air Maroc Sentiment Analysis Pipeline")
    print("   Starting full pipeline run...\n")

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("  All outputs saved to outputs/")
    print("=" * 60 + "\n")
