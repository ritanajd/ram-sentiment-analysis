#!/usr/bin/env python3
"""
Sentiment labeling script.

Reads:  outputs/merged_clean_complete.csv
Writes: outputs/merged_labeled_3class.csv
        outputs/merged_labeled_binary.csv

Labeling rules:
    3-class: Negative (rating <= 2), Neutral (rating == 3), Positive (rating >= 4)
    Binary : Negative (rating <= 2), Positive (rating >= 4)  — Neutral rows dropped
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_labeling(df):
    """
    Applies 3-class and binary labeling to the cleaned dataset.
    Returns (df_3class, df_binary).
    """
    if df.empty:
        print("ERROR: Empty dataframe!")
        return df, pd.DataFrame()

    print("\n[Labeling]")
    print(f"Input dataset: {len(df)} rows")

    # Ensure rating_5 is numeric
    df = df.copy()
    df['rating_5'] = pd.to_numeric(df['rating_5'], errors='coerce')

    print(f"\nRating distribution (before labeling):")
    print(df['rating_5'].value_counts().sort_index())

    # 3-class labeling
    def label_3_class(rating):
        if pd.isna(rating):
            return None
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:  # >= 4
            return 'Positive'

    df['sentiment_3'] = df['rating_5'].apply(label_3_class)

    print(f"\n3-Class Sentiment Distribution:")
    print(df['sentiment_3'].value_counts())

    # Binary labeling (drop neutral)
    df_binary = df[df['rating_5'] != 3].copy()

    def label_binary(rating):
        if pd.isna(rating):
            return None
        if rating <= 2:
            return 'Negative'
        else:  # >= 4
            return 'Positive'

    df_binary['sentiment_binary'] = df_binary['rating_5'].apply(label_binary)

    print(f"\nBinary Sentiment Distribution (after removing neutral):")
    print(df_binary['sentiment_binary'].value_counts())

    # Save labeled datasets
    out_3class = OUTPUT_DIR / 'merged_labeled_3class.csv'
    out_binary = OUTPUT_DIR / 'merged_labeled_binary.csv'

    df.to_csv(out_3class, index=False)
    df_binary.to_csv(out_binary, index=False)

    print(f"\nSaved: {out_3class} ({len(df)} rows)")
    print(f"Saved: {out_binary} ({len(df_binary)} rows)")

    return df, df_binary


if __name__ == "__main__":
    clean_path = OUTPUT_DIR / 'merged_clean_complete.csv'
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        apply_labeling(df)
    else:
        print("Run 01_data_ingestion.py first.")
