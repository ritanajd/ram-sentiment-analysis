#!/usr/bin/env python3
"""
Baseline models: TF-IDF + Logistic Regression and Linear SVM.

Reads:  outputs/merged_labeled_3class.csv
Writes: outputs/tables/baseline_report_*.csv
        outputs/tables/baseline_comparison.csv
        outputs/figures/baseline_cm_*.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

REPO_ROOT   = Path(__file__).parent.parent
OUTPUT_DIR  = REPO_ROOT / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR  = OUTPUT_DIR / 'tables'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_baselines(df, random_seed=42):
    """Train and evaluate TF-IDF + LR and SVM baselines."""
    print("\n[Baselines — TF-IDF + Traditional ML]")

    X = df['review_text'].values
    y = df['sentiment_3'].values

    print(f"Total samples: {len(X)}")
    print(f"Sentiment distribution: {pd.Series(y).value_counts().to_dict()}")

    # 70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp)

    print(f"\nTrain set:      {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set:       {len(X_test)} samples")

    # TF-IDF vectorisation
    print("\nVectorizing text with TF-IDF (max_features=5000, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            min_df=2, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=random_seed),
        'Linear SVM': LinearSVC(
            class_weight='balanced', random_state=random_seed, max_iter=2000),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc         = accuracy_score(y_test, y_pred)
        macro_f1    = f1_score(y_test, y_pred, average='macro',    zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")

        results.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Macro F1': round(macro_f1, 4),
            'Weighted F1': round(weighted_f1, 4),
        })

        # Per-class report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        slug = model_name.lower().replace(' ', '_')
        report_df.to_csv(TABLES_DIR / f'baseline_report_{slug}.csv')

        # Confusion matrix figure
        cm = confusion_matrix(y_test, y_pred,
                              labels=['Negative', 'Neutral', 'Positive'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'baseline_cm_{slug}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(TABLES_DIR / 'baseline_comparison.csv', index=False)
    try:
        comparison_df.to_markdown(TABLES_DIR / 'baseline_comparison.md', index=False)
    except ImportError:
        pass  # tabulate not installed — skip markdown export

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    return comparison_df, results


if __name__ == "__main__":
    labeled_path = OUTPUT_DIR / 'merged_labeled_3class.csv'
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
        run_baselines(df)
    else:
        print("Run 02_labeling.py first.")
