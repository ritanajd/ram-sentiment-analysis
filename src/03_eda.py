#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) script.

Reads:  outputs/merged_labeled_3class.csv
        outputs/merged_labeled_binary.csv
Writes: outputs/figures/01_*.png … 10_*.png
        outputs/tables/
        outputs/core_statistics.json
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import numpy as np
from collections import Counter
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
OUTPUT_DIR  = REPO_ROOT / 'outputs'
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR  = OUTPUT_DIR / 'tables'

for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Language detection ────────────────────────────────────────────────────────
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed — using heuristic language detection.")


def detect_language_heuristic(text):
    text_lower = str(text).lower()
    french_kw  = ['le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'est',
                  'que', 'pour', 'avec', 'pas', 'très', 'bien']
    english_kw = ['the', 'and', 'was', 'were', 'for', 'with', 'not', 'very',
                  'good', 'bad', 'flight', 'service']
    arabic_kw  = ['في', 'من', 'إلى', 'هذا', 'كان', 'لا', 'على']
    fc = sum(1 for kw in french_kw  if kw in text_lower)
    ec = sum(1 for kw in english_kw if kw in text_lower)
    ac = sum(1 for kw in arabic_kw  if kw in text_lower)
    if fc > ec and fc > ac:
        return 'French'
    elif ec > fc and ec > ac:
        return 'English'
    elif ac > 0:
        return 'Arabic/Darija'
    return 'Other'


def detect_language(text):
    if HAS_LANGDETECT:
        try:
            code = detect(str(text))
            return {'fr': 'French', 'en': 'English', 'ar': 'Arabic/Darija'}.get(code, 'Other')
        except Exception:
            pass
    return detect_language_heuristic(text)


# ── EDA figures ───────────────────────────────────────────────────────────────
def generate_eda_figures(df, df_binary):
    """Generate all EDA figures and save statistics."""
    print("\n[EDA - Exploratory Data Analysis]")
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    fig_dir = str(FIGURES_DIR)

    # 1. Platform composition
    print("  Figure 1: Platform composition...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    pc = df['platform'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    ax1.pie(pc.values, labels=pc.index, autopct='%1.1f%%',
            colors=colors[:len(pc)], startangle=90)
    ax1.set_title('Dataset Composition by Platform', fontsize=14, fontweight='bold')
    pc.plot(kind='bar', ax=ax2, color=colors[:len(pc)])
    ax2.set_title('Review Count by Platform', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Reviews')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/01_platform_composition.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Global rating distribution
    print("  Figure 2: Rating distribution...")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['rating_5'], bins=15, kde=True, color='steelblue')
    plt.title('Global Rating Distribution (1–5 Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Rating'); plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/02_global_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Sentiment by platform
    print("  Figure 3: Sentiment by platform...")
    plt.figure(figsize=(12, 6))
    pd.crosstab(df['platform'], df['sentiment_3']).plot(
        kind='bar', color=['#FF6B6B', '#FFD93D', '#6BCB77'])
    plt.title('Sentiment Distribution by Platform', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews'); plt.xlabel('Platform')
    plt.legend(title='Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/03_sentiment_by_platform.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 3-class imbalance
    print("  Figure 4: 3-class sentiment distribution...")
    plt.figure(figsize=(10, 6))
    sc = df['sentiment_3'].value_counts()
    c_map = {'Negative': '#FF6B6B', 'Neutral': '#FFD93D', 'Positive': '#6BCB77'}
    sc.plot(kind='bar', color=[c_map.get(s, '#999') for s in sc.index])
    plt.title('3-Class Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews'); plt.xlabel('Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    for i, v in enumerate(sc.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/04_sentiment_3class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Binary imbalance
    print("  Figure 5: Binary sentiment distribution...")
    plt.figure(figsize=(10, 6))
    bc = df_binary['sentiment_binary'].value_counts()
    b_map = {'Negative': '#FF6B6B', 'Positive': '#6BCB77'}
    bc.plot(kind='bar', color=[b_map.get(s, '#999') for s in bc.index])
    plt.title('Binary Sentiment Distribution (Neutral Removed)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews'); plt.xlabel('Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    for i, v in enumerate(bc.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/05_sentiment_binary_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Review length histogram
    print("  Figure 6: Review length histogram...")
    df['review_len'] = df['review_text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(12, 6))
    plt.hist(df['review_len'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(df['review_len'].mean(),   color='red',    linestyle='--', linewidth=2,
                label=f'Mean: {df["review_len"].mean():.1f} words')
    plt.axvline(df['review_len'].median(), color='green',  linestyle='-',  linewidth=2,
                label=f'Median: {df["review_len"].median():.1f} words')
    plt.axvline(512, color='orange', linestyle=':', linewidth=2,
                label='BERT 512-token limit (~2048 words)')
    plt.title('Review Length Distribution (Word Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words'); plt.ylabel('Frequency'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/06_review_length_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Top keywords overall
    print("  Figure 7: Top keywords overall...")
    stopwords = {
        'the', 'and', 'to', 'a', 'in', 'is', 'it', 'of', 'for', 'with', 'on',
        'was', 'were', 'at', 'by', 'an', 'be', 'as', 'this', 'that', 'ram',
        'royal', 'air', 'maroc', 'flight', 'airline', 'i', 'my', 'me', 'you',
        'your', 'we', 'our', 'they', 'their', 'have', 'has', 'had', 'le', 'la',
        'les', 'de', 'des', 'un', 'une', 'et', 'est', 'que', 'pour', 'avec',
        'pas', 'or', 'if', 'but', 'not', 'so', 'than', 'can', 'will', 'would',
        'could', 'should', 'from', 'into', 'through', 'before', 'after', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'only', 'own', 'same', 'too', 'very', 'just', 'now', 'do',
        'does', 'did', 'doing'
    }

    def get_top_words(texts, n=20):
        words = []
        for t in texts:
            words.extend(re.findall(r'\b\w+\b', str(t).lower()))
        words = [w for w in words if w not in stopwords and len(w) > 3]
        return Counter(words).most_common(n)

    tw = get_top_words(df['review_text'])
    if tw:
        words, counts = zip(*tw)
        plt.figure(figsize=(12, 8))
        plt.barh(list(words), list(counts), color='steelblue')
        plt.xlabel('Frequency')
        plt.title('Top 20 Keywords Overall', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/07_top_keywords_overall.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 8. Top keywords by sentiment
    print("  Figure 8: Top keywords by sentiment...")
    for sentiment in ['Negative', 'Positive']:
        subset = df[df['sentiment_3'] == sentiment]['review_text']
        tw_s = get_top_words(subset, n=15)
        if tw_s:
            words, counts = zip(*tw_s)
            plt.figure(figsize=(12, 8))
            plt.barh(list(words), list(counts),
                     color='#FF6B6B' if sentiment == 'Negative' else '#6BCB77')
            plt.xlabel('Frequency')
            plt.title(f'Top 15 Keywords — {sentiment} Sentiment', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/08_top_keywords_{sentiment.lower()}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    # 9. Language distribution
    print("  Figure 9: Language distribution...")
    df['language'] = df['review_text'].apply(detect_language)
    lc = df['language'].value_counts()
    colors_lang = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    lc.plot(kind='bar', color=colors_lang[:len(lc)])
    plt.title('Language Distribution Across Reviews', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews'); plt.xlabel('Language')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    for i, v in enumerate(lc.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/09_language_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Language by platform
    print("  Figure 10: Language by platform...")
    pd.crosstab(df['platform'], df['language']).plot(kind='bar')
    plt.title('Language Distribution by Platform', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews'); plt.xlabel('Platform')
    plt.legend(title='Language')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/10_language_by_platform.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save core statistics
    print("  Saving core statistics...")
    stats = {
        'total_reviews': len(df),
        'platform_counts': df['platform'].value_counts().to_dict(),
        'sentiment_3_counts': df['sentiment_3'].value_counts().to_dict(),
        'sentiment_binary_counts': df_binary['sentiment_binary'].value_counts().to_dict(),
        'avg_rating': float(df['rating_5'].mean()),
        'median_rating': float(df['rating_5'].median()),
        'avg_review_len': float(df['review_len'].mean()),
        'median_review_len': float(df['review_len'].median()),
        'language_distribution': df['language'].value_counts().to_dict(),
    }

    with open(OUTPUT_DIR / 'core_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    n_figs = len(list(FIGURES_DIR.glob('*.png')))
    print(f"\nEDA complete! {n_figs} figures saved to {FIGURES_DIR}")
    print(f"Core statistics saved to {OUTPUT_DIR / 'core_statistics.json'}")
    return stats


if __name__ == "__main__":
    labeled_path = OUTPUT_DIR / 'merged_labeled_3class.csv'
    binary_path  = OUTPUT_DIR / 'merged_labeled_binary.csv'
    if labeled_path.exists() and binary_path.exists():
        df        = pd.read_csv(labeled_path)
        df_binary = pd.read_csv(binary_path)
        generate_eda_figures(df, df_binary)
    else:
        print("Run 02_labeling.py first.")
