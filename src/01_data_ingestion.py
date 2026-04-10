#!/usr/bin/env python3
"""
Complete data ingestion script handling all four data sources with proper encoding.

Data sources:
    data/raw/ram_internal.csv            — RAM internal feedback (pipe-delimited)
    data/raw/trustpilot_reviews.csv      — Trustpilot reviews (French)
    data/raw/ram_skytrax_reviews.csv     — Skytrax reviews
    data/raw/tripadvisor_reviews_combined.csv — TripAdvisor reviews

Outputs (written to outputs/):
    merged_raw_complete.csv
    merged_clean_complete.csv
    ingestion_statistics_complete.json
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# Set up paths relative to this script's location
REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / 'data' / 'raw'
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ram_internal():
    """Load RAM internal feedback data (pipe-delimited)"""
    print("📖 Loading RAM Internal data...")
    df = pd.read_csv(DATA_DIR / 'ram_internal.csv', sep='|')
    print(f"   ✅ Loaded {len(df)} reviews")
    print(f"   Columns: {df.columns.tolist()}")
    return df


def load_trustpilot():
    """Load Trustpilot reviews (French, CSV)"""
    print("📖 Loading Trustpilot data...")
    df = pd.read_csv(DATA_DIR / 'trustpilot_reviews.csv', encoding='utf-8')
    print(f"   ✅ Loaded {len(df)} reviews")
    print(f"   Columns: {df.columns.tolist()}")
    return df


def load_skytrax():
    """Load Skytrax reviews"""
    print("📖 Loading Skytrax data...")
    df = pd.read_csv(DATA_DIR / 'ram_skytrax_reviews.csv')
    print(f"   ✅ Loaded {len(df)} reviews")
    print(f"   Columns: {df.columns.tolist()}")
    return df


def load_tripadvisor():
    """Load TripAdvisor reviews"""
    print("📖 Loading TripAdvisor data...")
    df = pd.read_csv(DATA_DIR / 'tripadvisor_reviews_combined.csv')
    print(f"   ✅ Loaded {len(df)} reviews")
    print(f"   Columns: {df.columns.tolist()}")
    return df


def harmonize_schemas(ram_df, trustpilot_df, skytrax_df, tripadvisor_df):
    """Harmonize all dataframes to a common schema"""
    print("\n🔄 Harmonizing schemas...")

    frames = []

    # RAM Internal (pipe-delimited format)
    print("   Processing RAM Internal...")
    ram_clean = pd.DataFrame()
    ram_clean['review_text'] = ram_df['Comments'].fillna('')
    ram_clean['rating_5'] = ram_df['Rating of 5'].fillna(0)
    ram_clean['platform'] = 'RAM Internal'
    frames.append(ram_clean)
    print(f"      ✅ {len(ram_clean)} reviews")

    # Trustpilot (French)
    print("   Processing Trustpilot...")
    trustpilot_clean = pd.DataFrame()
    trustpilot_clean['review_text'] = trustpilot_df['texte'].fillna('')
    trustpilot_clean['rating_5'] = trustpilot_df['note'].fillna(0)
    trustpilot_clean['platform'] = 'Trustpilot'
    frames.append(trustpilot_clean)
    print(f"      ✅ {len(trustpilot_clean)} reviews")

    # Skytrax
    print("   Processing Skytrax...")
    skytrax_clean = pd.DataFrame()
    skytrax_clean['review_text'] = skytrax_df['texte'].fillna('')
    # Normalize Skytrax 1-10 scale to 1-5
    skytrax_clean['rating_5'] = (skytrax_df['note_globale_10'].fillna(0) / 2).round(1)
    skytrax_clean['platform'] = 'Skytrax'
    frames.append(skytrax_clean)
    print(f"      ✅ {len(skytrax_clean)} reviews")

    # TripAdvisor
    print("   Processing TripAdvisor...")
    tripadvisor_clean = pd.DataFrame()
    tripadvisor_clean['review_text'] = tripadvisor_df['Comments'].fillna('')
    tripadvisor_clean['rating_5'] = tripadvisor_df['Rating of 5'].fillna(0)
    tripadvisor_clean['platform'] = 'TripAdvisor'
    frames.append(tripadvisor_clean)
    print(f"      ✅ {len(tripadvisor_clean)} reviews")

    # Merge all
    merged = pd.concat(frames, ignore_index=True)
    print(f"\n   ✅ Total merged: {len(merged)} reviews")

    return merged


def clean_data(df):
    """Clean and deduplicate data"""
    print("\n🧹 Cleaning data...")

    initial_count = len(df)

    # Remove missing values
    df_clean = df.dropna(subset=['review_text', 'rating_5']).copy()
    print(f"   After removing NaN: {len(df_clean)} reviews (removed {initial_count - len(df_clean)})")

    # Remove empty reviews
    df_clean = df_clean[df_clean['review_text'].astype(str).str.strip().str.len() > 0].copy()
    print(f"   After removing empty: {len(df_clean)} reviews")

    # Normalize text for deduplication
    df_clean['text_normalized'] = df_clean['review_text'].astype(str).str.lower().str.strip()

    # Remove duplicates
    pre_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['text_normalized'], keep='first')
    duplicates_removed = pre_dedup - len(df_clean)
    print(f"   Duplicates removed: {duplicates_removed}")
    print(f"   Final clean dataset: {len(df_clean)} reviews")
    print(f"   Retention rate: {(len(df_clean) / initial_count * 100):.1f}%")

    # Drop temporary column
    df_clean = df_clean.drop('text_normalized', axis=1)

    return df_clean


def analyze_distribution(df):
    """Analyze dataset distribution"""
    print("\n📊 Dataset Distribution:")
    print(f"\n   Platform Distribution:")
    for platform, count in df['platform'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"      {platform}: {count} ({pct:.1f}%)")

    print(f"\n   Rating Distribution:")
    for rating in sorted(df['rating_5'].unique()):
        count = (df['rating_5'] == rating).sum()
        pct = (count / len(df)) * 100
        print(f"      {rating} stars: {count} ({pct:.1f}%)")

    print(f"\n   Statistics:")
    print(f"      Mean rating: {df['rating_5'].mean():.2f}")
    print(f"      Median rating: {df['rating_5'].median():.2f}")
    print(f"      Std dev: {df['rating_5'].std():.2f}")
    print(f"      Min: {df['rating_5'].min()}")
    print(f"      Max: {df['rating_5'].max()}")


def main():
    """Main execution"""
    print("=" * 60)
    print("COMPLETE DATA INGESTION PIPELINE (ALL SOURCES)")
    print("=" * 60)

    # Load all data sources
    ram_df = load_ram_internal()
    trustpilot_df = load_trustpilot()
    skytrax_df = load_skytrax()
    tripadvisor_df = load_tripadvisor()

    # Harmonize schemas
    merged_df = harmonize_schemas(ram_df, trustpilot_df, skytrax_df, tripadvisor_df)

    # Save raw merged
    raw_path = OUTPUT_DIR / 'merged_raw_complete.csv'
    merged_df.to_csv(raw_path, index=False)
    print(f"\n💾 Raw merged data saved to {raw_path}")

    # Clean data
    clean_df = clean_data(merged_df)

    # Analyze distribution
    analyze_distribution(clean_df)

    # Save clean dataset — used by downstream scripts as 'outputs/merged_clean_complete.csv'
    clean_path = OUTPUT_DIR / 'merged_clean_complete.csv'
    clean_df.to_csv(clean_path, index=False)
    print(f"\n💾 Clean dataset saved to {clean_path}")

    # Save statistics
    stats = {
        'total_reviews_raw': len(merged_df),
        'total_reviews_clean': len(clean_df),
        'duplicates_removed': len(merged_df) - len(clean_df),
        'retention_rate': float((len(clean_df) / len(merged_df)) * 100),
        'platform_distribution': clean_df['platform'].value_counts().to_dict(),
        'mean_rating': float(clean_df['rating_5'].mean()),
        'median_rating': float(clean_df['rating_5'].median()),
        'std_dev': float(clean_df['rating_5'].std()),
    }

    stats_path = OUTPUT_DIR / 'ingestion_statistics_complete.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Statistics saved to {stats_path}")

    print("\n" + "=" * 60)
    print("✅ DATA INGESTION COMPLETE")
    print("=" * 60)

    return clean_df


if __name__ == '__main__':
    df = main()
