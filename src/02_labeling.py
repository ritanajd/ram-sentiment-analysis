import pandas as pd
import os
import numpy as np

def apply_labeling(df):
    """
    Applies 3-class and binary labeling to the cleaned dataset.
    3-class: Negative (<=2), Neutral (3), Positive (>=4)
    Binary: Negative (<=2), Positive (>=4), drop Neutral
    """
    if df.empty:
        print("ERROR: Empty dataframe!")
        return df, pd.DataFrame()
    
    print("\n[Labeling]")
    print(f"Input dataset: {len(df)} rows")
    
    # Ensure rating_5 is numeric
    df = df.copy()
    df['rating_5'] = pd.to_numeric(df['rating_5'], errors='coerce')
    
    # Check rating distribution
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
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    df.to_csv('outputs/merged_labeled_3class.csv', index=False)
    df_binary.to_csv('outputs/merged_labeled_binary.csv', index=False)
    
    print(f"\nSaved: outputs/merged_labeled_3class.csv ({len(df)} rows)")
    print(f"Saved: outputs/merged_labeled_binary.csv ({len(df_binary)} rows)")
    
    return df, df_binary

if __name__ == "__main__":
    if os.path.exists('outputs/merged_clean.csv'):
        df = pd.read_csv('outputs/merged_clean.csv')
        apply_labeling(df)
    else:
        print("Run data_ingestion.py first.")
