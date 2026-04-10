import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def run_baselines(df, random_seed=42):
    """
    Trains and evaluates baseline models: TF-IDF + Logistic Regression and Linear SVM.
    """
    if not os.path.exists('tables'):
        os.makedirs('tables')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    print("\n[Baselines - TF-IDF + Traditional ML]")
    
    # Prepare data
    X = df['review_text'].values
    y = df['sentiment_3'].values
    
    print(f"Total samples: {len(X)}")
    print(f"Sentiment distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_seed),
        'Linear SVM': LinearSVC(class_weight='balanced', random_state=random_seed, max_iter=2000)
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Predictions on test set
        y_pred = model.predict(X_test_tfidf)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        
        results.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'Macro F1': round(macro_f1, 4),
            'Weighted F1': round(weighted_f1, 4)
        })
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'tables/baseline_report_{model_name.lower().replace(" ", "_")}.csv')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['Negative', 'Neutral', 'Positive'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Neutral', 'Positive'], 
                   yticklabels=['Negative', 'Neutral', 'Positive'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'figures/baseline_cm_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv('tables/baseline_comparison.csv', index=False)
    
    # Save as markdown
    comparison_df.to_markdown('tables/baseline_comparison.md', index=False)
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    return comparison_df, results

if __name__ == "__main__":
    if os.path.exists('outputs/merged_labeled_3class.csv'):
        df = pd.read_csv('outputs/merged_labeled_3class.csv')
        run_baselines(df)
    else:
        print("Run labeling.py first.")
