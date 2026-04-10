import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import re
import json
import numpy as np

# Try to import language detection
try:
    from langdetect import detect, detect_langs
    HAS_LANGDETECT = True
except:
    HAS_LANGDETECT = False
    print("Warning: langdetect not available, will use heuristic language detection")

def detect_language_heuristic(text):
    """Simple heuristic language detection based on keywords"""
    text_lower = str(text).lower()
    
    # French keywords
    french_keywords = ['le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'est', 'que', 'pour', 'avec', 'pas', 'très', 'bien', 'mauvais', 'excellent', 'terrible']
    french_count = sum(1 for kw in french_keywords if kw in text_lower)
    
    # English keywords
    english_keywords = ['the', 'a', 'and', 'is', 'was', 'were', 'for', 'with', 'not', 'very', 'good', 'bad', 'excellent', 'terrible', 'flight', 'service']
    english_count = sum(1 for kw in english_keywords if kw in text_lower)
    
    # Arabic/Darija keywords (basic)
    arabic_keywords = ['في', 'من', 'إلى', 'هذا', 'كان', 'لا', 'على', 'هو', 'هي']
    arabic_count = sum(1 for kw in arabic_keywords if kw in text_lower)
    
    if french_count > english_count and french_count > arabic_count:
        return 'French'
    elif english_count > french_count and english_count > arabic_count:
        return 'English'
    elif arabic_count > 0:
        return 'Arabic/Darija'
    else:
        return 'Other'

def detect_language(text):
    """Detect language of text"""
    if HAS_LANGDETECT:
        try:
            lang_code = detect(str(text))
            if lang_code == 'fr':
                return 'French'
            elif lang_code == 'en':
                return 'English'
            elif lang_code in ['ar', 'rif']:
                return 'Arabic/Darija'
            else:
                return 'Other'
        except:
            return detect_language_heuristic(text)
    else:
        return detect_language_heuristic(text)

def generate_eda_figures(df, df_binary, output_dir='figures'):
    """
    Generates thesis-ready figures for EDA.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists('tables'):
        os.makedirs('tables')
    
    print("\n[EDA - Exploratory Data Analysis]")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    
    # 1. Dataset composition by platform (pie chart)
    print("Generating platform composition charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    platform_counts = df['platform'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    ax1.pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Dataset Composition by Platform (Pie Chart)', fontsize=14, fontweight='bold')
    
    platform_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Review Count by Platform (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Reviews')
    ax2.set_xlabel('Platform')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_platform_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Global rating distribution
    print("Generating rating distribution chart...")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['rating_5'], bins=15, kde=True, color='steelblue')
    plt.title('Global Rating Distribution (1-5 Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/02_global_rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sentiment distribution by platform
    print("Generating sentiment by platform chart...")
    plt.figure(figsize=(12, 6))
    sentiment_platform = pd.crosstab(df['platform'], df['sentiment_3'])
    sentiment_platform.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#FFD93D', '#6BCB77'])
    plt.title('Sentiment Distribution by Platform', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Platform')
    plt.legend(title='Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_sentiment_by_platform.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 3-class imbalance chart
    print("Generating 3-class sentiment distribution chart...")
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_3'].value_counts()
    colors_sentiment = {'Negative': '#FF6B6B', 'Neutral': '#FFD93D', 'Positive': '#6BCB77'}
    colors_list = [colors_sentiment.get(s, '#999999') for s in sentiment_counts.index]
    
    sentiment_counts.plot(kind='bar', color=colors_list)
    plt.title('3-Class Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add count labels on bars
    for i, v in enumerate(sentiment_counts.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_sentiment_3class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Binary imbalance chart
    print("Generating binary sentiment distribution chart...")
    plt.figure(figsize=(10, 6))
    binary_counts = df_binary['sentiment_binary'].value_counts()
    colors_binary = {'Negative': '#FF6B6B', 'Positive': '#6BCB77'}
    colors_list = [colors_binary.get(s, '#999999') for s in binary_counts.index]
    
    binary_counts.plot(kind='bar', color=colors_list)
    plt.title('Binary Sentiment Distribution (Neutral Removed)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Sentiment')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add count labels on bars
    for i, v in enumerate(binary_counts.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_sentiment_binary_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Review length histogram
    print("Generating review length histogram...")
    df['review_len'] = df['review_text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    plt.hist(df['review_len'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(df['review_len'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["review_len"].mean():.1f} words')
    plt.axvline(df['review_len'].median(), color='green', linestyle='-', linewidth=2, label=f'Median: {df["review_len"].median():.1f} words')
    plt.axvline(512, color='orange', linestyle=':', linewidth=2, label='BERT 512-token limit (~2048 words)')
    plt.title('Review Length Distribution (Word Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_review_length_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Top keywords overall
    print("Generating top keywords chart...")
    def get_top_words(texts, n=20):
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b', str(text).lower()))
        
        # Filter out common stopwords
        stopwords = {
            'the', 'and', 'to', 'a', 'in', 'is', 'it', 'of', 'for', 'with', 'on', 'was', 'were', 'at', 'by', 'an', 'be', 'as', 'this', 'that',
            'ram', 'royal', 'air', 'maroc', 'flight', 'airline', 'i', 'my', 'me', 'you', 'your', 'we', 'our', 'they', 'their', 'have', 'has', 'had',
            'le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'est', 'que', 'pour', 'avec', 'pas', 'très', 'bien', 'mauvais', 'excellent', 'terrible',
            'or', 'if', 'but', 'not', 'so', 'than', 'can', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'do', 'does', 'did', 'doing'
        }
        
        words = [w for w in words if w not in stopwords and len(w) > 3]
        return Counter(words).most_common(n)
    
    top_words = get_top_words(df['review_text'])
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(12, 8))
    plt.barh(list(words), list(counts), color='steelblue')
    plt.xlabel('Frequency')
    plt.title('Top 20 Keywords Overall', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_top_keywords_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Top keywords by sentiment class
    print("Generating top keywords by sentiment...")
    for sentiment in ['Negative', 'Positive']:
        subset = df[df['sentiment_3'] == sentiment]['review_text']
        top_words_s = get_top_words(subset, n=15)
        
        if top_words_s:
            words, counts = zip(*top_words_s)
            plt.figure(figsize=(12, 8))
            plt.barh(list(words), list(counts), color='#FF6B6B' if sentiment == 'Negative' else '#6BCB77')
            plt.xlabel('Frequency')
            plt.title(f'Top 15 Keywords - {sentiment} Sentiment', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/08_top_keywords_{sentiment.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 9. Language distribution
    print("Detecting languages...")
    df['language'] = df['review_text'].apply(detect_language)
    
    plt.figure(figsize=(12, 6))
    lang_counts = df['language'].value_counts()
    colors_lang = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D']
    lang_counts.plot(kind='bar', color=colors_lang[:len(lang_counts)])
    plt.title('Language Distribution Across Reviews', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Language')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for i, v in enumerate(lang_counts.values):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_language_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Language by platform
    plt.figure(figsize=(12, 6))
    lang_platform = pd.crosstab(df['platform'], df['language'])
    lang_platform.plot(kind='bar', ax=plt.gca())
    plt.title('Language Distribution by Platform', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Reviews')
    plt.xlabel('Platform')
    plt.legend(title='Language')
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_language_by_platform.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save core statistics
    print("Saving core statistics...")
    stats = {
        'total_reviews': len(df),
        'reviews_after_cleaning': len(df),
        'platform_counts': df['platform'].value_counts().to_dict(),
        'sentiment_3_counts': df['sentiment_3'].value_counts().to_dict(),
        'sentiment_binary_counts': df_binary['sentiment_binary'].value_counts().to_dict(),
        'avg_rating': float(df['rating_5'].mean()),
        'median_rating': float(df['rating_5'].median()),
        'avg_review_len': float(df['review_len'].mean()),
        'median_review_len': float(df['review_len'].median()),
        'language_distribution': df['language'].value_counts().to_dict()
    }
    
    with open('outputs/core_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save as CSV too
    pd.Series(stats).to_csv('outputs/core_statistics.csv')
    
    print(f"\nEDA complete! Generated {len(os.listdir(output_dir))} figures in /{output_dir}")
    print(f"Core statistics saved to outputs/core_statistics.json")
    
    return stats

if __name__ == "__main__":
    if os.path.exists('outputs/merged_labeled_3class.csv'):
        df = pd.read_csv('outputs/merged_labeled_3class.csv')
        df_binary = pd.read_csv('outputs/merged_labeled_binary.csv')
        generate_eda_figures(df, df_binary)
    else:
        print("Run labeling.py first.")
