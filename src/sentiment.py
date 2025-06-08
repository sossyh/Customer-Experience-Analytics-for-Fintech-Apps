# src/sentiment_analysis.py

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load model once globally for reuse
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_sentiment(text):
    """Classify sentiment using the transformer pipeline."""
    try:
        result = sentiment_pipeline(text[:512])[0]  # Truncate to max token length
        label = result['label']
        score = result['score']

        if label == 'NEGATIVE':
            sentiment = 'negative'
        elif label == 'POSITIVE':
            sentiment = 'positive'
        else:
            sentiment = 'neutral'

        return pd.Series([sentiment, score])
    except Exception as e:
        return pd.Series(['neutral', 0.0])  # Fallback

def analyze_sentiments(df):
    """Apply sentiment analysis to a DataFrame and add sentiment columns."""
    tqdm.pandas(desc="🔍 Analyzing Sentiments")
    df[['sentiment_label', 'sentiment_score']] = df['review'].progress_apply(classify_sentiment)
    return df

def aggregate_sentiments(df):
    """Aggregate sentiment scores by bank and rating."""
    agg_df = df.groupby(['bank', 'rating']).agg(
        mean_sentiment_score=('sentiment_score', 'mean'),
        sentiment_distribution=('sentiment_label', lambda x: x.value_counts(normalize=True).to_dict())
    ).reset_index()
    return agg_df
