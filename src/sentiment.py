import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER once globally
vader_analyzer = SentimentIntensityAnalyzer()

def classify_sentiment_vader(text):
    """Classify sentiment using VADER."""
    try:
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return pd.Series([sentiment, compound])
    except:
        return pd.Series(['neutral', 0.0])

def classify_sentiment_textblob(text):
    """Classify sentiment using TextBlob."""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return pd.Series([sentiment, polarity])
    except:
        return pd.Series(['neutral', 0.0])

def classify_sentiment(text, model='vader'):
    """Wrapper function to select which model to use."""
    if model == 'vader':
        return classify_sentiment_vader(text)
    elif model == 'textblob':
        return classify_sentiment_textblob(text)
    else:
        raise ValueError(f"Unsupported model: {model}")

def analyze_sentiments(df, model='vader'):
    """Apply sentiment analysis to a DataFrame and add sentiment columns."""
    tqdm.pandas(desc=f"🔍 Analyzing Sentiments ({model})")
    df[['sentiment_label', 'sentiment_score']] = df['review'].progress_apply(lambda x: classify_sentiment(x, model=model))
    return df

def aggregate_sentiments(df):
    """Aggregate sentiment scores by bank and rating."""
    agg_df = df.groupby(['source', 'rating']).agg(
        mean_sentiment_score=('sentiment_score', 'mean'),
        sentiment_distribution=('sentiment_label', lambda x: x.value_counts(normalize=True).to_dict())
    ).reset_index()
    return agg_df
