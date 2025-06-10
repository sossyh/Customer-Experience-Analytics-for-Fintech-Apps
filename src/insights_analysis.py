import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def load_data(filepath):
    """Load data and ensure theme column is properly formatted"""
    df = pd.read_csv(filepath)
    if 'theme' in df.columns and isinstance(df['theme'].iloc[0], str):
        df['theme'] = df['theme'].apply(eval)  # Convert string to list
    return df

def analyze_themes(df, min_count=5):
    """Analyze theme frequencies without any bank/source grouping"""
    if 'theme' not in df.columns:
        raise ValueError("Missing 'theme' column in data")
    
    # Explode themes into separate rows
    theme_counts = (
        df.explode('theme')
        ['theme'].value_counts()
        .reset_index()
    )
    theme_counts.columns = ['theme', 'count']
    return theme_counts[theme_counts['count'] >= min_count]

def analyze_keywords(df, min_count=5):
    """Analyze keyword frequencies if needed"""
    if 'keywords' not in df.columns:
        return None
    
    keyword_counts = (
        df.explode('keywords')
        ['keywords'].value_counts()
        .reset_index()
    )
    keyword_counts.columns = ['keyword', 'count']
    return keyword_counts[keyword_counts['count'] >= min_count]

def plot_top_items(count_df, item_type="Theme", top_n=15, save_path=None):
    """Generic function to plot top themes/keywords"""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='count', 
        y=item_type.lower(),
        data=count_df.head(top_n),
        palette="viridis"
    )
    plt.title(f"Top {top_n} {item_type}s")
    plt.xlabel("Frequency")
    plt.ylabel(item_type)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def generate_wordcloud(text_series, title="Review Word Cloud", save_path=None):
    """Generate word cloud from text data"""
    text = " ".join(text_series.dropna().astype(str))
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        stopwords=None,
        collocations=True
    ).generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()