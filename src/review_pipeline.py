# src/review_pipeline.py

import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def preprocess_text(text):
    """Clean and lemmatize text using spaCy."""
    if pd.isnull(text):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_keywords_tfidf(texts, top_n=10):
    """Extract top N keywords per document using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    keywords_per_doc = []
    for row in tfidf_matrix:
        row_array = row.toarray()[0]
        top_indices = row_array.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices if row_array[i] > 0]
        keywords_per_doc.append(keywords)
    return keywords_per_doc

def group_keywords_by_theme(keywords):
    """Group keywords into themes based on rules."""
    keyword_theme_map = {
        "login": "Account Access",
        "logout": "Account Access",
        "password": "Account Access",
        "transfer": "Transaction Performance",
        "transaction": "Transaction Performance",
        "delay": "Transaction Performance",
        "slow": "Transaction Performance",
        "crash": "Reliability",
        "bug": "Reliability",
        "freeze": "Reliability",
        "feature": "Feature Requests",
        "update": "Feature Requests",
        "interface": "User Interface & Experience",
        "design": "User Interface & Experience",
        "easy": "User Interface & Experience",
        "support": "Customer Support",
        "help": "Customer Support",
        "agent": "Customer Support"
    }

    themes = set()
    for word in keywords:
        theme = keyword_theme_map.get(word.lower(), "Other")
        themes.add(theme)
    return list(themes)

def run_pipeline(df, review_col="review", id_col=None):
    """Run the full pipeline: preprocess, extract keywords, assign themes."""
    tqdm.pandas(desc="🧼 Preprocessing Reviews")
    df["cleaned_review"] = df[review_col].progress_apply(preprocess_text)

    tqdm.pandas(desc="🧠 Extracting Keywords")
    df["keywords"] = extract_keywords_tfidf(df["cleaned_review"].tolist())

    tqdm.pandas(desc="🔍 Grouping Keywords into Themes")
    df["theme"] = df["keywords"].progress_apply(group_keywords_by_theme)

    # Select relevant columns
    columns = [review_col, "cleaned_review", "keywords", "theme"]
    if id_col and id_col in df.columns:
        columns.insert(0, id_col)
    return df[columns]
