# src/keyword_extraction.py

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def preprocess_text(text):
    """Clean and lemmatize text using spaCy."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_keywords_tfidf(corpus, top_n=20, ngram_range=(1, 2)):
    """Extract top TF-IDF keywords and phrases from a list of reviews."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]

def group_keywords_by_theme(keywords):
    """Manually group related keywords into themes."""
    themes = defaultdict(list)
    for kw, score in keywords:
        if re.search(r'login|access|password|signin', kw):
            themes['Account Access Issues'].append(kw)
        elif re.search(r'transfer|delay|transaction|payment|load', kw):
            themes['Transaction Performance'].append(kw)
        elif re.search(r'interface|design|ux|ui|navigation', kw):
            themes['User Interface & Experience'].append(kw)
        elif re.search(r'support|help|agent|response|feedback', kw):
            themes['Customer Support'].append(kw)
        elif re.search(r'feature|add|option|update', kw):
            themes['Feature Requests'].append(kw)
        else:
            themes['Other'].append(kw)
    return dict(themes)
