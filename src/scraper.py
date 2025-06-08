# src/scraper.py
from google_play_scraper import reviews, app
import pandas as pd
from datetime import datetime

def fetch_reviews_for_app(app_id, max_count=400):
    """Fetch reviews for a single app from Google Play Store."""
    try:
        app_details = app(app_id, lang='en', country='us')
        app_name = app_details['title']
        
        result, _ = reviews(
            app_id,
            lang='en',
            country='us',
            count=max_count
        )

        reviews_data = []
        for r in result:
            reviews_data.append({
                "review": r.get('content', '').strip(),
                "rating": r.get('score', 0),
                "date": r.get('at', datetime.now()).strftime("%Y-%m-%d"),
                "bank": app_name,
                "source": "Google Play"
            })
        
        return reviews_data
    
    except Exception as e:
        print(f"Error fetching reviews for {app_id}: {str(e)}")
        return []

def preprocess_reviews(reviews_list):
    """Clean and preprocess the reviews data."""
    df = pd.DataFrame(reviews_list)
    
    # Handle missing data
    df['review'].fillna('', inplace=True)
    df['rating'].fillna(0, inplace=True)
    df['date'].fillna(datetime.now().strftime("%Y-%m-%d"), inplace=True)
    
    # Remove duplicates (keeping first occurrence)
    df.drop_duplicates(subset=['review', 'bank'], keep='first', inplace=True)
    
    # Convert rating to integer if it's not already
    df['rating'] = df['rating'].astype(int)
    
    # Ensure date format consistency
    df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    
    return df

def save_reviews_to_csv(reviews_list, filename):
    """Preprocess and save the reviews to a CSV file."""
    df = preprocess_reviews(reviews_list)
    
    # Ensure we have all required columns
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    df = df[required_columns]
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df)} preprocessed reviews to {filename}")

# Example usage:
# reviews_data = fetch_reviews_for_app("com.example.app")
# save_reviews_to_csv(reviews_data, "reviews.csv")