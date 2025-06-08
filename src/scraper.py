# src/scraper.py

from google_play_scraper import reviews, Sort, app
import pandas as pd

def fetch_reviews_for_app(app_id, max_count=400):
    """Fetch reviews for a single app from Google Play Store."""
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
            "review": r['content'],
            "rating": r['score'],
            "date": r['at'].strftime("%Y-%m-%d"),
            "app_name": app_name
        })

    return reviews_data


def save_reviews_to_csv(reviews_list, filename):
    """Save the reviews to a CSV file."""
    df = pd.DataFrame(reviews_list)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} reviews to {filename}")
