import pandas as pd
import requests
import logging 
import time
import re
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

TMDB_API_KEY = "725ff13f46d3ee977ebb76000ab184d3"

# Load movie and rating data
def load_data():
    try:
        movies = pd.read_csv('movies.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv('ratings.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
    except FileNotFoundError:
        movies = pd.read_csv('C:/Users/kiran/OneDrive/Desktop/AIML internship/movies.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv('C:/Users/kiran/OneDrive/Desktop/AIML internship/ratings.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')

    movies.columns = movies.columns.str.lower().str.strip()
    ratings.columns = ratings.columns.str.lower().str.strip()

    if 'genres' in movies.columns:
        movies['genres'] = movies['genres'].fillna('(no genres listed)')
        movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    else:
        movies['genres'] = [[] for _ in range(len(movies))]

    if 'title' in movies.columns:
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
        movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    merged_df = pd.merge(ratings, movies, on='movie_id')
    return merged_df, movies

# Train the SVD model
def train_model(merged_df):
    data = merged_df[['user_id', 'movie_id', 'rating']]
    reader = Reader(rating_scale=(0.5, 5.0))
    surprise_data = Dataset.load_from_df(data, reader)
    trainset, _ = train_test_split(surprise_data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model

# Fetch poster using TMDB API
logging.basicConfig(level=logging.INFO)

def fetch_poster(title, year):
    query = f"{title} {int(year) if pd.notnull(year) else ''}"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"[TMDb Retry {attempt+1}] Error fetching poster for '{title}' ({year}): {e}")
            time.sleep(1.5)

    return None

# Hybrid recommendation logic
def recommend_hybrid(user_id, model, ratings_data, movies_df, top_n=5):
    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]

    preferred_genres = set()
    for genres in liked_movies['genres']:
        preferred_genres.update(genres)

    seen_ids = user_ratings['movie_id'].tolist()
    all_ids = ratings_data['movie_id'].unique()
    unseen_ids = [m for m in all_ids if m not in seen_ids]

    preds = [(mid, model.predict(user_id, mid).est) for mid in unseen_ids]
    pred_df = pd.DataFrame(preds, columns=['movie_id', 'predicted_rating'])
    pred_df = pd.merge(pred_df, movies_df[['movie_id', 'title', 'genres', 'year']], on='movie_id')

    def matches_genre(genres):
        return len(preferred_genres.intersection(genres)) > 0

    pred_df['genre_match'] = pred_df['genres'].apply(matches_genre)
    filtered = pred_df[pred_df['genre_match']].sort_values(by='predicted_rating', ascending=False).head(top_n)

    filtered['poster_url'] = filtered.apply(lambda row: fetch_poster(row['title'], row['year']), axis=1)
    return filtered[['title', 'predicted_rating', 'poster_url']]
