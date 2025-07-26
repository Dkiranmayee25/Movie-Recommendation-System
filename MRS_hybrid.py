#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import requests
import logging 
import time
import re
import base64
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# === TMDb API Key ===
TMDB_API_KEY = "725ff13f46d3ee977ebb76000ab184d3"  # ← Replace this with your real key

# === Load Data ===
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv('ratings.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
    except FileNotFoundError:
        movies = pd.read_csv('C:/Users/kiran/OneDrive/Desktop/AIML internship/movies.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv('C:/Users/kiran/OneDrive/Desktop/AIML internship/ratings.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')

    # Preprocess movies
   # Normalize column names
    movies.columns = movies.columns.str.lower().str.strip()
    ratings.columns = ratings.columns.str.lower().str.strip()
    
    # Handle 'genres' column
    if 'genres' in movies.columns:
        movies['genres'] = movies['genres'].fillna('(no genres listed)')
        movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    else:
        st.warning("⚠️ 'genres' column not found in movies.csv. Using empty genres list.")
        movies['genres'] = [[] for _ in range(len(movies))]
    
    # Extract and clean year from title
    if 'title' in movies.columns:
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
        movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)



    # Clean column names
    ratings.columns = ratings.columns.str.lower()
    movies.columns = movies.columns.str.lower()

    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    merged_df = pd.merge(ratings, movies, on='movie_id')
    return merged_df, movies

# === Train SVD Model ===
@st.cache_resource
def train_model(merged_df):
    data = merged_df[['user_id', 'movie_id', 'rating']]
    reader = Reader(rating_scale=(0.5, 5.0))
    surprise_data = Dataset.load_from_df(data, reader)
    trainset, _ = train_test_split(surprise_data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model

# === TMDb Poster Fetch ===
logging.basicConfig(level=logging.INFO)

@st.cache_data(show_spinner=False)
def fetch_poster(title, year):
    query = f"{title} {int(year) if pd.notnull(year) else ''}"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    
    for attempt in range(3):  # Retry 3 times
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
            break  # No results, no point retrying
        except requests.exceptions.RequestException as e:
            logging.warning(f"[TMDb Retry {attempt+1}] Error fetching poster for '{title}' ({year}): {e}")
            time.sleep(1.5)  # Wait a bit before retry

    return None  # Fallback if all retries fail
# === Hybrid Recommendation Logic ===
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

    # Add posters
    filtered['poster_url'] = filtered.apply(lambda row: fetch_poster(row['title'], row['year']), axis=1)
    return filtered[['title', 'predicted_rating', 'poster_url']]

# === Streamlit UI ===
# Function to set background with tint overlay
def set_background_with_overlay(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background with overlay
set_background_with_overlay("background.png")

# Streamlit content
st.set_page_config(page_title="MRS", layout="wide")
st.title("🎬 Movie Recommender 😎😎")
st.write("Your cozy space for discovering great films!")

merged_df, movies_df = load_data()
model = train_model(merged_df)

user_ids = merged_df['user_id'].unique()
user_id = st.selectbox("Select a User ID:", sorted(user_ids))



if st.button("🎯 Recommend Movies"):
    top_movies = recommend_hybrid(user_id, model, merged_df, movies_df)

    st.subheader(f"🎉 Top 5 Recommendations for User {user_id}")

    for _, row in top_movies.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            if row['poster_url']:
                st.image(row['poster_url'], width=120)
            else:
                st.markdown("No poster")
        with col2:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"⭐ Predicted Rating: {row['predicted_rating']:.2f}")
        st.markdown("---")


# In[ ]:




