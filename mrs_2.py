#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

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
TMDB_API_KEY = "725ff13f46d3ee977ebb76000ab184d3"

# === Page Config ===
st.set_page_config(page_title="Movie Recommender", layout="wide")

# === Theme Toggle ===
theme = st.radio("Choose Theme:", ["Dark", "Light", "Cozy"], horizontal=True)
if theme == "Dark":
    overlay_color = "rgba(0, 0, 0, 0.6)"
elif theme == "Light":
    overlay_color = "rgba(255, 255, 255, 0.6)"
else:
    overlay_color = "rgba(30, 30, 30, 0.3)"

# === Set Background ===
def set_background_with_overlay(image_file, overlay_color):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient({overlay_color}, {overlay_color}), 
                              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_with_overlay("background.png", overlay_color)

# === Load Data ===
@st.cache_data
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
        st.warning("⚠️ 'genres' column not found in movies.csv.")
        movies['genres'] = [[] for _ in range(len(movies))]

    if 'title' in movies.columns:
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
        movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    merged_df = pd.merge(ratings, movies, on='movie_id')
    return merged_df, movies

# === Train Model ===
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
            logging.warning(f"[Retry {attempt+1}] Poster fetch error: {e}")
            time.sleep(1.5)
    return None

# === Hybrid Recommender ===
def recommend_hybrid(user_id, model, ratings_data, movies_df, top_n=5):
    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]

    preferred_genres = set()
    for genres in liked_movies['genres']:
        preferred_genres.update(genres)

    seen_ids = user_ratings['movie_id'].tolist()
    unseen_ids = [m for m in ratings_data['movie_id'].unique() if m not in seen_ids]

    preds = [(mid, model.predict(user_id, mid).est) for mid in unseen_ids]
    pred_df = pd.DataFrame(preds, columns=['movie_id', 'predicted_rating'])
    pred_df = pd.merge(pred_df, movies_df[['movie_id', 'title', 'genres', 'year']], on='movie_id')

    def matches_genre(genres):
        return len(preferred_genres.intersection(genres)) > 0

    pred_df['genre_match'] = pred_df['genres'].apply(matches_genre)
    filtered = pred_df[pred_df['genre_match']].sort_values(by='predicted_rating', ascending=False).head(top_n)
    filtered['poster_url'] = filtered.apply(lambda row: fetch_poster(row['title'], row['year']), axis=1)
    return filtered[['title', 'predicted_rating', 'poster_url']]

# === Content-Based Recommender ===
def find_similar_movies(title_input, movies_df, top_n=5):
    title_input = title_input.lower()
    matched = movies_df[movies_df['title'].str.lower().str.contains(title_input)]
    
    if matched.empty:
        return pd.DataFrame()
    
    target_genres = matched.iloc[0]['genres']
    if not target_genres:
        return pd.DataFrame()

    def genre_similarity(genres):
        return len(set(genres).intersection(set(target_genres)))

    movies_df['similarity'] = movies_df['genres'].apply(genre_similarity)
    similar = movies_df[movies_df['title'].str.lower() != matched.iloc[0]['title'].lower()]
    top_similar = similar.sort_values(by='similarity', ascending=False).head(top_n)
    top_similar['poster_url'] = top_similar.apply(lambda row: fetch_poster(row['title'], row['year']), axis=1)
    return top_similar[['title', 'genres', 'poster_url']]

# === Streamlit UI ===
st.title("🎬 Movie Recommender 😎")
st.write("Your cozy space for discovering great films!")

merged_df, movies_df = load_data()
model = train_model(merged_df)

st.markdown("---")
st.markdown("## 🔍 Find Similar Movies (Content-Based)")
search_query = st.text_input("Enter a movie title:")
if search_query:
    similar_movies = find_similar_movies(search_query, movies_df)
    if not similar_movies.empty:
        st.subheader("🎥 Movies Similar to Your Search")
        for _, row in similar_movies.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if row['poster_url']:
                    st.image(row['poster_url'], width=100)
                else:
                    st.write("No Poster")
            with col2:
                st.markdown(f"**{row['title']}**")
                st.markdown(f"Genres: {', '.join(row['genres'])}")
            st.markdown("---")
    else:
        st.warning("No similar movies found.")

st.markdown("---")
st.markdown("## 🧠 Personalized Recommendations")
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

# === Footer ===
st.markdown("---")
st.markdown("<center>© 2025 Movie Recommender. Built with ❤️ using Streamlit.</center>", unsafe_allow_html=True)


# In[ ]:




