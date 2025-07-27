#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os
import re
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# === Load Datasets ===
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv('ratings.csv', sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
    except FileNotFoundError:
        movies = pd.read_csv("C:/Users/kiran/Downloads/movies.csv", sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        ratings = pd.read_csv("C:/Users/kiran/Downloads/ratings.csv", sep='\t', encoding='ISO-8859-1', on_bad_lines='skip')

    # Preprocess movies
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if x != '(no genres listed)' else [])
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
    movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

    # Clean column names
    ratings.columns = ratings.columns.str.lower()
    movies.columns = movies.columns.str.lower()

    # Convert timestamp
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Merge
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

# === Recommendation Logic ===
def recommend_movies(user_id, model, ratings_data, movies, top_n=5):
    seen = ratings_data[ratings_data['user_id'] == user_id]['movie_id'].tolist()
    unseen = [m for m in ratings_data['movie_id'].unique() if m not in seen]
    preds = [(mid, model.predict(user_id, mid).est) for mid in unseen]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]
    top_df = pd.DataFrame(top_preds, columns=['movie_id', 'predicted_rating'])
    return pd.merge(top_df, movies[['movie_id', 'title']], on='movie_id')[['title', 'predicted_rating']]

# === Streamlit UI ===
st.title("🎬 Movie Recommender System - Collaborative Filtering (SVD)")

# Load and train
merged_df, movies_df = load_data()
model = train_model(merged_df)

# User selection
user_ids = merged_df['user_id'].unique()
user_id = st.selectbox("Select a User ID:", sorted(user_ids))

# Recommend button
if st.button("Recommend Top 5 Movies"):
    top_movies = recommend_movies(user_id, model, merged_df, movies_df)
    st.subheader(f"Top 5 Recommended Movies for User {user_id}")
    st.table(top_movies)

