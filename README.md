# 🎬 Movie Recommendation System

A hybrid movie recommendation app built with **Streamlit**, combining **collaborative filtering (SVD)** and **content-based filtering (genre-based)**. It also integrates with the **TMDb API** to fetch high-quality movie posters for a cozy and visually engaging experience.

## 🚀 Features

- 🎯 **Hybrid Recommendation**: Combines user preferences with genre-based filtering
- 🧠 **Collaborative Filtering**: Uses SVD from `scikit-surprise`
- 🎥 **Movie Posters**: Integrated with TMDb API for visual recommendations
- 🎨 **Custom Background**: Supports a themed background image with a dark overlay
- 📊 **Interactive UI**: Simple, modern UI with dropdowns and buttons using Streamlit

---

## 📁 Project Structure

movie-recommender/
├── app.py # Main Streamlit app
├── recommender.py # Core logic (data loading, SVD training, hybrid filtering)
├── utils.py # TMDb poster fetcher and background overlay
├── MRS_hybrid.py # overall code in a single file(optional)
├── data/
│ ├── movies.csv # Movie metadata (title, genres, year)
│ └── ratings.csv # User ratings
├── assets/
│ └── background.png # Optional background image
├── requirements.txt # Python dependencies
└── README.md # This file

---

🔑 TMDb API Key
To fetch movie posters from The Movie Database (TMDb):
  -Create a free account at https://www.themoviedb.org
  -Navigate to your profile > Settings > API > Create an API key
  -Replace the placeholder key in utils.py with your actual API key
📊 Datasets Used
This app uses the MovieLens dataset (e.g., movies.csv, ratings.csv)
Ensure:
  movies.csv contains: movie_id, title, genres
  ratings.csv contains: user_id, movie_id, rating, timestamp

📸 Screenshot
<img width="1297" height="640" alt="2025-07-26" src="https://github.com/user-attachments/assets/a571e89a-feaa-4bfa-b4e5-46b7893ad5ba" />
