import streamlit as st
from recommender import load_data, train_model, recommend_hybrid
from utils import set_background_with_overlay
import os

st.set_page_config(page_title="Movie Recommender", layout="wide")

if os.path.exists("assets/background.png"):
    set_background_with_overlay("assets/background.png")

st.title("🎬 Movie Recommender 😎")
st.write("Your cozy space for discovering great films!")

merged_df, movies_df = load_data()
model = train_model(merged_df)

user_ids = merged_df['user_id'].unique()
user_id = st.selectbox("Select a User ID:", sorted(user_ids))

if st.button("🎯 Recommend Movies"):
    top_movies = recommend_hybrid(user_id, model, merged_df, movies_df)

    if top_movies.empty:
        st.warning("No recommendations found. Try a different user.")
    else:
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
