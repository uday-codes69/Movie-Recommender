import streamlit as st
import requests
import os
import pickle
import pandas as pd

# ---------- SETTINGS ----------
# Hugging Face download URLs
MOVIES_URL = "https://huggingface.co/uday-codes69/movie-recommender-data/resolve/main/movies.pkl"
SIMILARITY_URL = "https://huggingface.co/uday-codes69/movie-recommender-data/resolve/main/similarity.pkl"

# Local filenames
MOVIES_FILE = "movies.pkl"
SIMILARITY_FILE = "similarity.pkl"

# ---------- HELPERS ----------
def download_file(url, local_path):
    """Download file from Hugging Face if not already cached locally."""
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}...")
        r = requests.get(url)
        r.raise_for_status()  # fail fast if URL is wrong
        with open(local_path, "wb") as f:
            f.write(r.content)
        st.success(f"{os.path.basename(local_path)} downloaded.")
    else:
        st.write(f"{os.path.basename(local_path)} already present.")

def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- DOWNLOAD DATA ----------
download_file(MOVIES_URL, MOVIES_FILE)
download_file(SIMILARITY_URL, SIMILARITY_FILE)

# ---------- LOAD DATA ----------
movies_df = load_pickle(MOVIES_FILE)
similarity = load_pickle(SIMILARITY_FILE)

# ---------- STREAMLIT UI ----------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie to get recommendations:",
    movies_df['title'].values
)

def recommend(movie):
    index = movies_df[movies_df['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    recommended_movies = [movies_df.iloc[i[0]].title for i in distances]
    return recommended_movies

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)
