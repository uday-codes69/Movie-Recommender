import streamlit as st
import pickle
import pandas as pd
import joblib

# Load data
movies = pd.DataFrame(pickle.load(open('movies.pkl', 'rb')))
similarity = joblib.load('similarity.pkl')

# Recommend function
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return movies.iloc[[i[0] for i in movie_list]]['title'].tolist()

# UI
st.title("🎬 Movie Recommender")
movie_name = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    for m in recommend(movie_name):
        st.write(m)