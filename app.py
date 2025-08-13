import streamlit as st
import pickle
import pandas as pd

# Load local data
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.title("🎬 Simple Movie Recommender")

movie_name = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    recs = recommend(movie_name)
    st.write("### Recommended Movies:")
    for m in recs:
        st.write(m)