import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies data (movies.pkl should have 'title' and 'tags')
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Compute similarity matrix on the fly
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommend movies function
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index].flatten()
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.title("🎬 Simple Movie Recommender")

movie_name = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    recs = recommend(movie_name)
    if recs:
        st.subheader("Recommended Movies:")
        for m in recs:
            st.write(m)
    else:
        st.warning("Movie not found in database!")