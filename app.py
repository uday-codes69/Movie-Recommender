import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# Load movies data (movies.pkl should have 'title' and 'tags')
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Compute similarity matrix
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommend movies
def recommend(movie):
    if movie not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index].flatten()
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [(movies.iloc[i[0]].title, distances[i[0]]) for i in movie_list]

# Streamlit UI
st.title("🎬 Movie Recommender System")

tab1, tab2 = st.tabs(["🔎 Recommendation", "📊 Dashboard"])

# --- TAB 1: Recommendation ---
with tab1:
    movie_name = st.selectbox("Choose a movie", movies['title'].values)

    if st.button("Recommend"):
        recs = recommend(movie_name)
        if recs:
            st.subheader("Recommended Movies:")
            for m, score in recs:
                st.write(f"🎥 {m}  (similarity: {score:.2f})")

            # Plot similarity scores
            rec_df = pd.DataFrame(recs, columns=["Movie", "Score"])
            fig, ax = plt.subplots()
            sns.barplot(x="Score", y="Movie", data=rec_df, ax=ax, palette="viridis")
            ax.set_title("Similarity Scores for Recommendations")
            st.pyplot(fig)

        else:
            st.warning("Movie not found in database!")

# --- TAB 2: Dashboard / EDA ---
with tab2:
    st.subheader("Dataset Overview")
    st.write(f"Total Movies: {len(movies)}")

    # WordCloud of tags
    st.subheader("Most Common Tags")
    all_tags = " ".join(movies['tags'])
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(all_tags)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Distribution of movie lengths (if available in dataset)
    if 'genre' in movies.columns:
        st.subheader("Movies per Genre")
        fig, ax = plt.subplots()
        movies['genre'].value_counts().head(10).plot(kind='bar', ax=ax, color="skyblue")
        st.pyplot(fig)