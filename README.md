# 🎬 Movie Recommender System

A simple movie recommendation app built with **Python** and **Streamlit**.  
Pick a movie and the app suggests 5 similar movies. No API key required.

---

## 📌 Features
- Recommend top 5 similar movies.
- Easy web UI using Streamlit.
- Works fully offline with local pickles (`movies.pkl`, `model.pkl`).

---

## 🛠 Technologies Used
- Python  
- Pandas & NumPy  
- scikit-learn (TF-IDF / cosine similarity)  
- Streamlit (UI)

---

## 🚀 Quick Start (local)
1. Clone the repo:
git clone https://github.com/uday-codes69/movie-recommender.git
cd movie-recommender 

3. Install depedencies:
pip install -r requirements.txt

4. Run the App:
streamlit run app.py


movie-recommender/
├── app.py             # Streamlit app
├── movies.pkl         # Movie dataset (DataFrame with 'title' column)
├── model.pkl          # Similarity matrix (numpy array or pickle)
├── requirements.txt   # Python dependencies
├── README.md

