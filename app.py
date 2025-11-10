import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Merge and calculate average ratings
    movie_data = pd.merge(ratings, movies, on='movieId')
    movie_mean_rating = movie_data.groupby('title')['rating'].mean().reset_index()
    movie_mean_rating = movie_mean_rating.rename(columns={'rating': 'avg_rating'})
    movies = pd.merge(movies, movie_mean_rating, on='title', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    return movies

movies = load_data()

# -----------------------------------------------------------
# STEP 2: CREATE GENRE MATRIX
# -----------------------------------------------------------
count = CountVectorizer(tokenizer=lambda x: x.split('|'))
count_matrix = count.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# -----------------------------------------------------------
# STEP 3: RECOMMENDATION FUNCTION
# -----------------------------------------------------------
def recommend(movie_title, num=5):
    movie_title = movie_title.strip().lower()
    try:
        idx = movies[movies['title'].str.lower() == movie_title].index[0]
    except IndexError:
        return pd.DataFrame(columns=['title', 'genres', 'avg_rating'])

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres', 'avg_rating']].iloc[movie_indices]

# -----------------------------------------------------------
# STEP 4: FETCH MOVIE POSTER USING TMDB API
# -----------------------------------------------------------
API_KEY = "d047124b426ff38bd1037694f1a4b8ac"

def get_poster(title):
    try:
        # remove year and brackets for cleaner search
        cleaned_title = title.split('(')[0].strip()
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned_title}&language=en-US"
        response = requests.get(url)
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        # fallback if not found
        return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"
    except Exception as e:
        print("Poster fetch error:", e)
        return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"

# -----------------------------------------------------------
# STEP 5: STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="centered")

st.title("🎥 Movie Recommendation System")
st.markdown("##### Get personalized movie suggestions based on your preferences 🎬")

preferred_genres = st.multiselect(
    "🎭 Select your preferred genres:",
    ['Action', 'Adventure', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Thriller', 'Animation', 'Fantasy']
)

age_group = st.selectbox(
    "👤 Select your age group:",
    ['Under 18', '18-25', '26-35', '36-50', '50+']
)

platform = st.selectbox(
    "📺 Preferred viewing platform:",
    ['Netflix', 'Amazon Prime', 'Disney+', 'Hotstar', 'Others']
)

last_movie = st.text_input("🎞️ Last movie you watched:")
last_rating = st.slider("⭐ How much did you like it?", 1, 5, 3)

if st.button("🔍 Get Recommendations"):
    st.write("---")

    if last_movie and last_rating >= 4:
        st.subheader(f"Because you loved '{last_movie}' ❤️, you might also like:")
        recs = recommend(last_movie)
        if recs.empty:
            st.warning("Movie not found in dataset! Try a popular title like 'Inception' or 'Titanic'.")
        else:
            for _, row in recs.iterrows():
                col1, col2 = st.columns([1, 3])
                poster = get_poster(row['title'])
                with col1:
                    if poster:
                        st.image(poster, width=120)
                    else:
                        st.write("🎞️ No poster found")
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")
                    st.write(f"⭐ Average Rating: {round(row['avg_rating'], 2)}")
    elif preferred_genres:
        st.subheader("Based on your favorite genres 🎭")
        recs = movies[movies['genres'].apply(lambda x: any(g in x for g in preferred_genres))]
        recs = recs.sort_values(by='avg_rating', ascending=False).head(5)
        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])
            poster = get_poster(row['title'])
            with col1:
                if poster:
                    st.image(poster, width=120)
                else:
                    st.write("🎞️ No poster found")
            with col2:
                st.markdown(f"**{row['title']}**")
                st.caption(f"Genres: {row['genres']}")
                st.write(f"⭐ Average Rating: {round(row['avg_rating'], 2)}")
    else:
        st.warning("Please select genres or enter a movie title.")

st.write("---")
st.caption("Built with ❤️ by Aishwarya for Internship Project")
