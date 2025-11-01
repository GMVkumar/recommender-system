import streamlit as st
import joblib
import requests

# ---- Load Data ----
movies = joblib.load('movies.pkl')
similarity = joblib.load('similarity_compressed.pkl.gz')


# ---- Fetch Poster ----
def fetch_poster(movie_title):
    api_key = "a83c924bb71013432f2379a4c8898cba"
    movie_id = movies[movies['title'] == movie_title]['movie_id'].values[0]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()
    if 'poster_path' in data and data['poster_path'] is not None:
        full_path = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        return full_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"


# ---- Recommend Function ----
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for idx in movie_list:
        title = movies.iloc[idx[0]].title
        recommended_movies.append(title)
        recommended_posters.append(fetch_poster(title))
    return recommended_movies, recommended_posters


# ---- Streamlit UI ----
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to see top 5 similar recommendations with posters!")

selected_movie = st.selectbox("🎞️ Choose a movie", movies['title'].values)

if st.button("Show Recommendations"):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
