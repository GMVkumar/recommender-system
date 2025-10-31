import pandas as pd
import joblib



credit = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')
movies.head()

movies = movies.merge(credit, on='title')

movies.head()

movies.info()

Movies = movies[['movie_id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew' ]]
Movies.head()

Movies.isnull().sum()


Movies.dropna(inplace=True)


Movies.duplicated()

import ast

def convert(obj):
  l = []
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l

Movies['genres'] = Movies['genres'].apply(convert)

Movies.head()

Movies['keywords'] = Movies['keywords'].apply(convert)

Movies.head()


def convert(obj):
  l = []
  counter=0
  for i in ast.literal_eval(obj):
    if counter != 3:
     l.append(i['name'])
     counter += 1
    else:
      break
  return l

Movies['cast'] = Movies['cast'].apply(convert)

Movies.head()

def fetch_director(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      l.append(i['name'])
  return l

Movies['crew'] = Movies['crew'].apply(fetch_director)

Movies.head()

Movies['overview'] = Movies['overview'].apply(lambda x:x.split())

Movies.head()

Movies['genres'] = Movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
Movies['overview'] = Movies['overview'].apply(lambda x:[i.replace(" ", "") for i in x])
Movies['keywords'] = Movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
Movies['cast'] = Movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
Movies['crew'] = Movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

Movies.head()

Movies['tag'] = Movies['overview'] + Movies['genres'] + Movies['cast'] + Movies['crew'] + Movies['keywords']

Movies['tag'] = Movies['tag'].apply(lambda x:' '.join(x))

Movies.head()

new_df = Movies[['movie_id', 'title', 'tag']]


new_df['tag'] = new_df['tag'].apply(lambda x: x.lower())

new_df.head()


from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_df['tag']= new_df['tag'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 5000, stop_words='english')
vector = cv.fit_transform(new_df['tag']).toarray()


cv.get_feature_names_out()


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distance = similarity[movie_index]
  movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
  for i in movie_list:
    print(new_df.iloc[i[0]].title)

joblib.dump(new_df, 'movies.pkl')
joblib.dump(similarity, 'similarity.pkl')

print("✅ Models saved successfully!")

import joblib, gzip

model = joblib.load("similarity.pkl")
with gzip.open("similarity_compressed.pkl.gz", "wb") as f:
    joblib.dump(model, f)