import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# STEP 1: LOAD DATASETS
# ==============================
print("Loading datasets...")

try:
    movies = pd.read_csv("movie.csv")
    ratings = pd.read_csv("rating.csv")
except FileNotFoundError:
    print("ERROR: movies.csv or ratings.csv not found in this folder.")
    exit()

print("Movies Loaded:", movies.shape)
print("Ratings Loaded:", ratings.shape)


# ==============================
# STEP 2: CLEAN TITLES
# ==============================
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


movies["clean_title"] = movies["title"].apply(clean_title)


# ==============================
# STEP 3: TF-IDF SEARCH
# ==============================
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


def search_movie(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    idx = similarity.argsort()[-5:][::-1]
    return movies.iloc[idx][["movieId", "title"]]


# ==============================
# STEP 4: RECOMMENDATION SYSTEM
# ==============================
def recommend(movie_id):

    # Users who liked this movie
    similar_users = ratings[
        (ratings["movieId"] == movie_id) &
        (ratings["rating"] >= 4)
    ]["userId"].unique()

    if len(similar_users) == 0:
        return pd.DataFrame({"Message": ["Not enough similar users found"]})

    # Movies liked by similar users
    similar_user_recs = ratings[
        (ratings["userId"].isin(similar_users)) &
        (ratings["rating"] >= 4)
    ]["movieId"]

    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]

    # Movies liked by all users
    all_users = ratings[
        (ratings["movieId"].isin(similar_user_recs.index)) &
        (ratings["rating"] >= 4)
    ]

    all_user_recs = (
        all_users["movieId"].value_counts()
        / all_users["userId"].nunique()
    )

    # Score calculation
    df = pd.concat([similar_user_recs, all_user_recs], axis=1)
    df.columns = ["similar", "all"]
    df["score"] = df["similar"] / df["all"]

    df = df.sort_values(by="score", ascending=False)

    # Merge movie metadata
    df = df.merge(movies, left_index=True, right_on="movieId")

    # FINAL CLEAN OUTPUT (title + genres + score only)
    result = df[["title", "genres", "score"]].head(10).reset_index(drop=True)

    return result



# ==============================
# STEP 5: TERMINAL INTERFACE
# ==============================
print("\nMovie Recommender System Ready")
print("Type movie name to get recommendations")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter movie name: ")

    if text.lower() == "exit":
        print("Exiting recommender...")
        break

    results = search_movie(text)

    if results.empty:
        print("No movies found.\n")
        continue

    print("\nTop Matches:")
    print(results)

    movie_id = int(results.iloc[0]["movieId"])

    print("\nRecommended Movies:")
    print(recommend(movie_id))
    print("\n" + "-" * 60 + "\n")
