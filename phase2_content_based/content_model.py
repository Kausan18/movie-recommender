"""
Phase 2 — Content-Based Filtering
Builds a TF-IDF genre+tag matrix and computes a cosine similarity matrix.
Run this script once to train and serialise the model to models/.

Usage:
    python phase2_content_based/content_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

RAW_MOVIES = os.path.join(DATA_DIR, "raw", "movies.csv")
RAW_TAGS   = os.path.join(DATA_DIR, "raw", "tags.csv")

# Fallback: if no data/raw/ directory, look in the project root
if not os.path.exists(RAW_MOVIES):
    RAW_MOVIES = os.path.join(os.path.dirname(__file__), "..", "movies.csv")
    RAW_TAGS   = os.path.join(os.path.dirname(__file__), "..", "tags.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(RAW_MOVIES)
    tags   = pd.read_csv(RAW_TAGS)
    print(f"Loaded {len(movies):,} movies  |  {len(tags):,} tags")
    return movies, tags


def build_soup(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Merge genres + user tags into a single text string per movie.

    Genre string  → 'Action SciFi Thriller'  (pipe-delimited → space-delimited)
    Tags          → aggregated, lowercased, de-duplicated, appended to genres

    Genres carry more weight because every movie has them;
    tags are a bonus signal for movies that have community annotations.
    """
    # ── 1. Clean genres ───────────────────────────────────────────────────
    movies = movies.copy()
    movies["genres_clean"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("-", "", regex=False)   # SciFi not Sci-Fi so it stays one token
        .str.lower()
    )

    # ── 2. Aggregate tags per movie ───────────────────────────────────────
    tag_agg = (
        tags.dropna(subset=["tag"])
        .assign(tag=lambda df: df["tag"].str.lower().str.strip())
        .groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(sorted(set(ts))))
        .reset_index()
        .rename(columns={"tag": "tags_clean"})
    )

    # ── 3. Join ───────────────────────────────────────────────────────────
    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tags_clean"] = movies["tags_clean"].fillna("")

    # Genres weighted ×2 (repeat them) to outweigh sparse tags
    movies["soup"] = movies["genres_clean"] + " " + movies["genres_clean"] + " " + movies["tags_clean"]
    movies["soup"] = movies["soup"].str.strip()

    print(f"\nSample soups:")
    for _, row in movies.sample(3, random_state=42).iterrows():
        print(f"  [{row['title']}]  →  {row['soup'][:80]}...")

    return movies


def build_tfidf(movies: pd.DataFrame):
    """
    Fit TF-IDF on the soup column.
    Returns the vectoriser and the sparse feature matrix.
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),      # unigrams + bigrams ("sci fi", "dark comedy")
        min_df=2,                # ignore tokens in fewer than 2 movies
        max_features=5_000,
        sublinear_tf=True,       # log(1+tf) dampens very common genres
    )
    tfidf_matrix = tfidf.fit_transform(movies["soup"])
    print(f"\nTF-IDF matrix shape : {tfidf_matrix.shape}")
    print(f"Vocabulary size     : {len(tfidf.vocabulary_):,}")
    return tfidf, tfidf_matrix


def build_cosine_sim(tfidf_matrix) -> np.ndarray:
    """
    Compute the full (n_movies × n_movies) cosine similarity matrix.
    For 9 742 movies this is ~380 MB in float32 — manageable in RAM.
    """
    print("\nComputing cosine similarity matrix…")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype("float32")
    print(f"Cosine sim shape    : {cosine_sim.shape}")
    return cosine_sim


def save_artifacts(movies, tfidf, cosine_sim):
    movies_clean_path   = os.path.join(DATA_DIR, "processed", "movies_clean.csv")
    tfidf_path          = os.path.join(MODELS_DIR, "tfidf_vectoriser.pkl")
    cosine_path         = os.path.join(MODELS_DIR, "cosine_sim_matrix.pkl")

    movies.to_csv(movies_clean_path, index=False)
    joblib.dump(tfidf,      tfidf_path)
    joblib.dump(cosine_sim, cosine_path)

    print(f"\nSaved:")
    print(f"  {movies_clean_path}")
    print(f"  {tfidf_path}")
    print(f"  {cosine_path}")


def train():
    print("=" * 55)
    print("  Phase 2 — Content-Based Model: Training")
    print("=" * 55)

    movies, tags = load_data()
    movies       = build_soup(movies, tags)
    tfidf, tfidf_matrix = build_tfidf(movies)
    cosine_sim   = build_cosine_sim(tfidf_matrix)
    save_artifacts(movies, tfidf, cosine_sim)

    print("\n✓ Training complete. Run recommend.py to test recommendations.")
    return movies, tfidf, cosine_sim


if __name__ == "__main__":
    train()