"""
Run this once to train and save the SVD model.

Usage:
    python phase3_collaborative_filtering/train.py

Outputs:
    models/svd_model.pkl
    data/processed/user_item_matrix.pkl
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib

from phase3_collaborative_filtering.matrix_factorisation import (
    build_user_item_matrix, compute_sparsity
)
from phase3_collaborative_filtering.svd_model import (
    train_svd, evaluate_model, get_recommendations, save_model, build_sparse_matrix
)

ROOT_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_PATH         = os.path.join(ROOT_DIR, "models", "svd_model.pkl")

# ── Pre-filter thresholds ──────────────────────────────────────────────────────
# Filters out low-signal users/movies before building the user-item matrix.
# On 100K these thresholds drop almost nothing (most users have many ratings).
# On 25M they trim the matrix from 162K×62K to a much more tractable size
# (~130K users × 45K movies) while keeping 18–20M of the 25M ratings.
MIN_RATINGS_PER_USER  = 20   # drop users who rated fewer than 20 movies
MIN_RATINGS_PER_MOVIE = 10   # drop movies with fewer than 10 ratings


def main():
    print("=" * 60)
    print("PHASE 3 — COLLABORATIVE FILTERING TRAINING")
    print("=" * 60)

    # ── [1/6] Load ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    ratings_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "ratings.csv"))
    movies_df  = pd.read_csv(os.path.join(DATA_RAW_DIR, "movies.csv"))
    print(f"  Ratings : {len(ratings_df):,}")
    print(f"  Users   : {ratings_df['userId'].nunique():,}")
    print(f"  Movies  : {ratings_df['movieId'].nunique():,}")

    # ── [2/6] Pre-filter ──────────────────────────────────────────────────────
    # This block is safe to run on 100K — it just filters very little.
    # On 25M it is essential: without it, the user-item matrix is 10 billion cells.
    print(f"\n[2/6] Pre-filtering (min {MIN_RATINGS_PER_USER} ratings/user,"
          f" min {MIN_RATINGS_PER_MOVIE} ratings/movie)...")

    user_counts  = ratings_df.groupby("userId")["rating"].count()
    movie_counts = ratings_df.groupby("movieId")["rating"].count()

    ratings_df = ratings_df[
        ratings_df["userId"].isin(user_counts[user_counts >= MIN_RATINGS_PER_USER].index) &
        ratings_df["movieId"].isin(movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index)
    ].copy()   # .copy() prevents SettingWithCopyWarning on later assignments

    print(f"  Ratings after filter : {len(ratings_df):,}")
    print(f"  Users  after filter  : {ratings_df['userId'].nunique():,}")
    print(f"  Movies after filter  : {ratings_df['movieId'].nunique():,}")

    # ── [3/6] Build sparse user-item matrix ───────────────────────────────────
    print("\n[3/6] Building sparse user-item matrix...")
    matrix, user_index, movie_index, user_means = build_sparse_matrix(ratings_df)

    # Compute sparsity from sparse matrix shape + nnz (no dense conversion)
    n_users, n_movies = matrix.shape
    density = matrix.nnz / (n_users * n_movies)
    print(f"  Matrix shape  : {n_users:,} users × {n_movies:,} movies")
    print(f"  Matrix density: {density:.4%}")
    # Note: we no longer call build_user_item_matrix() here — that builds a
    # dense DataFrame which is fine for 100K but would OOM on 25M (10 B cells).

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(
        {"matrix": matrix, "user_index": user_index, "movie_index": movie_index},
        os.path.join(DATA_PROCESSED_DIR, "user_item_matrix.pkl")
    )
    print(f"  Saved user_item_matrix.pkl")

    # ── [4/6] Train SVD ───────────────────────────────────────────────────────
    print("\n[4/6] Training SVD (k=100 latent factors)...")
    print("  (On 25M this takes 2–5 minutes — normal, no need to interrupt)")
    model = train_svd(ratings_df, k=100, test_size=0.2)

    # ── [5/6] Evaluate ────────────────────────────────────────────────────────
    print("\n[5/6] Evaluating...")
    metrics = evaluate_model(model, ratings_df)

    # ── [6/6] Save ────────────────────────────────────────────────────────────
    print("\n[6/6] Saving model...")
    save_model(model, MODEL_PATH)

    # ── Demo recommendations ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO RECOMMENDATIONS")
    print("=" * 60)

    # Only demo users that exist in the filtered dataset
    available_users = ratings_df["userId"].unique()
    demo_candidates = [1, 42, 100]
    demo_users = [u for u in demo_candidates if u in available_users]
    if not demo_users:
        demo_users = list(available_users[:3])   # fallback: first 3 users

    for demo_user in demo_users:
        print(f"\nTop 5 for User {demo_user}:")
        user_rated = (
            ratings_df[ratings_df["userId"] == demo_user]
            .merge(movies_df, on="movieId")
            .sort_values("rating", ascending=False)
            .head(3)
        )
        print("  Liked:")
        for _, r in user_rated.iterrows():
            print(f"    {r['rating']}★  {r['title']}")

        recs = get_recommendations(demo_user, model, ratings_df, movies_df, n=5)
        print("  Recommended:")
        for _, r in recs.iterrows():
            print(f"    [{r['predicted_rating']:.2f}★]  {r['title']}")

    print("\n✅ Phase 3 complete.")
    return metrics


if __name__ == "__main__":
    main()