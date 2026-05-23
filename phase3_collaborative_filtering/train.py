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


def main():
    print("=" * 60)
    print("PHASE 3 — COLLABORATIVE FILTERING TRAINING")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    ratings_df = pd.read_csv("data/raw/ratings.csv")
    movies_df = pd.read_csv("data/raw/movies.csv")
    print(f"  Ratings : {len(ratings_df):,}")
    print(f"  Users   : {ratings_df['userId'].nunique()}")
    print(f"  Movies  : {ratings_df['movieId'].nunique()}")

    print("\n[2/5] Building sparse user-item matrix...")
    matrix, user_index, movie_index, user_means = build_sparse_matrix(ratings_df)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(
        {"matrix": matrix, "user_index": user_index, "movie_index": movie_index},
        "data/processed/user_item_matrix.pkl"
    )
    dense_matrix = build_user_item_matrix(ratings_df)
    density = 1 - (dense_matrix == 0).values.sum() / (dense_matrix.shape[0] * dense_matrix.shape[1])
    print(f"  Matrix density: {density:.2%}")

    print("\n[3/5] Training SVD (k=100 latent factors)...")
    model = train_svd(ratings_df, k=100, test_size=0.2)

    print("\n[4/5] Evaluating...")
    metrics = evaluate_model(model, ratings_df)

    print("\n[5/5] Saving model...")
    save_model(model)

    print("\n" + "=" * 60)
    print("DEMO RECOMMENDATIONS")
    print("=" * 60)
    for demo_user in [1, 42, 100]:
        print(f"\nTop 5 for User {demo_user}:")
        user_rated = (
            ratings_df[ratings_df['userId'] == demo_user]
            .merge(movies_df, on='movieId')
            .sort_values('rating', ascending=False)
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
