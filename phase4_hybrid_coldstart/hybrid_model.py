"""
Phase 4 — Hybrid Model
Blends content-based (Phase 2) and collaborative filtering (Phase 3) scores
into a unified hybrid recommendation.

Adapted to work with the actual codebase APIs:
  - Phase 2: get_similar_movies(title, n) returns DataFrame with similarity_score
  - Phase 3: get_recommendations() returns a DataFrame (not list[dict])
  - user_means is a numpy array indexed by row, not a dict
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase2_content_based.recommend import get_similar_movies
from phase3_collaborative_filtering.svd_model import get_recommendations, load_model


# ────────────────────────────────────────────────────────────────────────────
#  Function 1 — normalise_scores
# ────────────────────────────────────────────────────────────────────────────

def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalisation to [0, 1].
    If all scores are identical (max == min), returns an array of 0.5.
    """
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - min_val) / (max_val - min_val)


# ────────────────────────────────────────────────────────────────────────────
#  Function 2 — get_content_candidates
# ────────────────────────────────────────────────────────────────────────────

def get_content_candidates(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_candidates: int = 50
) -> dict:
    # Use Phase 2's internal movies_clean.csv for title lookups,
    # NOT the raw movies_df passed in (they have different movieId sets)
    from phase2_content_based.recommend import _recommender
    _recommender._load()  # trigger lazy load if not already loaded
    internal_movies = _recommender._movies  # this is movies_clean.csv

    # Get user's ratings
    user_ratings = ratings_df[ratings_df["userId"] == user_id].sort_values(
        "rating", ascending=False
    )
    if user_ratings.empty:
        return {}

    # Look up seed titles using movies_clean.csv (same source Phase 2 uses)
    seed_movies = user_ratings.head(3)
    score_collector = {}

    for _, seed_row in seed_movies.iterrows():
        seed_movie_id = int(seed_row["movieId"])

        # Look up in internal_movies, not movies_df
        title_rows = internal_movies[internal_movies["movieId"] == seed_movie_id]["title"]
        if title_rows.empty:
            continue

        seed_title = title_rows.iloc[0]

        try:
            similar_df = get_similar_movies(seed_title, n=n_candidates)
        except (ValueError, FileNotFoundError):
            continue

        for _, row in similar_df.iterrows():
            mid = int(row["movieId"])
            score = float(row["similarity_score"])
            if mid not in score_collector:
                score_collector[mid] = []
            score_collector[mid].append(score)

    # Exclude movies the user already rated
    rated_ids = set(int(x) for x in ratings_df[ratings_df["userId"] == user_id]["movieId"])
    candidates = {}
    for mid, scores in score_collector.items():
        if mid not in rated_ids:
            candidates[mid] = float(np.mean(scores))

    return candidates


# ────────────────────────────────────────────────────────────────────────────
#  Function 3 — get_cf_candidates
# ────────────────────────────────────────────────────────────────────────────

def get_cf_candidates(
    user_id: int,
    svd_model: dict,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_candidates: int = 50
) -> dict:
    """
    Generate collaborative filtering candidate scores for a user via SVD.

    NOTE: The actual get_recommendations() returns a DataFrame with columns
    [movieId, title, genres, predicted_rating], not a list of dicts.

    Returns:
        dict of {movie_id: predicted_rating}
    """
    try:
        # get_recommendations returns a DataFrame, not list[dict]
        recs_df = get_recommendations(user_id, svd_model, ratings_df, movies_df, n=n_candidates)
        return {
            int(row["movieId"]): float(row["predicted_rating"])
            for _, row in recs_df.iterrows()
        }
    except Exception as e:
        print(f"  [Warning] CF candidates failed for user {user_id}: {e}")
        return {}


# ────────────────────────────────────────────────────────────────────────────
#  Function 4 — blend_scores
# ────────────────────────────────────────────────────────────────────────────

def blend_scores(
    content_candidates: dict,
    cf_candidates: dict,
    alpha: float = 0.4
) -> list[tuple[int, float]]:
    """
    Blend content and CF scores with min-max normalisation.

    hybrid = alpha * content_norm + (1 - alpha) * cf_norm

    Args:
        content_candidates: {movie_id: raw_content_score}
        cf_candidates:      {movie_id: raw_cf_score}
        alpha:              weight for content scores (default 0.4)

    Returns:
        Sorted list of (movie_id, hybrid_score) tuples, descending by score.
    """
    # Union of all movie IDs
    all_ids = list(set(content_candidates.keys()) | set(cf_candidates.keys()))

    if not all_ids:
        return []

    # Build parallel arrays
    content_scores = np.array([content_candidates.get(mid, 0.0) for mid in all_ids])
    cf_scores = np.array([cf_candidates.get(mid, 0.0) for mid in all_ids])

    # Normalise each array separately
    content_norm = normalise_scores(content_scores)
    cf_norm = normalise_scores(cf_scores)

    # Blend
    hybrid = alpha * content_norm + (1 - alpha) * cf_norm

    # Sort descending by hybrid score
    return sorted(zip(all_ids, hybrid), key=lambda x: x[1], reverse=True)


# ────────────────────────────────────────────────────────────────────────────
#  Function 5 — get_hybrid_recommendations (main entry point)
# ────────────────────────────────────────────────────────────────────────────

def get_hybrid_recommendations(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    svd_model: dict,
    n: int = 10,
    alpha: float = 0.4
) -> list[dict]:
    from phase2_content_based.recommend import _recommender
    _recommender._load()
    internal_movies = _recommender._movies  # use this for metadata lookups

    content_candidates = get_content_candidates(user_id, ratings_df, movies_df, n_candidates=50)
    cf_candidates = get_cf_candidates(user_id, svd_model, ratings_df, movies_df, n_candidates=50)

    if not content_candidates and not cf_candidates:
        from phase4_hybrid_coldstart.cold_start import get_popular_recommendations
        return get_popular_recommendations(ratings_df, movies_df, n=n)

    blended = blend_scores(content_candidates, cf_candidates, alpha=alpha)

    results = []
    for movie_id, hybrid_score in blended:
        if len(results) >= n:
            break

        # Use internal_movies for lookup, not movies_df
        movie_rows = internal_movies[internal_movies["movieId"] == movie_id]
        if movie_rows.empty:
            continue

        movie_row = movie_rows.iloc[0]
        results.append({
            "movie_id": int(movie_id),
            "title": str(movie_row["title"]),
            "genres": str(movie_row["genres"]),
            "content_score": content_candidates.get(movie_id, 0.0),
            "cf_score": cf_candidates.get(movie_id, 0.0),
            "hybrid_score": float(hybrid_score),
        })

    return results


# ────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    movies = pd.read_csv("data/raw/movies.csv")
    svd_model = load_model()
    #cosine_sim = joblib.load("models/cosine_sim_matrix.pkl")

    print("=== Hybrid recommendations for user 42 ===")
    recs = get_hybrid_recommendations(42, ratings, movies, svd_model, n=10)
    for r in recs:
        print(f"  {r['title']}  hybrid={r['hybrid_score']:.4f}  "
              f"content={r['content_score']:.4f}  cf={r['cf_score']:.4f}")

    print("\n=== Hybrid recommendations for user 1 ===")
    recs = get_hybrid_recommendations(1, ratings, movies, svd_model, n=10)
    for r in recs:
        print(f"  {r['title']}")
