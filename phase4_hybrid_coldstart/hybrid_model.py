"""
Phase 4 — Hybrid Model
Blends content-based (Phase 2) and collaborative filtering (Phase 3) scores
into a unified hybrid recommendation.
"""
import pandas as pd
import numpy as np
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
    Sigmoid normalisation centred on the mean.
    Compresses outliers — never hard-clamps to exactly 0 or 1.
    """
    if scores.max() == scores.min():
        return np.full_like(scores, 0.5, dtype=float)

    mean = scores.mean()
    std  = scores.std()

    if std == 0:
        return np.full_like(scores, 0.5, dtype=float)

    z = (scores - mean) / (std * 2)   # *2 widens the spread
    return 1 / (1 + np.exp(-z))


# ────────────────────────────────────────────────────────────────────────────
#  Function 2 — get_content_candidates
# ────────────────────────────────────────────────────────────────────────────

def get_content_candidates(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_candidates: int = 200
) -> dict:
    """
    Generate content-based candidate scores for a user.
    Seeds from the user's top-10 rated movies, weighted by their rating.
    """
    from phase2_content_based.recommend import _recommender
    _recommender._load()
    internal_movies = _recommender._movies

    print(f"  [DEBUG] internal_movies shape : {internal_movies.shape}")
    print(f"  [DEBUG] internal_movies cols  : {list(internal_movies.columns)}")

    user_ratings = ratings_df[ratings_df["userId"] == user_id].sort_values(
        "rating", ascending=False
    )
    if user_ratings.empty:
        print(f"  [DEBUG] No ratings found for user {user_id}")
        return {}

    seed_movies = user_ratings.head(10)
    print(f"  [DEBUG] Seed movieIds for user {user_id}: {seed_movies['movieId'].tolist()}")

    score_collector: dict[int, list[float]] = {}

    for _, seed_row in seed_movies.iterrows():
        seed_movie_id = int(seed_row["movieId"])
        seed_rating   = float(seed_row["rating"])
        weight        = seed_rating / 5.0          # normalise rating to [0, 1]

        title_rows = internal_movies[internal_movies["movieId"] == seed_movie_id]["title"]
        if title_rows.empty:
            print(f"  [DEBUG] movieId {seed_movie_id} not found — skipping")
            continue

        seed_title = title_rows.iloc[0]
        print(f"  [DEBUG] Seed title resolved: '{seed_title}' (weight={weight:.2f})")

        try:
            similar_df = get_similar_movies(seed_title, n=n_candidates)
            print(f"  [DEBUG] get_similar_movies returned {len(similar_df)} rows")
        except Exception as e:
            print(f"  [DEBUG] get_similar_movies FAILED for '{seed_title}': {type(e).__name__}: {e}")
            continue

        for _, row in similar_df.iterrows():
            mid   = int(row["movieId"])
            score = float(row["similarity_score"]) * weight   # weight by seed rating
            if mid not in score_collector:
                score_collector[mid] = []
            score_collector[mid].append(score)

    print(f"  [DEBUG] score_collector size before rated-exclusion: {len(score_collector)}")

    rated_ids = set(int(x) for x in ratings_df[ratings_df["userId"] == user_id]["movieId"])
    candidates: dict[int, float] = {
        mid: float(np.mean(scores))
        for mid, scores in score_collector.items()
        if mid not in rated_ids
    }

    print(f"  [DEBUG] content_candidates size after rated-exclusion: {len(candidates)}")
    return candidates


# ────────────────────────────────────────────────────────────────────────────
#  Function 3 — get_cf_candidates
# ────────────────────────────────────────────────────────────────────────────

def get_cf_candidates(
    user_id: int,
    svd_model: dict,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_candidates: int = 200
) -> dict:
    """
    Generate collaborative filtering candidate scores for a user via SVD.
    """
    try:
        recs_df = get_recommendations(user_id, svd_model, ratings_df, movies_df, n=n_candidates)
        result = {
            int(row["movieId"]): float(row["predicted_rating"])
            for _, row in recs_df.iterrows()
        }
        print(f"  [DEBUG] cf_candidates size: {len(result)}")
        return result
    except Exception as e:
        print(f"  [Warning] CF candidates failed for user {user_id}: {type(e).__name__}: {e}")
        return {}


# ────────────────────────────────────────────────────────────────────────────
#  Function 4 — blend_scores
# ────────────────────────────────────────────────────────────────────────────

def blend_scores(
    content_candidates: dict,
    cf_candidates: dict,
    alpha: float = 0.5
) -> list[tuple[int, float]]:

    all_ids = list(
        set(content_candidates.keys()) |
        set(cf_candidates.keys())
    )

    if not all_ids:
        return []

    overlap = set(content_candidates.keys()) & set(cf_candidates.keys())
    print(f"\n[DEBUG] overlap count: {len(overlap)}")

    content_scores = np.array([content_candidates.get(mid, 0.0) for mid in all_ids])
    cf_scores      = np.array([cf_candidates.get(mid, 0.0)      for mid in all_ids])

    content_norm = normalise_scores(content_scores)
    cf_norm      = normalise_scores(cf_scores)

    hybrid_scores = []

    for i, mid in enumerate(all_ids):
        c_score  = content_norm[i]
        cf_score = cf_norm[i]

        hybrid = alpha * c_score + (1 - alpha) * cf_score

        # penalise one-sided candidates slightly
        if c_score == 0 or cf_score == 0:
            hybrid *= 0.90

        hybrid = min(hybrid, 1.0)
        hybrid_scores.append((mid, hybrid))

    return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)


# ────────────────────────────────────────────────────────────────────────────
#  Function 5 — get_hybrid_recommendations  (main entry point)
# ────────────────────────────────────────────────────────────────────────────

def get_hybrid_recommendations(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    svd_model: dict,
    n: int = 10,
    alpha: float = 0.5
) -> list[dict]:

    from phase2_content_based.recommend import _recommender
    _recommender._load()
    internal_movies = _recommender._movies

    content_candidates = get_content_candidates(user_id, ratings_df, movies_df, n_candidates=200)
    cf_candidates      = get_cf_candidates(user_id, svd_model, ratings_df, movies_df, n_candidates=200)

    if not content_candidates and not cf_candidates:
        print("  [Warning] Both candidate sets empty — falling back to popularity")
        from phase4_hybrid_coldstart.cold_start import get_popular_recommendations
        return get_popular_recommendations(ratings_df, movies_df, n=n)

    blended = blend_scores(content_candidates, cf_candidates, alpha=alpha)
    print(f"  [DEBUG] blended list length: {len(blended)}")
    if blended:
        print(f"  [DEBUG] top 3 blended: {blended[:3]}")

    # Build normalised score dicts for display
    all_ids       = list(set(content_candidates.keys()) | set(cf_candidates.keys()))
    content_norm  = normalise_scores(np.array([content_candidates.get(m, 0.0) for m in all_ids]))
    cf_norm       = normalise_scores(np.array([cf_candidates.get(m, 0.0)      for m in all_ids]))
    content_norm_dict = {mid: float(content_norm[i]) for i, mid in enumerate(all_ids)}
    cf_norm_dict      = {mid: float(cf_norm[i])      for i, mid in enumerate(all_ids)}

    results = []
    for movie_id, hybrid_score in blended:
        if len(results) >= n:
            break

        movie_rows = internal_movies[internal_movies["movieId"] == movie_id]
        if movie_rows.empty:
            continue

        movie_row = movie_rows.iloc[0]
        results.append({
            "movie_id":      int(movie_id),
            "title":         str(movie_row["title"]),
            "genres":        str(movie_row["genres"]),
            "content_score": content_norm_dict.get(movie_id, 0.0),
            "cf_score":      cf_norm_dict.get(movie_id, 0.0),
            "hybrid_score":  float(hybrid_score),
        })

    print(f"  [DEBUG] final results count: {len(results)}")
    return results


# ────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings   = pd.read_csv("data/raw/ratings.csv")
    movies    = pd.read_csv("data/raw/movies.csv")
    svd_model = load_model()

    print("=== Hybrid recommendations for user 42 ===")
    recs = get_hybrid_recommendations(42, ratings, movies, svd_model, n=10)
    for r in recs:
        print(f"  {r['title']}  hybrid={r['hybrid_score']:.4f}  "
              f"content={r['content_score']:.4f}  cf={r['cf_score']:.4f}")

    print("\n=== Hybrid recommendations for user 1 ===")
    recs = get_hybrid_recommendations(1, ratings, movies, svd_model, n=10)
    for r in recs:
        print(f"  {r['title']}  hybrid={r['hybrid_score']:.4f}  "
              f"content={r['content_score']:.4f}  cf={r['cf_score']:.4f}")