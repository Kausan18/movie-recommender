"""
Phase 4 — Cold Start Handler
Provides popularity-based fallback and routing logic for new/sparse users.

Adapted to work with the actual Phase 2 recommend.py API:
  - get_similar_movies(title, n=10) — loads its own cosine_sim internally,
    uses movies_clean.csv, returns DataFrame with movieId, title, genres, similarity_score.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Make project root importable so Phase 2 functions can be called
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ────────────────────────────────────────────────────────────────────────────
#  Function 1 — compute_popularity_scores
# ────────────────────────────────────────────────────────────────────────────

def compute_popularity_scores(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    min_vote_percentile: int = 50
) -> pd.DataFrame:
    """
    Compute IMDb-style weighted popularity scores for all movies.

    Uses the Bayesian weighted rating formula:
        weighted_score = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = movie's vote count
        R = movie's average rating
        m = minimum votes required (percentile threshold)
        C = global mean rating
    """
    # 1. Group ratings by movieId
    stats = ratings_df.groupby("movieId")["rating"].agg(
        vote_count="count",
        avg_rating="mean"
    ).reset_index()

    # 2. Global mean rating (across all individual ratings)
    C = ratings_df["rating"].mean()

    # 3. Minimum vote threshold
    m = stats["vote_count"].quantile(min_vote_percentile / 100.0)

    # 4. IMDb weighted rating formula
    v = stats["vote_count"]
    R = stats["avg_rating"]
    stats["weighted_score"] = (v / (v + m)) * R + (m / (v + m)) * C

    # 5. Merge with movies_df to attach title and genres
    popularity_df = stats.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="inner")

    # 6. Sort descending by weighted_score
    popularity_df = popularity_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)

    # 7. Select output columns
    popularity_df = popularity_df[["movieId", "title", "genres", "vote_count", "avg_rating", "weighted_score"]]

    # Save to CSV
    os.makedirs("data/processed", exist_ok=True)
    popularity_df.to_csv("data/processed/popularity_scores.csv", index=False)

    return popularity_df


# ────────────────────────────────────────────────────────────────────────────
#  Function 2 — get_popular_recommendations
# ────────────────────────────────────────────────────────────────────────────

def get_popular_recommendations(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    exclude_movie_ids: list = None
) -> list[dict]:
    """
    Return top-n popular movies as a list of dicts.
    Optionally exclude specific movie IDs (e.g. already-rated movies).
    """
    popularity_df = compute_popularity_scores(ratings_df, movies_df)

    if exclude_movie_ids is not None and len(exclude_movie_ids) > 0:
        popularity_df = popularity_df[~popularity_df["movieId"].isin(exclude_movie_ids)]

    top_n = popularity_df.head(n)

    return [
        {
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "weighted_score": float(row["weighted_score"]),
        }
        for _, row in top_n.iterrows()
    ]


# ────────────────────────────────────────────────────────────────────────────
#  Function 3 — get_user_rating_count
# ────────────────────────────────────────────────────────────────────────────

def get_user_rating_count(user_id: int, ratings_df: pd.DataFrame) -> int:
    """Return the number of ratings a user has. Returns 0 if user not found."""
    return int((ratings_df["userId"] == user_id).sum())


# ────────────────────────────────────────────────────────────────────────────
#  Function 4 — route_recommendation
# ────────────────────────────────────────────────────────────────────────────

def route_recommendation(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10
) -> tuple[list | str, str]:
    """
    Route a recommendation request based on how many ratings the user has.

    Returns:
        (recommendations, model_used) where:
        - rating_count == 0  -> popularity-based recs, "popularity"
        - 1-4 ratings        -> content-based recs, "content"
        - 5+ ratings         -> "use_hybrid" string, "hybrid"
    """
    rating_count = get_user_rating_count(user_id, ratings_df)

    # ── Branch 1: No ratings → popularity ──────────────────────────────────
    if rating_count == 0:
        recs = get_popular_recommendations(ratings_df, movies_df, n=n)
        return recs, "popularity"

    # ── Branch 2: 1-4 ratings → content-based ──────────────────────────────
    if 1 <= rating_count <= 4:
        # Import the actual Phase 2 function
        # NOTE: The actual get_similar_movies(title, n) loads its own cosine_sim
        # and movies_clean.csv internally — it does NOT accept cosine_sim or
        # movies_df as parameters.
        from phase2_content_based.recommend import get_similar_movies

        # Find the movie the user rated highest
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        best_movie_id = user_ratings.loc[user_ratings["rating"].idxmax(), "movieId"]
        best_title_rows = movies_df[movies_df["movieId"] == best_movie_id]["title"]

        if best_title_rows.empty:
            # Fallback to popularity if the movie title can't be found
            recs = get_popular_recommendations(ratings_df, movies_df, n=n)
            return recs, "popularity"

        best_title = best_title_rows.iloc[0]

        try:
            # Call with the actual API: get_similar_movies(title, n)
            similar_df = get_similar_movies(best_title, n=n)
        except (ValueError, FileNotFoundError):
            # Fallback to popularity if content model fails
            recs = get_popular_recommendations(ratings_df, movies_df, n=n)
            return recs, "popularity"

        # Convert to list of dicts matching the standard return format
        recs = []
        for _, row in similar_df.iterrows():
            recs.append({
                "movie_id": int(row["movieId"]) if "movieId" in row else 0,
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "weighted_score": None
            })
        return recs, "content"

    # ── Branch 3: 5+ ratings → hybrid ─────────────────────────────────────
    return "use_hybrid", "hybrid"


# ────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    movies = pd.read_csv("data/raw/movies.csv")

    print("=== Top 10 popular movies ===")
    top10 = get_popular_recommendations(ratings, movies, n=10)
    for m in top10:
        print(f"  {m['title']}  score={m['weighted_score']:.4f}")

    print("\n=== Routing for user 1 (has many ratings) ===")
    result, model = route_recommendation(1, ratings, movies, n=5)
    print(f"  Model used: {model}")
    if isinstance(result, list):
        for r in result:
            print(f"  {r['title']}")
    else:
        print(f"  Result: {result}")

    print("\n=== Routing for existing high-activity user (id=99999) ===")
    result, model = route_recommendation(99999, ratings, movies, n=5)
    print(f"  Model used: {model}")
    if isinstance(result, list):
        for r in result:
            print(f"  {r['title']}")
    else:
        print(f"  Result: {result}")
