"""
Phase 6 — Recommend Routes

Endpoints
---------
GET /recommend/popular               → popularity-based (cold-start fallback)
GET /recommend/movie/{title}         → content-based similarity
GET /recommend/user/{user_id}        → routed: popularity / content / hybrid
GET /search?q=                       → fuzzy title autocomplete (for Phase 7 UI)

Design notes
------------
- All heavy objects are injected via Depends() — no I/O in request handlers.
- route_recommendation() handles 3-branch routing; this file only deals with
  the "use_hybrid" sentinel for Branch 3.
- The `explanation` field in every MovieRecommendation is deliberately None
  throughout Phase 6. Phase 7 will populate it via the Anthropic API.
"""
import time
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from phase6_api.dependencies import get_ratings, get_movies, get_svd
from phase6_api.schemas import (
    MovieRecommendation,
    RecommendationResponse,
    SearchResult,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ── Helper mappers ─────────────────────────────────────────────────────────────

def _map_popularity_rec(r: dict) -> MovieRecommendation:
    """Map a cold_start.get_popular_recommendations() dict → schema."""
    return MovieRecommendation(
        movie_id      = r["movie_id"],
        title         = r["title"],
        genres        = r["genres"],
        weighted_score= r.get("weighted_score"),   # may be None for content branch
    )


def _map_hybrid_rec(r: dict) -> MovieRecommendation:
    """Map a hybrid_model.get_hybrid_recommendations() dict → schema."""
    return MovieRecommendation(
        movie_id      = r["movie_id"],
        title         = r["title"],
        genres        = r["genres"],
        hybrid_score  = r.get("hybrid_score"),
        content_score = r.get("content_score"),
        cf_score      = r.get("cf_score"),
    )


# ── GET /recommend/popular ────────────────────────────────────────────────────

@router.get("/popular", response_model=RecommendationResponse)
def popular(
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    ratings = Depends(get_ratings),
    movies  = Depends(get_movies),
) -> RecommendationResponse:
    """
    Popularity-based recommendations using the IMDb weighted rating formula.
    No user_id required — suitable for anonymous / new users.
    """
    t0 = time.perf_counter()

    from phase4_hybrid_coldstart.cold_start import get_popular_recommendations
    recs_raw = get_popular_recommendations(ratings, movies, n=n)

    return RecommendationResponse(
        model_used      = "popularity",
        recommendations = [_map_popularity_rec(r) for r in recs_raw],
        latency_ms      = (time.perf_counter() - t0) * 1000,
    )


# ── GET /recommend/movie/{title} ───────────────────────────────────────────────

@router.get("/movie/{title}", response_model=RecommendationResponse)
def by_movie(
    title: str,
    n: int = Query(default=10, ge=1, le=100, description="Number of similar movies"),
) -> RecommendationResponse:
    """
    Content-based recommendations: find movies similar to a given title.
    Title matching is case-insensitive and supports partial matches
    (e.g. "inception" matches "Inception (2010)").
    """
    t0 = time.perf_counter()

    from phase2_content_based.recommend import get_similar_movies
    try:
        similar_df = get_similar_movies(title, n=n)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{title}' not found in catalogue. "
                   f"Try GET /search?q={title} to find the exact title.",
        )

    recs = [
        MovieRecommendation(
            movie_id      = int(row["movieId"]),
            title         = row["title"],
            genres        = row["genres"],
            content_score = float(row["similarity_score"]),
        )
        for _, row in similar_df.iterrows()
    ]

    return RecommendationResponse(
        query_title     = title,
        model_used      = "content",
        recommendations = recs,
        latency_ms      = (time.perf_counter() - t0) * 1000,
    )


# ── GET /recommend/user/{user_id} ──────────────────────────────────────────────

@router.get("/user/{user_id}", response_model=RecommendationResponse)
def by_user(
    user_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    ratings = Depends(get_ratings),
    movies  = Depends(get_movies),
    svd     = Depends(get_svd),
) -> RecommendationResponse:
    """
    Personalised recommendations for a user.

    Routing logic (delegated to cold_start.route_recommendation):
    - 0 ratings   → popularity-based fallback
    - 1–4 ratings → content-based (seeds from highest-rated movie)
    - 5+ ratings  → hybrid (content + SVD collaborative filtering)

    Unknown user_id returns popularity-based recommendations, not a 404,
    because a new user is a valid use case.
    """
    t0 = time.perf_counter()

    if svd is None:
        raise HTTPException(
            status_code=503,
            detail="SVD model not loaded. Check server logs for startup errors.",
        )

    from phase4_hybrid_coldstart.cold_start import route_recommendation

    result, model_used = route_recommendation(user_id, ratings, movies, n=n)

    # Branch 3 sentinel: 5+ ratings → run the full hybrid model
    if result == "use_hybrid":
        from phase4_hybrid_coldstart.hybrid_model import get_hybrid_recommendations
        recs_raw = get_hybrid_recommendations(
            user_id, ratings, movies, svd, n=n
        )
        recs = [_map_hybrid_rec(r) for r in recs_raw]
        model_used = "hybrid"
    else:
        # Branches 1 and 2: result is already a list of dicts
        recs = [_map_popularity_rec(r) for r in result]

    log.info(
        "user/%d → %s | %d recs | %.1f ms",
        user_id, model_used, len(recs), (time.perf_counter() - t0) * 1000,
    )

    return RecommendationResponse(
        user_id         = user_id,
        model_used      = model_used,
        recommendations = recs,
        latency_ms      = (time.perf_counter() - t0) * 1000,
    )


# ── GET /search ────────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResult)
def search(
    q: str = Query(..., min_length=1, description="Partial movie title to search"),
    top_k: int = Query(default=5, ge=1, le=20, description="Max results"),
) -> SearchResult:
    """
    Fuzzy title autocomplete — returns up to top_k titles containing `q`.
    Wired up here so Phase 7 Streamlit needs no API changes.

    Example: GET /recommend/search?q=inception
    Returns: {"query": "inception", "titles": ["Inception (2010)", ...]}
    """
    from phase2_content_based.recommend import _recommender
    titles = _recommender.fuzzy_search(q, top_k=top_k)
    return SearchResult(query=q, titles=titles)