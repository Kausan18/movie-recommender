"""
Phase 6 — Pydantic Schemas
All request/response models for the Movie Recommender API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str

    # Score fields — only one or two will be non-null per endpoint
    predicted_rating: Optional[float] = None   # Raw CF predicted rating (0–5 scale)
    hybrid_score:     Optional[float] = None   # Blended hybrid score (0–1)
    content_score:    Optional[float] = None   # Normalised content-based score (0–1)
    cf_score:         Optional[float] = None   # Normalised CF score (0–1)
    weighted_score:   Optional[float] = None   # IMDb-style popularity score

    # Phase 7 LLM layer — null throughout Phase 6
    explanation: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id:     Optional[int] = None
    query_title: Optional[str] = None
    model_used:  str                    # "content" | "collaborative" | "hybrid" | "popularity"
    recommendations: List[MovieRecommendation]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict                 # {"svd": True, "cosine_sim": True, "kmeans": True}


class SearchResult(BaseModel):
    titles: List[str]
    query:  str