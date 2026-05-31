"""
Phase 6 — Health Route
GET /health — reports API status and which models loaded successfully.
"""
from fastapi import APIRouter

from phase6_api import dependencies as deps
from phase6_api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Liveness + readiness check.
    Returns 200 with a per-model status dict so you can see at a glance
    which artefacts loaded.  Useful for debugging deployment issues.
    """
    # Content recommender is considered loaded once its cosine_sim is set
    from phase2_content_based.recommend import _recommender
    content_ready = _recommender._cosine_sim is not None

    return HealthResponse(
        status="ok",
        models_loaded={
            "svd":        deps._svd_model   is not None,
            "cosine_sim": content_ready,
            "kmeans":     deps._kmeans      is not None,
            "ratings_df": deps._ratings_df  is not None,
            "movies_df":  deps._movies_df   is not None,
        },
    )