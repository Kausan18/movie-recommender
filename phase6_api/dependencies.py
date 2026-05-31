"""
Phase 6 — Dependency Injection
Loads every heavy artefact ONCE during the FastAPI lifespan.
Request handlers receive them via FastAPI's Depends() mechanism.

Never import this module and call load_all_models() more than once —
the module-level singletons are deliberately overwritten by that call.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

log = logging.getLogger(__name__)

# ── Module-level singletons — None until load_all_models() is called ─────────
_ratings_df:    pd.DataFrame | None = None
_movies_df:     pd.DataFrame | None = None
_svd_model:     dict | None         = None
_kmeans:        object | None       = None
_kmeans_scaler: object | None       = None


def load_all_models() -> None:
    """
    Called exactly once from the FastAPI lifespan context manager.
    Populates all module-level singletons so request handlers pay zero
    I/O cost at runtime.

    Load order matters:
      1. DataFrames first (needed by hybrid / cold-start logic)
      2. SVD model (Phase 3 — scipy truncated SVD, NOT scikit-surprise)
      3. K-Means artefacts (optional — API degrades gracefully if absent)
      4. Content recommender warm-up (triggers lazy _load() so the first
         request doesn't pay the cosine-sim load cost)
    """
    global _ratings_df, _movies_df, _svd_model, _kmeans, _kmeans_scaler

    # 1 ── Raw data ────────────────────────────────────────────────────────
    log.info("Loading ratings and movies DataFrames...")
    _ratings_df = pd.read_csv("data/raw/ratings.csv")
    _movies_df  = pd.read_csv("data/raw/movies.csv")
    log.info("  ratings: %d rows | movies: %d rows", len(_ratings_df), len(_movies_df))

    # 2 ── SVD model (Phase 3) ─────────────────────────────────────────────
    log.info("Loading SVD model...")
    from phase3_collaborative_filtering.svd_model import load_model
    _svd_model = load_model()   # loads models/svd_model.pkl
    log.info("  SVD model loaded: %s", type(_svd_model))

    # 3 ── K-Means artefacts (Phase 4) — optional ──────────────────────────
    kmeans_path  = Path("models/kmeans_model.pkl")
    scaler_path  = Path("models/kmeans_scaler.pkl")
    if kmeans_path.exists() and scaler_path.exists():
        log.info("Loading K-Means model and scaler...")
        _kmeans        = joblib.load(kmeans_path)
        _kmeans_scaler = joblib.load(scaler_path)
        log.info("  K-Means loaded successfully")
    else:
        log.warning(
            "K-Means artefacts not found at %s / %s — clustering disabled. "
            "Run the Phase 4 smoke test to generate them.",
            kmeans_path, scaler_path
        )

    # 4 ── Content recommender warm-up (Phase 2) ───────────────────────────
    log.info("Warming up content recommender (loading cosine_sim matrix)...")
    from phase2_content_based.recommend import _recommender
    _recommender._load()
    log.info("  Content recommender ready")

    log.info("✅ All models loaded successfully")


# ── FastAPI Depends() injectors ───────────────────────────────────────────────
# These are thin functions so FastAPI can inject them via Depends().
# They deliberately return the module-level singletons — no copying.

def get_ratings() -> pd.DataFrame:
    return _ratings_df

def get_movies() -> pd.DataFrame:
    return _movies_df

def get_svd() -> dict:
    return _svd_model

def get_kmeans() -> tuple:
    """Returns (kmeans_model, scaler) — either or both may be None."""
    return (_kmeans, _kmeans_scaler)