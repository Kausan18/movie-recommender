"""
Phase 6 — FastAPI Application Entry Point
Run from project root:
    uvicorn phase6_api.main:app --reload --port 8000
or from inside phase6_api/:
    uvicorn main:app --reload --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from phase6_api.dependencies import load_all_models
from phase6_api.routes import health, recommend

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs before the first request is accepted (startup) and after the last
    response is sent (shutdown).  All ML artefacts are loaded here so request
    handlers pay zero I/O cost.
    """
    log.info("Starting up Movie Recommender API...")
    load_all_models()
    log.info("🎬 API ready — visit http://localhost:8000/docs for Swagger UI")
    yield
    log.info("👋 Shutting down Movie Recommender API")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommender API",
    version="1.0.0",
    description=(
        "Content-based, collaborative filtering, and hybrid movie recommendations "
        "built on MovieLens 100K. Phase 7 will add LLM-generated explanations."
    ),
    lifespan=lifespan,
)

app.include_router(health.router,    tags=["health"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])