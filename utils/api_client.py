"""
api_client.py — All HTTP calls to the Phase 6 FastAPI backend go through here.
No page or component should ever call `requests` directly.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TMDB_KEY = os.getenv("TMDB_KEY") or os.getenv("TMDB_API_KEY")
TMDB_IMG = "https://image.tmdb.org/t/p/w300"


def _get(endpoint: str, params: dict = None) -> dict | None:
    """Internal GET helper. Returns parsed JSON or None on error."""
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ── Public API wrappers ────────────────────────────────────────────────────

def get_health() -> dict:
    """GET /health — returns status + models_loaded dict."""
    return _get("/health") or {}


def get_popular(n: int = 10) -> list[dict]:
    """GET /recommend/popular — returns list of recommendation dicts."""
    data = _get("/recommend/popular", params={"n": n})
    return data.get("recommendations", []) if data else []


def get_by_movie(title: str, n: int = 10) -> dict:
    """
    GET /recommend/movie/{title}
    Returns full response dict: recommendations, model_used, latency_ms.
    """
    encoded = requests.utils.quote(title, safe="")
    return _get(f"/recommend/movie/{encoded}", params={"n": n}) or {}


def get_by_user(user_id: int, n: int = 10) -> dict:
    """
    GET /recommend/user/{user_id}
    Returns full response dict: recommendations, model_used, latency_ms.
    Handles the hybrid/popularity/content routing automatically via Phase 6 logic.
    """
    return _get(f"/recommend/user/{user_id}", params={"n": n}) or {}


def search_titles(q: str, top_k: int = 7) -> list[str]:
    """GET /recommend/search?q= — powers autocomplete on the home page."""
    data = _get("/recommend/search", params={"q": q, "top_k": top_k})
    if not data:
        return []
    # Response may be a list of SearchResult objects or a dict with "titles"
    if isinstance(data, list):
        return [item.get("title", "") for item in data if item.get("title")]
    return data.get("titles", [])


def get_poster_url(tmdb_id: int | float | None) -> str | None:
    """
    Fetch poster path from TMDB API.
    Returns full image URL or None (never raises — cards degrade gracefully).
    """
    if not tmdb_id or not TMDB_KEY:
        return None
    try:
        tmdb_id = int(tmdb_id)
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_KEY},
            timeout=5,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"{TMDB_IMG}{path}" if path else None
    except Exception:
        return None


def is_backend_alive() -> bool:
    """Quick liveness check — used to show a warning banner in the UI."""
    health = get_health()
    return bool(health and health.get("status") == "ok")