import os
import logging
import threading
import pandas as pd
import numpy as np
import joblib

log = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(__file__)
MODELS_DIR = os.path.join(_HERE, "..", "models")
DATA_DIR   = os.path.join(_HERE, "..", "data", "processed")
 
MOVIES_CLEAN_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
COSINE_PATH       = os.path.join(MODELS_DIR, "cosine_sim_matrix.pkl")
 
 
class ContentRecommender:
    """
    Lazy-loading content-based recommender.
    Loads artefacts once on first use.
    """
 
    def __init__(self):
        self._movies     = None
        self._cosine_sim = None
        self._idx_map    = None   # title → integer index
        self._lock       = threading.Lock()

    def _load(self):
        with self._lock:
            if self._movies is not None:
                return  # already loaded

            if not os.path.exists(COSINE_PATH):
                raise FileNotFoundError(
                    "Model artefacts not found. "
                    "Run `python phase2_content_based/content_model.py` first."
                )

            self._movies     = pd.read_csv(MOVIES_CLEAN_PATH)
            self._cosine_sim = joblib.load(COSINE_PATH)

            # Build a case-insensitive title → row-index map
            self._idx_map = {
                title.lower(): idx
                for idx, title in enumerate(self._movies["title"])
            }
 
    # ── Public API ─────────────────────────────────────────────────────────
 
    def get_similar_movies(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Return the top-n most similar movies to `title`.
 
        Parameters
        ----------
        title : str   Movie title (case-insensitive, partial match supported)
        n     : int   Number of recommendations to return
 
        Returns
        -------
        pd.DataFrame with columns: title, genres, similarity_score
        """
        self._load()
 
        # ── 1. Fuzzy title match ──────────────────────────────────────────
        query    = title.lower()
        matches  = [t for t in self._idx_map if query in t]
 
        if not matches:
           raise ValueError(f"Movie '{title}' not found in catalogue.") 
 
        # Prefer exact match, then shortest match (most specific)
        best_match = min(matches, key=lambda t: (t != query, len(t)))
        idx        = self._idx_map[best_match]
 
        matched_title = self._movies.iloc[idx]["title"]
        log.debug("Matched: %s", matched_title)
 
        # ── 2. Score all movies ───────────────────────────────────────────
        sim_scores = list(enumerate(self._cosine_sim[idx]))
 
        # Sort by score descending, skip the query movie itself (score = 1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]
 
        # ── 3. Build result DataFrame ─────────────────────────────────────
        indices = [i for i, _ in sim_scores]
        scores  = [round(float(s), 4) for _, s in sim_scores]
 
        result = self._movies.iloc[indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = scores
        result = result.reset_index(drop=True)
 
        return result
 
    def fuzzy_search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Return up to top_k movie titles that contain `query` (case-insensitive).
        Useful for building autocomplete.
        """
        self._load()
        q       = query.lower()
        matches = [t for t in self._idx_map if q in t]
        matches.sort(key=len)  # shorter = more exact
        # Return original-case titles
        return [self._movies.iloc[self._idx_map[m]]["title"] for m in matches[:top_k]]
 
 
# ── Module-level convenience function ────────────────────────────────────────
_recommender = ContentRecommender()
 
 
def get_similar_movies(title: str, n: int = 10) -> pd.DataFrame:
    """Convenience wrapper around ContentRecommender.get_similar_movies."""
    return _recommender.get_similar_movies(title, n)
 
 
# ── Manual test ──────────────────────────────────────────────────────────────
def _run_tests():
    test_cases = [
        ("Inception",      10),
        ("Toy Story",      5),
        ("The Dark Knight", 8),
        ("Pulp Fiction",   5),
    ]
 
    for query, n in test_cases:
        print(f"\n{'─'*55}")
        print(f"  Query: '{query}'  →  top {n} similar movies")
        print(f"{'─'*55}")
        try:
            results = get_similar_movies(query, n=n)
            print(results[["title", "genres", "similarity_score"]].to_string())
        except ValueError as e:
            print(f"  ERROR: {e}")
 
 
if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 2 — Content-Based Recommender: Tests")
    print("=" * 55)
    _run_tests()