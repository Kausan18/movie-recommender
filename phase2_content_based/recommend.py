import os
import re as _re
import logging
import threading
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import linear_kernel

log = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(__file__)
MODELS_DIR = os.path.join(_HERE, "..", "models")
DATA_DIR   = os.path.join(_HERE, "..", "data", "processed")

MOVIES_CLEAN_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")


# ── Title normalisation (for robust search) ───────────────────────────────
_YEAR_RE  = _re.compile(r'\(\d{4}\)')
_PUNCT_RE = _re.compile(r'[^a-z0-9 ]')
_WS_RE    = _re.compile(r'\s+')

def _norm(text: str) -> str:
    """Lowercase, strip year, strip punctuation, collapse whitespace."""
    t = text.lower()
    t = _YEAR_RE.sub('', t)
    t = _PUNCT_RE.sub(' ', t)
    t = _WS_RE.sub(' ', t).strip()
    return t

def _word_match(query_norm: str, title_norm: str) -> bool:
    """True if query appears in title at a word boundary (not mid-word)."""
    return (' ' + query_norm + ' ') in (' ' + title_norm + ' ')


class ContentRecommender:
    """
    Lazy-loading content-based recommender.
    Loads artefacts once on first use.
    """

    def __init__(self):
        self._movies       = None
        self._tfidf_matrix = None
        self._idx_map      = None   # raw lowercase title → integer index
        self._norm_map     = None   # normalised title    → integer index
        self._lock         = threading.Lock()

    def _load(self):
        with self._lock:
            if self._movies is not None:
                return  # already loaded

            if not os.path.exists(TFIDF_MATRIX_PATH):
                raise FileNotFoundError(
                    "Model artefacts not found. "
                    "Run `python phase2_content_based/content_model.py` first."
                )

            self._movies       = pd.read_csv(MOVIES_CLEAN_PATH)
            self._tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

            # Build a case-insensitive title → row-index map (raw)
            self._idx_map = {
                title.lower(): idx
                for idx, title in enumerate(self._movies["title"])
            }

            # Build a normalised title → row-index map (strips year + punctuation)
            self._norm_map = {
                _norm(title): idx
                for idx, title in enumerate(self._movies["title"])
            }

            log.info("[ContentRecommender] Loaded %d movies, TF-IDF matrix %s",
                     len(self._movies), self._tfidf_matrix.shape)

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
        pd.DataFrame with columns: movieId, title, genres, similarity_score
        """
        self._load()

        # ── 1. Fuzzy title match ──────────────────────────────────────────
        query_norm = _norm(title)

        # Word-boundary match on normalised titles (handles colons, years, etc.)
        matches = [t for t in self._norm_map if _word_match(query_norm, t)]

        if not matches:
            raise ValueError(f"Movie '{title}' not found in catalogue.")

        # Prefer exact match, then shortest (most specific title)
        best_match = min(matches, key=lambda t: (t != query_norm, len(t)))
        idx        = self._norm_map[best_match]

        matched_title = self._movies.iloc[idx]["title"]
        log.debug("Matched '%s' → '%s'", title, matched_title)

        # ── 2. Score all movies ───────────────────────────────────────────
        sim_scores = linear_kernel(
            self._tfidf_matrix[idx : idx + 1],
            self._tfidf_matrix
        ).flatten()

        # argsort ascending → flip → skip self → take top n
        top_indices = sim_scores.argsort()[::-1]
        top_indices = top_indices[top_indices != idx][:n]

        # ── 3. Build result DataFrame ─────────────────────────────────────
        result = self._movies.iloc[top_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = sim_scores[top_indices].round(4)
        result = result.reset_index(drop=True)

        return result

    def fuzzy_search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Return up to top_k movie titles that contain `query` (case-insensitive).
        Useful for building autocomplete.
        """
        self._load()
        q_norm  = _norm(query)
        matches = [t for t in self._norm_map if _word_match(q_norm, t)]
        matches.sort(key=len)   # shorter = more exact
        # Return original-case titles
        return [self._movies.iloc[self._norm_map[m]]["title"] for m in matches[:top_k]]


# ── Module-level convenience function ────────────────────────────────────────
_recommender = ContentRecommender()


def get_similar_movies(title: str, n: int = 10) -> pd.DataFrame:
    """Convenience wrapper around ContentRecommender.get_similar_movies."""
    return _recommender.get_similar_movies(title, n)


# ── Manual test ──────────────────────────────────────────────────────────────
def _run_tests():
    test_cases = [
        ("Inception",             10),
        ("Toy Story",              5),
        ("The Dark Knight",        8),
        ("Pulp Fiction",           5),
        ("Avengers Age of Ultron", 5),   # colon + year stripped → should match
        ("avengers",               5),   # should NOT match "Scavengers"
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