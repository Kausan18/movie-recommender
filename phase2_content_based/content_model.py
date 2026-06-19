"""
Phase 2 — Content-Based Filtering
Builds a TF-IDF genre+tag matrix. Saves tfidf_matrix.pkl (sparse) instead of
cosine_sim_matrix.pkl. Similarity is computed on-the-fly at query time via
linear_kernel — scales to 62K movies (25M dataset) without memory issues.

Usage (training):
    python phase2_content_based/content_model.py

Usage (inference):
    from phase2_content_based.content_model import get_similar_movies
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel   # ← replaces cosine_similarity
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

RAW_MOVIES  = os.path.join(DATA_DIR, "raw", "movies.csv")
RAW_TAGS    = os.path.join(DATA_DIR, "raw", "tags.csv")
RAW_GENOME_SCORES = os.path.join(DATA_DIR, "raw", "genome-scores.csv")
RAW_GENOME_TAGS   = os.path.join(DATA_DIR, "raw", "genome-tags.csv")

# Fallback: look in project root (matches your existing fallback logic)
if not os.path.exists(RAW_MOVIES):
    RAW_MOVIES = os.path.join(os.path.dirname(__file__), "..", "movies.csv")
    RAW_TAGS   = os.path.join(os.path.dirname(__file__), "..", "tags.csv")

# ── Artifact paths ─────────────────────────────────────────────────────────────
TFIDF_VECTORISER_PATH = os.path.join(MODELS_DIR, "tfidf_vectoriser.pkl")
TFIDF_MATRIX_PATH     = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")       # NEW — sparse matrix
MOVIES_CLEAN_PATH     = os.path.join(DATA_DIR, "processed", "movies_clean.csv")

# ── Text cleaning constants ────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","of","and","or","in","on","at","to","for",
    "is","it","its","from","with","by","as","that","this","part",
    "one","two","three","four","five","i","ii","iii","iv","v"
}

NOISE_TAGS = {
    "in netflix queue", "imax", "watched", "seen", "own",
    "favorite", "favourite", "dvd", "blu-ray", "bluray",
}

# Genome relevance threshold — only keep tags where relevance >= this value.
# 0.5 keeps strong signals; lowering to 0.4 adds more tags, raising to 0.6 keeps fewer.
GENOME_RELEVANCE_THRESHOLD = 0.5

# Top-N genome tags per movie to include in soup (keeps soup size bounded)
GENOME_TOP_N = 10


# ── Helper functions ───────────────────────────────────────────────────────────

def _title_tokens(title: str) -> str:
    """'Iron Man (2008)' → 'iron man'  (no year, no stopwords)"""
    clean = re.sub(r"\(\d{4}\)", "", title).strip()
    tokens = re.findall(r"[a-z]+", clean.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)


def _year_token(title: str) -> str:
    """'Iron Man (2008)' → 'yr2008'  (gives year a distinct non-numeric token)"""
    m = re.search(r"\((\d{4})\)", title)
    return f"yr{m.group(1)}" if m else ""


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Load movies, user tags, and genome tags (if available).
    Genome files are optional — if absent (100K dataset), genome_tags is None.
    """
    movies = pd.read_csv(RAW_MOVIES)
    tags   = pd.read_csv(RAW_TAGS)
    print(f"Loaded {len(movies):,} movies  |  {len(tags):,} user tags")

    genome_tags = None
    if os.path.exists(RAW_GENOME_SCORES) and os.path.exists(RAW_GENOME_TAGS):
        print("Genome files detected — loading (this takes ~30 s for 25M)...")
        genome_scores = pd.read_csv(RAW_GENOME_SCORES)   # movieId, tagId, relevance
        genome_labels = pd.read_csv(RAW_GENOME_TAGS)     # tagId, tag
        # Merge tag names onto scores
        genome_merged = genome_scores.merge(genome_labels, on="tagId")
        print(f"  Genome entries: {len(genome_merged):,}  |  Unique tags: {genome_labels['tag'].nunique()}")
        genome_tags = genome_merged
    else:
        print("No genome files found — using user tags only (100K mode)")

    return movies, tags, genome_tags


# ── Genome tag aggregation ─────────────────────────────────────────────────────

def _build_genome_agg(genome_tags: pd.DataFrame) -> pd.DataFrame:
    """
    For each movie, take the top-N genome tags by relevance score.
    Returns a DataFrame with columns [movieId, genome_tags_clean].

    Why threshold + top-N instead of all tags?
    - The genome has 1,128 tags per movie. Including all of them at full weight
      would swamp the genre and user-tag signals.
    - Keeping only high-relevance tags (≥ 0.5) focuses on definitive descriptors.
    - Top-N cap keeps soup strings a bounded length.
    """
    agg = (
        genome_tags[genome_tags["relevance"] >= GENOME_RELEVANCE_THRESHOLD]
        .sort_values(["movieId", "relevance"], ascending=[True, False])
        .groupby("movieId")
        .head(GENOME_TOP_N)
        .assign(tag=lambda df: df["tag"].str.lower().str.strip())
        .groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(ts))
        .reset_index()
        .rename(columns={"tag": "genome_tags_clean"})
    )
    print(f"  Genome agg: {len(agg):,} movies have genome tags")
    return agg


# ── Soup building ──────────────────────────────────────────────────────────────

def build_soup(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    genome_tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge genres + user tags + genome tags (if available) into a single
    text string per movie.

    Soup composition:
        genres_clean  ×3  (dominant signal — every movie has these)
        title_tokens  ×1  (prevents identical-genre vectors for tag-less movies)
        year_token    ×1  (groups films from same era)
        user tags     ×1  (community annotations — sparse, bonus signal)
        genome tags   ×2  (curated, high-quality — weighted above user tags)
    """
    movies = movies.copy()

    # ── 1. Clean genres ───────────────────────────────────────────────────────
    movies["genres_clean"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.replace("IMAX", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.strip()
    )

    # Filter out movies with no usable genre (25M has more of these)
    no_genre_mask = movies["genres_clean"].isin(["no genres listed", "", "(no genres listed)"])
    n_dropped = no_genre_mask.sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped:,} movies with no genre information")
        movies = movies[~no_genre_mask].copy()

    # Reset index so positional indexing into tfidf_matrix stays aligned
    movies = movies.reset_index(drop=True)

    # ── 2. User tags ──────────────────────────────────────────────────────────
    user_tag_agg = (
        tags.dropna(subset=["tag"])
        .assign(tag=lambda df: df["tag"].str.lower().str.strip())
        .loc[lambda df: ~df["tag"].isin(NOISE_TAGS)]
        .drop_duplicates(subset=["movieId", "tag"])
        .groupby("movieId")["tag"]
        .apply(lambda ts: " ".join(sorted(set(ts))))
        .reset_index()
        .rename(columns={"tag": "tags_clean"})
    )
    movies = movies.merge(user_tag_agg, on="movieId", how="left")
    movies["tags_clean"] = movies["tags_clean"].fillna("")

    # ── 3. Genome tags (25M only) ─────────────────────────────────────────────
    if genome_tags is not None:
        genome_agg = _build_genome_agg(genome_tags)
        movies = movies.merge(genome_agg, on="movieId", how="left")
        movies["genome_tags_clean"] = movies["genome_tags_clean"].fillna("")
    else:
        movies["genome_tags_clean"] = ""

    # ── 4. Title and year tokens ──────────────────────────────────────────────
    movies["title_tokens"] = movies["title"].apply(_title_tokens)
    movies["year_token"]   = movies["title"].apply(_year_token)

    # ── 5. Assemble soup ──────────────────────────────────────────────────────
    # Genome tags weighted ×2 (repeat them) — richer signal than user tags
    movies["soup"] = (
        (movies["genres_clean"] + " ") * 3
        + movies["title_tokens"] + " "
        + movies["year_token"] + " "
        + movies["tags_clean"] + " "
        + (movies["genome_tags_clean"] + " ") * 2
    ).str.strip()

    print(f"\nSample soups:")
    for _, row in movies.sample(3, random_state=42).iterrows():
        print(f"  [{row['title']}]  →  {row['soup'][:100]}...")

    return movies


# ── TF-IDF ─────────────────────────────────────────────────────────────────────

def build_tfidf(movies: pd.DataFrame):
    """
    Fit TF-IDF on the soup column.
    max_features bumped to 8_000 for 25M (larger vocabulary from genome tags).
    Returns the vectoriser and the sparse feature matrix.
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=8_000,      # was 5_000 — genome vocab needs more room
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(movies["soup"])
    print(f"\nTF-IDF matrix shape : {tfidf_matrix.shape}")
    print(f"Vocabulary size     : {len(tfidf.vocabulary_):,}")
    return tfidf, tfidf_matrix


# ── Save ───────────────────────────────────────────────────────────────────────

def save_artifacts(movies: pd.DataFrame, tfidf, tfidf_matrix):
    """
    Save three artifacts:
      - movies_clean.csv           (processed movie metadata)
      - tfidf_vectoriser.pkl       (fitted vectoriser — for inspecting vocabulary)
      - tfidf_matrix.pkl           (sparse feature matrix — used at query time)

    cosine_sim_matrix.pkl is NOT saved. At 62K movies it would be ~31 GB.
    Similarity is computed on-the-fly via linear_kernel in get_similar_movies().
    """
    movies.to_csv(MOVIES_CLEAN_PATH, index=False)
    joblib.dump(tfidf,        TFIDF_VECTORISER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

    import os as _os
    matrix_mb = _os.path.getsize(TFIDF_MATRIX_PATH) / 1e6
    print(f"\nSaved:")
    print(f"  {MOVIES_CLEAN_PATH}")
    print(f"  {TFIDF_VECTORISER_PATH}")
    print(f"  {TFIDF_MATRIX_PATH}  ({matrix_mb:.1f} MB)")
    print(f"\n  ✗ cosine_sim_matrix.pkl — NOT saved (on-the-fly via linear_kernel)")


# ── Inference ──────────────────────────────────────────────────────────────────

# Module-level cache so models load once when imported, not per call
_movies_df: pd.DataFrame | None = None
_tfidf_matrix = None


def _load_inference_artifacts():
    """Load movies_clean.csv and tfidf_matrix.pkl once, cache in module globals."""
    global _movies_df, _tfidf_matrix
    if _movies_df is None:
        _movies_df    = pd.read_csv(MOVIES_CLEAN_PATH)
        _tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
        print(f"[content_model] Loaded {len(_movies_df):,} movies + TF-IDF matrix {_tfidf_matrix.shape}")


def get_similar_movies(title: str, n: int = 10) -> pd.DataFrame:
    """
    Return the top-n movies most similar to `title`.

    Uses linear_kernel on a single TF-IDF row — O(n_movies) not O(n_movies²).
    At 62K movies this takes ~50 ms. Safe on Streamlit Cloud (1 GB RAM).

    Returns a DataFrame with columns: movieId, title, genres, similarity_score.
    Returns empty DataFrame if title not found.
    """
    _load_inference_artifacts()

    # ── Match title ───────────────────────────────────────────────────────────
    matches = _movies_df[
        _movies_df["title"].str.contains(title, case=False, na=False, regex=False)
    ]
    if matches.empty:
        return pd.DataFrame()

    # Prefer exact prefix match when multiple results (e.g. "Alien" vs "Aliens")
    if len(matches) > 1:
        prefix = matches[matches["title"].str.lower().str.startswith(title.lower())]
        if not prefix.empty:
            matches = prefix

    # Use positional index (safe even if movies_df has non-contiguous index)
    idx_label = matches.index[0]
    idx_pos   = _movies_df.index.get_loc(idx_label)

    # ── Compute similarity for this ONE row ────────────────────────────────────
    # linear_kernel on L2-normalised TF-IDF = cosine similarity, just faster
    sim_scores = linear_kernel(
        _tfidf_matrix[idx_pos : idx_pos + 1],
        _tfidf_matrix
    ).flatten()

    # argsort ascending → reverse → skip index 0 (the query movie itself)
    top_indices = sim_scores.argsort()[::-1][1 : n + 1]

    result = _movies_df.iloc[top_indices][["movieId", "title", "genres"]].copy()
    result["similarity_score"] = sim_scores[top_indices].round(4)
    return result.reset_index(drop=True)


# ── Training entry point ───────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("  Phase 2 — Content-Based Model: Training")
    print("=" * 55)

    movies, tags, genome_tags = load_data()
    movies = build_soup(movies, tags, genome_tags)
    tfidf, tfidf_matrix = build_tfidf(movies)
    save_artifacts(movies, tfidf, tfidf_matrix)

    print("\n✓ Training complete.")
    print("  Run recommend.py to verify recommendations.")
    return movies, tfidf, tfidf_matrix


if __name__ == "__main__":
    train()