"""
home.py — Recommendations page.
Two modes: movie title search (content-based) and user ID lookup (hybrid/popularity).
Autocomplete is powered by GET /recommend/search.
LLM explanations are generated after recommendations arrive, never on page load.
"""

import pandas as pd
import streamlit as st

from utils.api_client import (
    get_by_movie,
    get_by_user,
    get_popular,
    is_backend_alive,
    search_titles,
)
from utils.llm_explainer import explain_batch
from phase7_frontend.components.movie_card import render_movie_card


# ── Cached data loaders ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_links() -> dict[int, float]:
    """Returns {movieId: tmdbId} mapping from links.csv."""
    try:
        df = pd.read_csv("data/raw/links.csv")
        return df.set_index("movieId")["tmdbId"].to_dict()
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner=False)
def _load_user_ratings() -> pd.DataFrame:
    """Load ratings so we can pull a user's top-rated titles for explanation context."""
    try:
        ratings = pd.read_csv("data/raw/ratings.csv")
        movies  = pd.read_csv("data/raw/movies.csv")
        return ratings.merge(movies[["movieId", "title"]], on="movieId")
    except FileNotFoundError:
        return pd.DataFrame()


def _get_user_history(user_id: int, top_n: int = 5) -> list[str]:
    """Return up to top_n titles the user rated highest (≥4.0), for explanation context."""
    df = _load_user_ratings()
    if df.empty:
        return []
    user_df = df[(df["userId"] == user_id) & (df["rating"] >= 4.0)]
    user_df = user_df.sort_values("rating", ascending=False).head(top_n)
    return user_df["title"].tolist()


def _render_recs(
    recs: list[dict],
    model_used: str,
    latency_ms: float,
    links: dict,
    user_history: list[str],
    generate_explanations: bool,
):
    """Render recommendation cards, optionally with LLM explanations."""
    if not recs:
        st.warning("No recommendations returned. Try a different title or user ID.")
        return

    st.caption(f"Model: `{model_used}` · {latency_ms:.0f} ms · {len(recs)} results")

    if generate_explanations:
        with st.spinner("✨ Generating explanations…"):
            explanations = explain_batch(recs, user_history, model_used)
    else:
        explanations = [None] * len(recs)

    for i, (rec, explanation) in enumerate(zip(recs, explanations), start=1):
        movie_id = rec.get("movie_id") or rec.get("movieId")
        tmdb_id  = links.get(int(movie_id)) if movie_id else None
        render_movie_card(
            rec,
            explanation=explanation,
            tmdb_id=tmdb_id,
            rank=i,
        )


def render():
    st.title("🎬 Movie Recommendations")

    # ── Backend liveness banner ────────────────────────────────────────────
    if not is_backend_alive():
        st.error(
            "⚠️ Cannot reach the FastAPI backend at `localhost:8000`. "
            "Run `uvicorn phase6_api.main:app --reload --port 8000` in another terminal."
        )
        return

    links = _load_links()

    # ── Mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "Find recommendations by",
        ["🎥 Movie title", "👤 User ID"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────
    # MODE 1 — Movie title (content-based similarity)
    # ─────────────────────────────────────────────────────────────────────
    if mode == "🎥 Movie title":
        st.subheader("Find similar movies")

        query = st.text_input(
            "Movie title",
            placeholder="e.g. Inception, The Matrix, Toy Story…",
            label_visibility="collapsed",
        )

        selected_title = None

        if query and len(query) >= 2:
            with st.spinner("Searching…"):
                suggestions = search_titles(query, top_k=7)

            if suggestions:
                selected_title = st.selectbox(
                    "Select a title",
                    options=suggestions,
                    label_visibility="collapsed",
                )
            else:
                st.caption("No matching titles found. Try a different search term.")

        col_n, col_explain, col_btn = st.columns([2, 2, 1])
        with col_n:
            n = st.slider("Number of recommendations", 5, 20, 10, key="movie_n")
        with col_explain:
            gen_explain = st.toggle("Generate AI explanations", value=True)
        with col_btn:
            st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
            clicked = st.button("Find similar", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            if not selected_title:
                st.warning("Please type a movie title and select from the suggestions.")
                return
            with st.spinner(f"Finding movies similar to *{selected_title}*…"):
                response = get_by_movie(selected_title, n=n)
            recs      = response.get("recommendations", [])
            model     = response.get("model_used", "content")
            latency   = response.get("latency_ms", 0)
            _render_recs(recs, model, latency, links, [selected_title], gen_explain)

    # ─────────────────────────────────────────────────────────────────────
    # MODE 2 — User ID (hybrid / popularity fallback)
    # ─────────────────────────────────────────────────────────────────────
    else:
        st.subheader("Get personalised recommendations")
        st.caption(
            "Users with ≥5 ratings get hybrid recommendations. "
            "Users with 1–4 ratings get content-based. "
            "Unknown users get the most popular movies."
        )

        col_uid, col_n2 = st.columns([2, 2])
        with col_uid:
            user_id = st.number_input(
                "User ID",
                min_value=1,
                max_value=999_999,
                value=42,
                step=1,
                label_visibility="collapsed",
            )
        with col_n2:
            n2 = st.slider("Number of recommendations", 5, 20, 10, key="user_n")

        col_explain2, col_btn2 = st.columns([3, 1])
        with col_explain2:
            gen_explain2 = st.toggle("Generate AI explanations", value=True, key="user_explain")
        with col_btn2:
            clicked2 = st.button("Get recommendations", type="primary", use_container_width=True)

        if clicked2:
            with st.spinner(f"Fetching recommendations for user {user_id}…"):
                response = get_by_user(int(user_id), n=n2)
            recs      = response.get("recommendations", [])
            model     = response.get("model_used", "popularity")
            latency   = response.get("latency_ms", 0)
            history   = _get_user_history(int(user_id))
            _render_recs(recs, model, latency, links, history, gen_explain2)

        # Show popular movies by default before any search
        if not clicked2:
            st.divider()
            st.subheader("🔥 Most popular movies right now")
            with st.spinner("Loading popular movies…"):
                popular = get_popular(n=6)
            for i, rec in enumerate(popular, start=1):
                movie_id = rec.get("movie_id") or rec.get("movieId")
                tmdb_id  = links.get(int(movie_id)) if movie_id else None
                render_movie_card(rec, explanation=None, tmdb_id=tmdb_id, rank=i)