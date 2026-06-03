"""
movie_card.py — Reusable movie recommendation card.
Renders poster, title, genres, score bar, and LLM explanation.
Degrades gracefully: missing poster → emoji; missing explanation → nothing shown.
"""

import streamlit as st
from utils.api_client import get_poster_url


# Model badge colours (background, text)
_MODEL_COLOURS = {
    "content":      ("#E8F4FF", "#1A6BB5"),
    "hybrid":       ("#E8F8F1", "#1D7A52"),
    "collaborative": ("#F0EBFF", "#5B3FBF"),
    "popularity":   ("#FFF4E0", "#A05C00"),
}


def _model_badge(model_used: str) -> str:
    bg, fg = _MODEL_COLOURS.get(model_used, ("#F0F0F0", "#444444"))
    label = model_used.capitalize()
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:999px;font-size:11px;font-weight:600;">{label}</span>'
    )


def render_movie_card(
    rec: dict,
    explanation: str | None = None,
    tmdb_id: int | float | None = None,
    show_model_badge: bool = False,
    model_used: str = "",
    rank: int | None = None,
):
    """
    Render a single movie recommendation card.

    Parameters
    ----------
    rec              : recommendation dict from the API.
                       Expected keys: title, genres, movie_id,
                       and one of: predicted_rating, hybrid_score,
                       content_score, weighted_score, similarity_score.
    explanation      : LLM-generated one-liner or None.
    tmdb_id          : TMDB movie ID for poster fetch (from links.csv lookup).
    show_model_badge : Whether to show a "Hybrid" / "Content" pill on the card.
    model_used       : Model name string for the badge.
    rank             : Optional rank number (1, 2, 3…) shown on the card.
    """
    with st.container(border=True):
        col_img, col_text = st.columns([1, 3], gap="medium")

        # ── Poster ─────────────────────────────────────────────────────────
        with col_img:
            poster_url = get_poster_url(tmdb_id) if tmdb_id else None
            if poster_url:
                st.image(poster_url, use_column_width=True)
            else:
                st.markdown(
                    '<div style="font-size:52px;text-align:center;'
                    'padding:20px 0;opacity:0.4;">🎬</div>',
                    unsafe_allow_html=True,
                )

        # ── Text content ────────────────────────────────────────────────────
        with col_text:
            # Rank + title row
            title = rec.get("title", "Unknown Title")
            rank_prefix = f"**#{rank}** &nbsp;" if rank else ""
            st.markdown(f"{rank_prefix}**{title}**", unsafe_allow_html=True)

            # Genres as pills
            genres_raw = rec.get("genres", "")
            if genres_raw and genres_raw != "(no genres listed)":
                genre_list = genres_raw.split("|")
                pills_html = " ".join(
                    f'<span style="background:#F0F4FF;color:#3D5A99;'
                    f'padding:2px 8px;border-radius:999px;font-size:11px;">'
                    f'{g}</span>'
                    for g in genre_list
                )
                st.markdown(pills_html, unsafe_allow_html=True)
            st.markdown("")  # small spacer

            # Score — use whichever field is populated
            score = (
                rec.get("predicted_rating")
                or rec.get("hybrid_score")
                or rec.get("content_score")
                or rec.get("weighted_score")
                or rec.get("similarity_score")
            )
            if score is not None:
                score_f = float(score)
                # Normalise: predicted_rating is on 0–5 scale, others are 0–1
                is_rating = score_f > 1.5
                label = f"Predicted rating: {score_f:.2f}/5" if is_rating else f"Match score: {score_f:.3f}"
                normalised = min(score_f / 5.0, 1.0) if is_rating else min(score_f, 1.0)
                st.progress(normalised, text=label)

            # Model badge (optional)
            if show_model_badge and model_used:
                st.markdown(_model_badge(model_used), unsafe_allow_html=True)

            # LLM explanation
            if explanation:
                st.markdown(
                    f'<div style="margin-top:8px;padding:8px 12px;'
                    f'background:#F7F9FF;border-left:3px solid #4A80D4;'
                    f'border-radius:0 6px 6px 0;font-size:13px;color:#333;">'
                    f'✨ {explanation}</div>',
                    unsafe_allow_html=True,
                )