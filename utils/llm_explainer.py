"""
llm_explainer.py — Calls Groq API to generate one-line recommendation
explanations. Never called on page load — only after the user has
requested recommendations.

Model: llama-3.3-70b-versatile (fast, free tier, great quality)
Set GROQ_API_KEY in your .env file.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_recommendation(
    movie: dict,
    user_history: list[str],
    model_used: str,
) -> str:
    """
    Generate a one-line explanation for a recommendation.

    Parameters
    ----------
    movie        : dict with keys 'title', 'genres' (pipe-separated e.g. "Action|Sci-Fi")
    user_history : list of movie titles the user has rated highly (pass [] if unknown)
    model_used   : "content" | "hybrid" | "popularity" | "collaborative"

    Returns
    -------
    One sentence under 25 words. Falls back to a genre-based string on any error —
    a failed API call should NEVER break a card render.
    """
    # Popularity fallback needs no LLM
    if model_used == "popularity":
        return "One of the highest-rated movies across all users."

    genres_display = movie.get("genres", "").replace("|", ", ")
    title = movie.get("title", "this movie")
    primary_genre = movie.get("genres", "").split("|")[0] if movie.get("genres") else "film"

    history_str = (
        ", ".join(user_history[:5])
        if user_history
        else "a variety of films"
    )

    if model_used in ("hybrid", "collaborative"):
        method_hint = "based on ratings from users who share your taste"
    else:
        method_hint = "based on genre and content similarity"

    prompt = f"""You are writing terse, specific one-line movie recommendation explanations.

User's top-rated movies: {history_str}
Recommended movie: {title} (genres: {genres_display})
Recommendation method: {method_hint}

Write exactly ONE sentence under 25 words explaining why this recommendation fits.
Rules:
- Be specific — reference genres, themes, or tones from the user's history
- Do NOT start with "I", "This movie", or "Based on"
- Do NOT use filler phrases like "you might enjoy" or "you may like"
- Output only the sentence, nothing else
"""

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        # Strip any accidental quotes the model might add
        return text.strip('"').strip("'")
    except Exception:
        return f"Matches your interest in {primary_genre} films."


def explain_batch(
    recs: list[dict],
    user_history: list[str],
    model_used: str,
) -> list[str]:
    """
    Generate explanations for a list of recommendations.
    Returns a list of strings in the same order as recs.
    Individual failures return the genre fallback string — they never propagate.
    """
    return [
        explain_recommendation(rec, user_history, model_used)
        for rec in recs
    ]