"""
dashboard.py — Evaluation dashboard page.
Shows live /health status + hardcoded Phase 5 metrics as Plotly bar charts.
Replace the placeholder values with actual Phase 5 output once available.
"""

import streamlit as st

from utils.api_client import get_health
from phase7_frontend.components.metrics_panel import (
    render_ab_result,
    render_coverage_diversity,
    render_model_comparison,
    render_single_model_metrics,
)

# ── Phase 5 results — update these after running phase5_evaluation/ ──────────
# SVD numbers are confirmed from Phase 5. Content and Hybrid filled once run.
PHASE5_METRICS = {
    # "Content":  {"rmse": None,   "p10": None},    # uncomment + fill once run
    "SVD":      {"rmse": 0.9483, "p10": 0.5138},
    # "Hybrid":   {"rmse": None,   "p10": None},    # uncomment + fill once run
}

SVD_FULL = {
    "rmse":    0.9483,
    "mae":     0.7353,
    "p10":     0.5138,
    "recall10": 0.6505,
}

# A/B test — fill from phase5_evaluation/ab_test.py output
AB_TEST = {
    "model_a_name": "Content",
    "model_b_name": "SVD",
    "model_a_p10":  0.0,      # replace with actual value
    "model_b_p10":  0.5138,
    "p_value":      0.0,      # replace with actual value
}

# Diversity / coverage — fill from phase5_evaluation/diversity_analysis.py
DIVERSITY = {
    "coverage":  None,   # replace with float e.g. 0.312
    "diversity": None,   # replace with float e.g. 0.847
}


def render():
    st.title("📊 Evaluation Dashboard")

    # ── Live API health ────────────────────────────────────────────────────
    st.subheader("API health")
    health = get_health()

    if not health:
        st.error(
            "Cannot reach the API. "
            "Run `uvicorn phase6_api.main:app --reload --port 8000` first."
        )
    else:
        models_loaded = health.get("models_loaded", {})
        if models_loaded:
            cols = st.columns(len(models_loaded))
            for col, (name, status) in zip(cols, models_loaded.items()):
                icon = "✅" if status else "❌"
                col.metric(name, f"{icon} {'loaded' if status else 'missing'}")
        st.caption(f"Status: `{health.get('status', 'unknown')}` · Version: `{health.get('version', '—')}`")

    st.divider()

    # ── SVD detailed metrics ───────────────────────────────────────────────
    st.subheader("SVD model — Phase 5 results")
    render_single_model_metrics(
        rmse=SVD_FULL["rmse"],
        mae=SVD_FULL["mae"],
        p10=SVD_FULL["p10"],
        recall10=SVD_FULL["recall10"],
    )

    st.divider()

    # ── Cross-model comparison ─────────────────────────────────────────────
    st.subheader("Model comparison")

    filled = {k: v for k, v in PHASE5_METRICS.items() if v.get("p10") is not None}
    if len(filled) >= 2:
        render_model_comparison(filled)
    else:
        st.info(
            "Only SVD metrics are available so far. "
            "Run `phase5_evaluation/evaluate.py` for content and hybrid numbers, "
            "then fill in the `PHASE5_METRICS` dict at the top of `dashboard.py`."
        )

    st.divider()

    # ── A/B test ───────────────────────────────────────────────────────────
    st.subheader("A/B test simulation")
    render_ab_result(**AB_TEST)

    st.divider()

    # ── Coverage / diversity ───────────────────────────────────────────────
    st.subheader("Diversity & coverage")
    render_coverage_diversity(
        coverage=DIVERSITY["coverage"],
        diversity=DIVERSITY["diversity"],
    )
    if DIVERSITY["coverage"] is None:
        st.caption(
            "Run `phase5_evaluation/diversity_analysis.py` and fill in the "
            "`DIVERSITY` dict at the top of `dashboard.py`."
        )

    st.divider()

    # ── What I'd improve ──────────────────────────────────────────────────
    with st.expander("📝 What I'd improve with more time"):
        st.markdown("""
1. **Implicit feedback** — Real users don't rate movies; they watch them. ALS/BPR on watch-time signals would be more realistic than explicit ratings.
2. **Online learning** — The current model retrains from scratch. A production system would update incrementally as new ratings arrive.
3. **Deep learning recommender** — Neural collaborative filtering or two-tower models capture non-linear interactions that SVD misses.
4. **Better cold-start onboarding** — Asking new users to rate 5 seed movies at signup eliminates cold-start more elegantly than popularity fallback.
5. **Redis caching** — The API recomputes recommendations on every request. Caching frequent users would cut latency significantly.
        """)