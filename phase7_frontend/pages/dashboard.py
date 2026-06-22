"""
dashboard.py — Evaluation dashboard page.
Shows live /health status + Phase 5 metrics read from evaluation output files.
"""

import csv
import json
import os

import streamlit as st

from utils.api_client import get_health
from phase7_frontend.components.metrics_panel import (
    render_ab_result,
    render_coverage_diversity,
    render_model_comparison,
    render_single_model_metrics,
)

# ── Path to phase5_evaluation/results/ regardless of launch directory ─────
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "phase5_evaluation", "results",
)

def _load_json(filename):
    try:
        with open(os.path.join(RESULTS_DIR, filename), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _load_metrics_csv():
    rows = {}
    try:
        with open(os.path.join(RESULTS_DIR, "metrics_summary.csv"), newline="") as f:
            for row in csv.DictReader(f):
                rows[row["Model"]] = row
    except FileNotFoundError:
        pass
    return rows

def _f(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# ── Load Phase 5 results from disk ───────────────────────────────────────────
_rows = _load_metrics_csv()
_cf   = _rows.get("Collaborative Filtering")

PHASE5_METRICS = {}
if _cf:
    PHASE5_METRICS["SVD"] = {"rmse": _f(_cf.get("RMSE")), "p10": _f(_cf.get("Precision@10"))}
# When Content/Hybrid rows exist in metrics_summary.csv, add them the same way.
_cb = _rows.get("Content-Based")
_hy = _rows.get("Hybrid")

if _cb:
    PHASE5_METRICS["Content-Based"] = {"rmse": _f(_cb.get("RMSE")), "p10": _f(_cb.get("Precision@10"))}

if _hy:
    PHASE5_METRICS["Hybrid"] = {"rmse": _f(_hy.get("RMSE")), "p10": _f(_hy.get("Precision@10"))}

SVD_FULL = {
    "rmse":     _f(_cf.get("RMSE"))         if _cf else None,
    "mae":      _f(_cf.get("MAE"))           if _cf else None,
    "p10":      _f(_cf.get("Precision@10")) if _cf else None,
    "recall10": _f(_cf.get("Recall@10"))    if _cf else None,
}

_ab = _load_json("ab_test_results.json")
AB_TEST = {
    "model_a_name": _ab["model_a"]["name"]               if _ab else "Content-Based",
    "model_b_name": _ab["model_b"]["name"]               if _ab else "SVD",
    "model_a_p10":  _ab["model_a"]["mean_precision_at_k"] if _ab else None,
    "model_b_p10":  _ab["model_b"]["mean_precision_at_k"] if _ab else None,
    "p_value":      _ab["p_value"]                        if _ab else None,
}

_div = _load_json("diversity_summary.json")
# coverage_pct → 0-1 float for render_coverage_diversity.
# diversity (intra-list dissimilarity) has not been computed yet — pass None.
DIVERSITY = {
    "coverage":  (_div["coverage_pct"] / 100) if _div else None,
    "diversity": None,
}
_div_extra = {
    "gini":    _div["gini_coefficient"]  if _div else None,
    "top1pct": _div["top1pct_share_pct"] if _div else None,
    "n_users": _div["sample_users"]      if _div else None,
    "n_unique": _div["unique_recommended"] if _div else None,
} if _div else None


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
            "Run `phase5_evaluation/evaluate.py` for content-based and hybrid numbers — "
            "the dashboard will pick them up automatically once their rows appear in `metrics_summary.csv`."
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
        diversity=None,
    )
    if _div_extra:
        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage", f"{_div['coverage_pct']}%",
                  help=f"{_div_extra['n_unique']} unique movies recommended across {_div_extra['n_users']} sampled users")
        c2.metric("Gini coefficient", f"{_div_extra['gini']:.4f}",
                  help="Popularity concentration (0 = perfectly even, 1 = one movie gets everything)")
        c3.metric("Top 1% share", f"{_div_extra['top1pct']}%",
                  help="Share of all recommendations captured by the top 1% most-recommended movies")
        st.caption(
            f"SVD model only, {_div_extra['n_users']} sampled users. "
            "Gini and top-1% share are popularity-concentration metrics — "
            "not intra-list diversity (average pairwise dissimilarity), which hasn't been computed yet."
        )
    elif DIVERSITY["coverage"] is None:
        st.caption(
            "Run `phase5_evaluation/diversity_analysis.py` — output will be read automatically."
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