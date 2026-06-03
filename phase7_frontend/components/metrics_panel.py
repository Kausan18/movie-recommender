"""
metrics_panel.py — Plotly charts and stat widgets for the evaluation dashboard.
All functions are pure render helpers — no data fetching happens here.
"""

import streamlit as st
import plotly.graph_objects as go


_PALETTE = ["#1D9E75", "#7F77DD", "#D85A30", "#378ADD", "#BA7517"]


def render_model_comparison(metrics: dict):
    """
    Side-by-side bar charts for RMSE and Precision@10.

    Parameters
    ----------
    metrics : {
        "Content":  {"rmse": float | None, "p10": float | None},
        "SVD":      {"rmse": 0.9483,       "p10": 0.5138},
        "Hybrid":   {"rmse": float | None, "p10": float | None},
    }
    Only models with non-None values are rendered.
    """
    models = list(metrics.keys())
    colours = _PALETTE[: len(models)]

    rmse_data = [(m, metrics[m]["rmse"]) for m in models if metrics[m].get("rmse") is not None]
    p10_data  = [(m, metrics[m]["p10"])  for m in models if metrics[m].get("p10")  is not None]

    col1, col2 = st.columns(2)

    with col1:
        if rmse_data:
            names, vals = zip(*rmse_data)
            fig = go.Figure(
                go.Bar(
                    x=list(names),
                    y=list(vals),
                    marker_color=colours[: len(names)],
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="RMSE — lower is better",
                yaxis_title="RMSE",
                height=320,
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(gridcolor="#EEEEEE")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RMSE values to display yet.")

    with col2:
        if p10_data:
            names, vals = zip(*p10_data)
            fig = go.Figure(
                go.Bar(
                    x=list(names),
                    y=list(vals),
                    marker_color=colours[: len(names)],
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Precision@10 — higher is better",
                yaxis_title="Precision@10",
                yaxis_range=[0, 1],
                height=320,
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(gridcolor="#EEEEEE")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Precision@10 values to display yet.")


def render_single_model_metrics(rmse: float, mae: float, p10: float, recall10: float):
    """
    Four KPI tiles for a single model — used to call out SVD Phase 5 results.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",       f"{rmse:.4f}",   help="Root Mean Squared Error — lower is better")
    c2.metric("MAE",        f"{mae:.4f}",    help="Mean Absolute Error — lower is better")
    c3.metric("Precision@10", f"{p10:.4f}", help="Fraction of top-10 recs the user actually rated ≥4")
    c4.metric("Recall@10",  f"{recall10:.4f}", help="Fraction of all liked movies that appear in top-10")


def render_ab_result(
    model_a_name: str,
    model_b_name: str,
    model_a_p10: float,
    model_b_p10: float,
    p_value: float,
):
    """
    Three-column A/B test summary with significance badge.
    """
    delta = model_b_p10 - model_a_p10
    sig = p_value < 0.05

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{model_a_name} Precision@10", f"{model_a_p10:.4f}")
    col2.metric(
        f"{model_b_name} Precision@10",
        f"{model_b_p10:.4f}",
        delta=f"{delta:+.4f}",
    )
    col3.metric(
        "p-value",
        f"{p_value:.4f}",
        delta="✅ significant" if sig else "❌ not significant",
        delta_color="normal" if sig else "off",
    )

    if p_value == 0.0:
        st.caption("⚠️ Fill in actual A/B test numbers from `phase5_evaluation/ab_test.py` output.")


def render_coverage_diversity(coverage: float | None, diversity: float | None):
    """
    Simple two-metric row for coverage and diversity scores.
    Pass None for metrics not yet computed.
    """
    col1, col2 = st.columns(2)
    col1.metric(
        "Catalogue Coverage",
        f"{coverage:.1%}" if coverage is not None else "—",
        help="% of all 9,742 movies that appear in at least one user's top-10",
    )
    col2.metric(
        "Intra-List Diversity",
        f"{diversity:.4f}" if diversity is not None else "—",
        help="Average pairwise dissimilarity within recommendation lists (higher = more varied)",
    )