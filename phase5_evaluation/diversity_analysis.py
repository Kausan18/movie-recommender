"""
Phase 5 — Diversity & Popularity Bias Analysis
Run after evaluate.py has trained the SVD model.

Outputs:
    results/plots/popularity_bias.png
    results/diversity_summary.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

from phase3_collaborative_filtering.svd_model import train_svd, get_recommendations

# ── Config ────────────────────────────────────────────────
SAMPLE_USERS   = 200   # how many users to generate recs for
N_RECS         = 10    # top-N per user
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
ratings = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "ratings.csv"))
movies  = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "movies.csv"))

# ── Train model ───────────────────────────────────────────
print("Training SVD model...")
model = train_svd(ratings, k=100, test_size=0.2, random_state=42)

# ── Generate recommendations for sample users ─────────────
print(f"\nGenerating top-{N_RECS} recs for {SAMPLE_USERS} users...")

all_users = list(model["user_index"].keys())[:SAMPLE_USERS]
all_recommendations = {}

for user_id in all_users:
    try:
        recs = get_recommendations(user_id, model, ratings, movies, n=N_RECS)
        all_recommendations[user_id] = recs["movieId"].tolist()
    except Exception:
        continue

print(f"Generated recs for {len(all_recommendations)} users.")

# =========================================================
# COVERAGE
# =========================================================

total_movies    = movies["movieId"].nunique()
recommended_set = set(mid for recs in all_recommendations.values() for mid in recs)
coverage        = len(recommended_set) / total_movies

print(f"\nCoverage : {len(recommended_set)} / {total_movies} movies "
      f"({coverage * 100:.1f}%)")

# =========================================================
# GINI COEFFICIENT
# =========================================================

rec_counts = Counter(
    mid for recs in all_recommendations.values() for mid in recs
)
freqs = sorted(rec_counts.values())
n     = len(freqs)
gini  = (
    2 * sum((i + 1) * f for i, f in enumerate(freqs))
) / (n * sum(freqs)) - (n + 1) / n

print(f"Gini coefficient : {gini:.4f}  "
      f"(0 = perfectly spread, 1 = always same movies)")

# ── Top 1% concentration ──────────────────────────────────
top_1pct_count = max(1, int(len(rec_counts) * 0.01))
top_1pct_share = (
    sum(sorted(rec_counts.values(), reverse=True)[:top_1pct_count])
    / sum(rec_counts.values()) * 100
)
print(f"Top 1% of recommended movies get "
      f"{top_1pct_share:.1f}% of all recommendations")

# =========================================================
# POPULARITY BIAS PLOT
# =========================================================

actual_popularity = ratings.groupby("movieId").size().to_dict()

movie_ids = list(rec_counts.keys())
rec_freq  = [rec_counts[m] for m in movie_ids]
act_pop   = [actual_popularity.get(m, 0) for m in movie_ids]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — top 20 most recommended
top20 = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:20]
top20_titles = [
    movies[movies["movieId"] == mid]["title"].values[0][:25]
    if len(movies[movies["movieId"] == mid]) > 0 else str(mid)
    for mid, _ in top20
]
axes[0].barh(
    range(20), [c for _, c in top20],
    color="#1D9E75"
)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top20_titles, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_title("Top 20 Most Recommended Movies")
axes[0].set_xlabel("Times Recommended")

# Right — rec frequency vs actual popularity
axes[1].scatter(act_pop, rec_freq, alpha=0.4, color="#D85A30", s=12)
axes[1].set_xlabel("Actual Popularity (# ratings in dataset)")
axes[1].set_ylabel("Times Recommended")
axes[1].set_title("Popularity Bias: Recommended vs Actual Popularity")
axes[1].set_xscale("log")

plt.suptitle("Diversity & Popularity Bias Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "popularity_bias.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved: popularity_bias.png")

# =========================================================
# SAVE SUMMARY
# =========================================================

summary = {
    "sample_users"      : len(all_recommendations),
    "total_movies"      : total_movies,
    "unique_recommended": len(recommended_set),
    "coverage_pct"      : round(coverage * 100, 2),
    "gini_coefficient"  : round(gini, 4),
    "top1pct_share_pct" : round(top_1pct_share, 2),
}

summary_path = os.path.join(RESULTS_DIR, "diversity_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved: diversity_summary.json")
print("\nDiversity Analysis Complete.")
print("-" * 40)
for k, v in summary.items():
    print(f"{k:<25} : {v}")
print("-" * 40)