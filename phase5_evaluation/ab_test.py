"""
Phase 5 — A/B Test Simulation
Compares Content-Based (Model A) vs SVD Collaborative (Model B)
using per-user Precision@10 as the metric.

Outputs:
    results/ab_test_results.json
    results/plots/ab_test_distribution.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

from phase3_collaborative_filtering.svd_model import train_svd, get_recommendations as svd_get_recs
from phase2_content_based.recommend import ContentRecommender

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

K          = 10
THRESHOLD  = 4.0
RANDOM_SEED = 42

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
ratings = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "ratings.csv"))
movies  = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "movies.csv"))

# ── Train SVD ─────────────────────────────────────────────
print("Training SVD model...")
model    = train_svd(ratings, k=100, test_size=0.2, random_state=RANDOM_SEED)
test_df  = model["test_df"]

# ── Load content-based recommender ────────────────────────
print("Loading content-based recommender...")
cb = ContentRecommender()
cb._load()

# ── Split users 50/50 ─────────────────────────────────────
all_users = list(model["user_index"].keys())
np.random.seed(RANDOM_SEED)
np.random.shuffle(all_users)
SAMPLE_SIZE_PER_GROUP = 500   # cap evaluation size — 500/group is statistically sufficient

mid            = len(all_users) // 2
group_a_users  = set(all_users[:mid][:SAMPLE_SIZE_PER_GROUP])
group_b_users  = set(all_users[mid:][:SAMPLE_SIZE_PER_GROUP])

print(f"\nGroup A (Content-Based) : {len(group_a_users)} users (sampled)")
print(f"Group B (SVD)           : {len(group_b_users)} users (sampled)")

# ── Ground truth: leave-one-out from FULL ratings ─────────
# For each user, hold back up to 5 liked movies as ground truth.
# The rest of their ratings are used for training.
# This way both models are evaluated on movies they could recommend.
np.random.seed(RANDOM_SEED)
user_liked      = defaultdict(set)   # held-out ground truth
user_train_seen = defaultdict(set)   # what we tell models the user has seen

for user_id, group in ratings.groupby("userId"):
    liked = group[group["rating"] >= THRESHOLD]["movieId"].values.copy()
    if len(liked) < 2:
        user_train_seen[user_id] = set(group["movieId"].values)
        continue
    np.random.shuffle(liked)
    holdout = set(liked[:5])          # up to 5 held-out liked movies
    seen    = set(group["movieId"].values) - holdout
    user_liked[user_id]      = holdout
    user_train_seen[user_id] = seen

# =========================================================
# PRECISION@K HELPER
# =========================================================

def precision_at_k(recommended_ids: list, liked_ids: set, k: int) -> float:
    top_k = recommended_ids[:k]
    hits  = sum(1 for mid in top_k if mid in liked_ids)
    return hits / k


# =========================================================
# MODEL A — CONTENT-BASED
# =========================================================

print("\nEvaluating Model A (Content-Based)...")
scores_a = []

for user_id in group_a_users:
    liked = user_liked.get(user_id, set())
    if len(liked) < 2:
        continue

    # Seed movie = first liked movie the user rated highly in TRAIN set
    train_liked = list(
        user_train_seen.get(user_id, set()) &
        set(ratings[ratings["rating"] >= THRESHOLD]["movieId"].values)
    )

    if len(train_liked) == 0:
        continue

    seed_id = train_liked[0]

    seed_title = movies[movies["movieId"] == seed_id]["title"].values
    if len(seed_title) == 0:
        continue

    try:
        recs        = cb.get_similar_movies(seed_title[0], n=K)
        rec_ids     = recs["movieId"].tolist()
        scores_a.append(precision_at_k(rec_ids, liked, K))
    except Exception:
        continue

print(f"  Evaluated {len(scores_a)} users")

# =========================================================
# MODEL B — SVD COLLABORATIVE
# =========================================================

print("Evaluating Model B (SVD)...")
scores_b = []

for user_id in group_b_users:
    liked = user_liked.get(user_id, set())
    if len(liked) < 2:
        continue

    try:
        # Pass only train-seen ratings so held-out movies stay recommendable
        ratings_filtered = ratings[
            ratings["movieId"].isin(user_train_seen.get(user_id, set())) |
            (ratings["userId"] != user_id)
        ]
        recs    = svd_get_recs(user_id, model, ratings_filtered, movies, n=K)
        rec_ids = recs["movieId"].tolist()
        scores_b.append(precision_at_k(rec_ids, liked, K))
    except Exception:
        continue


print(f"  Evaluated {len(scores_b)} users")

# =========================================================
# T-TEST
# =========================================================

t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
sig             = p_value < 0.05
winner          = "Model B (SVD)" if np.mean(scores_b) > np.mean(scores_a) else "Model A (Content-Based)"

print(f"\n{'='*52}")
print("A/B TEST RESULTS")
print(f"{'='*52}")
print(f"Model A (Content-Based)  mean Precision@{K} : {np.mean(scores_a):.4f}")
print(f"Model B (SVD)            mean Precision@{K} : {np.mean(scores_b):.4f}")
print(f"t-statistic : {t_stat:.4f}")
print(f"p-value     : {p_value:.6f}")
print(f"Significant : {'YES ✅' if sig else 'NO ❌'}")
print(f"Winner      : {winner}")
print(f"{'='*52}")

# =========================================================
# PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(scores_a, bins=20, alpha=0.6, label="Model A (Content-Based)", color="#7F77DD")
ax.hist(scores_b, bins=20, alpha=0.6, label="Model B (SVD)", color="#D85A30")
ax.axvline(np.mean(scores_a), color="#7F77DD", linestyle="--", linewidth=2,
           label=f"A mean: {np.mean(scores_a):.3f}")
ax.axvline(np.mean(scores_b), color="#D85A30", linestyle="--", linewidth=2,
           label=f"B mean: {np.mean(scores_b):.3f}")
ax.set_xlabel("Precision@10")
ax.set_ylabel("Number of Users")
ax.set_title("A/B Test: Distribution of Precision@10 per User")
ax.legend()
plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "ab_test_distribution.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved: ab_test_distribution.png")

# =========================================================
# SAVE JSON
# =========================================================

results = {
    "model_a": {
        "name"              : "Content-Based",
        "n_users"           : len(scores_a),
        "mean_precision_at_k": round(float(np.mean(scores_a)), 4),
        "std"               : round(float(np.std(scores_a)), 4),
    },
    "model_b": {
        "name"              : "SVD Collaborative",
        "n_users"           : len(scores_b),
        "mean_precision_at_k": round(float(np.mean(scores_b)), 4),
        "std"               : round(float(np.std(scores_b)), 4),
    },
    "t_statistic"           : round(float(t_stat), 4),
    "p_value"               : round(float(p_value), 6),
    "statistically_significant": bool(sig),
    "winner"                : winner,
}

json_path = os.path.join(RESULTS_DIR, "ab_test_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved: ab_test_results.json")
print("\nA/B Test Complete.")