import os
import sys
import numpy as np
import pandas as pd
import joblib
svd_model = joblib.load("models/svd_model.pkl")

# Ensure top-level package imports work when running this script directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# IMPORT METRICS
# ---------------------------------------------------------

from utils.metrics import (
    compute_rmse_mae,
    baseline_rmse,
    precision_recall_at_k
)

# ---------------------------------------------------------
# IMPORT YOUR MODEL
# ---------------------------------------------------------

from phase3_collaborative_filtering.svd_model import (
    train_svd,
    predict_rating,
    save_model,
    load_model,
)

from phase2_content_based.recommend import get_similar_movies, _recommender
from phase4_hybrid_coldstart.hybrid_model import get_hybrid_recommendations

# =========================================================
# CREATE RESULTS DIRECTORY
# =========================================================

os.makedirs(
    "phase5_evaluation/results",
    exist_ok=True
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading datasets...")

ratings = pd.read_csv(
    "data/raw/ratings.csv"
)

movies = pd.read_csv(
    "data/raw/movies.csv"
)

print(f"Ratings shape : {ratings.shape}")
print(f"Movies shape  : {movies.shape}")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

print("\nTraining SVD model (this does its own train/test split internally)...")

model = train_svd(ratings, k=100, test_size=0.2, random_state=42)
train_df = model["train_df"]
test_df  = model["test_df"]

print(f"Train size : {len(train_df)}")
print(f"Test size  : {len(test_df)}")

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

predictions = []

failed_predictions = 0

for _, row in test_df.iterrows():

    user_id = row["userId"]
    movie_id = row["movieId"]
    actual_rating = row["rating"]

    try:

        # -------------------------------------------------
        # YOUR MODEL PREDICTION
        # -------------------------------------------------

        predicted_rating = predict_rating(
            model,
            user_id,
            movie_id
        )

        # -------------------------------------------------
        # STORE IN STANDARD FORMAT
        # -------------------------------------------------

        predictions.append({
            "user_id": user_id,
            "movie_id": movie_id,
            "actual": actual_rating,
            "predicted": predicted_rating
        })

    except Exception as e:
        if failed_predictions == 0:
            print(f"\n[DEBUG] First failure — user_id={user_id}, movie_id={movie_id}, error: {e}")
        failed_predictions += 1
        continue

print(f"Successful predictions : {len(predictions)}")
print(f"Failed predictions     : {failed_predictions}")

# =========================================================
# COMPUTE RMSE + MAE
# =========================================================

print("\nComputing RMSE and MAE...")

rmse, mae = compute_rmse_mae(
    predictions
)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# =========================================================
# COMPUTE PRECISION@K + RECALL@K
# =========================================================

print("\nComputing Precision@10 and Recall@10...")

precisions, recalls = precision_recall_at_k(
    predictions,
    k=10,
    threshold=4.0
)

mean_precision = np.mean(
    list(precisions.values())
)

mean_recall = np.mean(
    list(recalls.values())
)

print(f"Precision@10 : {mean_precision:.4f}")
print(f"Recall@10    : {mean_recall:.4f}")

# =========================================================
# BASELINE RMSE
# =========================================================

print("\nComputing baseline RMSE...")

base_rmse = baseline_rmse(
    ratings
)

print(f"Baseline RMSE : {base_rmse:.4f}")

improvement = (
    (base_rmse - rmse)
    / base_rmse
) * 100

print(
    f"Improvement over baseline: "
    f"{improvement:.2f}%"
)

# =========================================================
# CONTENT-BASED PRECISION@10
# =========================================================

print("\nEvaluating content-based model (Precision@10)...")
_recommender._load()
_cb_movies = _recommender._movies  # has movieId + title

# Build per-user test positives (movies rated >= 4.0 in test set)
test_positives = (
    test_df[test_df["rating"] >= 4.0]
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Limit to users who have at least one test positive AND at least one training rating
cb_users = [u for u in test_positives if u in train_df["userId"].values][:200]

cb_precisions = []

for user_id in cb_users:
    user_train = train_df[train_df["userId"] == user_id].sort_values("rating", ascending=False)
    if user_train.empty:
        continue

    # Use highest-rated training movie as seed
    seed_movie_id = int(user_train.iloc[0]["movieId"])
    title_rows = _cb_movies[_cb_movies["movieId"] == seed_movie_id]["title"]
    if title_rows.empty:
        continue
    seed_title = title_rows.iloc[0]

    try:
        similar = get_similar_movies(seed_title, n=10)
        recommended_ids = set(similar["movieId"].astype(int).tolist())
    except Exception:
        continue

    positives = test_positives.get(user_id, set())
    hits = len(recommended_ids & positives)
    cb_precisions.append(hits / 10)

cb_mean_precision = float(np.mean(cb_precisions)) if cb_precisions else 0.0
print(f"Content-Based Precision@10 : {cb_mean_precision:.4f}  (over {len(cb_precisions)} users)")

# =========================================================
# HYBRID PRECISION@10
# =========================================================

print("\nEvaluating hybrid model (Precision@10) — this is slow, capped at 100 users...")

hybrid_users = cb_users[:100]  # hybrid is slow; 100 is enough for a fair estimate
hybrid_precisions = []

for user_id in hybrid_users:
    try:
        recs = get_hybrid_recommendations(
            user_id, train_df, movies, model, n=10, alpha=0.5
        )
        recommended_ids = set(int(r["movie_id"]) for r in recs)
    except Exception as e:
        print(f"  [Skip] user {user_id}: {e}")
        continue

    positives = test_positives.get(user_id, set())
    hits = len(recommended_ids & positives)
    hybrid_precisions.append(hits / 10)

hybrid_mean_precision = float(np.mean(hybrid_precisions)) if hybrid_precisions else 0.0
print(f"Hybrid Precision@10 : {hybrid_mean_precision:.4f}  (over {len(hybrid_precisions)} users)")


# =========================================================
# SAVE RESULTS
# =========================================================

print("\nSaving results...")

results_df = pd.DataFrame([
    {
        "Model": "Baseline",
        "RMSE": round(base_rmse, 4),
        "MAE": None,
        "Precision@10": None,
        "Recall@10": None,
    },
    {
        "Model": "Collaborative Filtering",
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Precision@10": round(mean_precision, 4),
        "Recall@10": round(mean_recall, 4),
    },
    {
        "Model": "Content-Based",
        "RMSE": None,
        "MAE": None,
        "Precision@10": round(cb_mean_precision, 4),
        "Recall@10": None,
    },
    {
        "Model": "Hybrid",
        "RMSE": None,
        "MAE": None,
        "Precision@10": round(hybrid_mean_precision, 4),
        "Recall@10": None,
    },
])

results_path = (
    "phase5_evaluation/results/"
    "metrics_summary.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(f"Results saved to: {results_path}")

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\nEvaluation Complete.")
print("-" * 50)

print(f"RMSE          : {rmse:.4f}")
print(f"MAE           : {mae:.4f}")
print(f"Precision@10  : {mean_precision:.4f}")
print(f"Recall@10     : {mean_recall:.4f}")

print("-" * 50)

# =========================================================
# PLOTS
# =========================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("phase5_evaluation/results/plots", exist_ok=True)

# ── RMSE comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(
    ["Baseline (mean)", "SVD Collaborative"],
    [base_rmse, rmse],
    color=["#888780", "#D85A30"],
    width=0.5
)
ax.set_title("RMSE Comparison (lower is better)")
ax.set_ylabel("RMSE")
for i, v in enumerate([base_rmse, rmse]):
    ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("phase5_evaluation/results/plots/rmse_comparison.png",dpi=500)
plt.close()
print("Saved: precision_comparison.png")