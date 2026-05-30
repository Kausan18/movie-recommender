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
# SAVE RESULTS
# =========================================================

print("\nSaving results...")

results_df = pd.DataFrame([
    {
        "Model": "Baseline",
        "RMSE": round(base_rmse, 4),
        "MAE": None,
        "Precision@10": None,
        "Recall@10": None
    },
    {
        "Model": "Collaborative Filtering",
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Precision@10": round(mean_precision, 4),
        "Recall@10": round(mean_recall, 4)
    }
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