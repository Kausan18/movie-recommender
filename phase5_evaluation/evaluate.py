import os
import sys
import numpy as np
import pandas as pd

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

# Example:
# from phase3_collaborative_filtering.svd_model import predict_rating

from phase3_collaborative_filtering.svd_model import predict_rating


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

print("\nSplitting dataset...")

train_df, test_df = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

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