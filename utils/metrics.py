import numpy as np
from collections import defaultdict


# =========================================================
# RMSE + MAE
# =========================================================

def compute_rmse_mae(predictions):
    """
    predictions:
    [
        {
            "user_id": ...,
            "movie_id": ...,
            "actual": ...,
            "predicted": ...
        }
    ]
    """

    actuals = np.array([
        pred["actual"]
        for pred in predictions
    ])

    estimated = np.array([
        pred["predicted"]
        for pred in predictions
    ])

    rmse = np.sqrt(
        np.mean((actuals - estimated) ** 2)
    )

    mae = np.mean(
        np.abs(actuals - estimated)
    )

    return rmse, mae
# =========================================================
# BASELINE RMSE
# =========================================================

def baseline_rmse(ratings_df):
    """
    Compute baseline RMSE using global mean rating.

    This simulates the dumbest possible recommender:
    predict the same mean rating for every movie.

    Parameters
    ----------
    ratings_df : pd.DataFrame

    Returns
    -------
    baseline_rmse : float
    """

    global_mean = ratings_df["rating"].mean()

    errors = (ratings_df["rating"] - global_mean) ** 2

    return np.sqrt(errors.mean())


# =========================================================
# PRECISION@K + RECALL@K
# =========================================================

def precision_recall_at_k(
    predictions,
    k=10,
    threshold=4.0
):

    user_predictions = defaultdict(list)

    # -------------------------------------------------
    # GROUP BY USER
    # -------------------------------------------------

    for pred in predictions:

        uid = pred["user_id"]

        estimated_rating = pred["predicted"]

        true_rating = pred["actual"]

        user_predictions[uid].append(
            (estimated_rating, true_rating)
        )

    precisions = {}
    recalls = {}

    # -------------------------------------------------
    # USER-LEVEL METRICS
    # -------------------------------------------------

    for uid, user_ratings in user_predictions.items():

        user_ratings.sort(
            key=lambda x: x[0],
            reverse=True
        )

        top_k = user_ratings[:k]

        relevant_in_k = sum(
            1 for (_, true_r) in top_k
            if true_r >= threshold
        )

        total_relevant = sum(
            1 for (_, true_r) in user_ratings
            if true_r >= threshold
        )

        precisions[uid] = relevant_in_k / k

        recalls[uid] = (
            relevant_in_k / total_relevant
            if total_relevant > 0
            else 0
        )

    return precisions, recalls