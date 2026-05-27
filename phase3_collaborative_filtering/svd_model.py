"""
Collaborative filtering using scipy truncated SVD.
Production implementation — no scikit-surprise dependency.

Key design:
- Build a sparse CSR matrix (memory-efficient, scipy-native)
- Mean-centre each user's ratings before decomposition
- Use svds() for truncated SVD — only computes top-k factors
- Reconstruct predicted ratings on demand
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "svd_model.pkl")
RATINGS_PATH = os.path.join(ROOT_DIR, "data", "raw", "ratings.csv")
MOVIES_PATH = os.path.join(ROOT_DIR, "data", "raw", "movies.csv")


# ─────────────────────────────────────────────────────────── #
#  Matrix construction
# ─────────────────────────────────────────────────────────── #

def build_sparse_matrix(ratings_df: pd.DataFrame):
    """
    Build a sparse CSR user-item matrix from a ratings DataFrame.

    Returns:
        matrix       : scipy sparse CSR matrix (n_users × n_movies)
        user_index   : dict {userId → row index}
        movie_index  : dict {movieId → col index}
        user_means   : np.array of mean rating per user (for de-centering)
    """
    users = sorted(ratings_df['userId'].unique())
    movies = sorted(ratings_df['movieId'].unique())

    user_index = {uid: i for i, uid in enumerate(users)}
    movie_index = {mid: j for j, mid in enumerate(movies)}

    rows = ratings_df['userId'].map(user_index).values
    cols = ratings_df['movieId'].map(movie_index).values
    vals = ratings_df['rating'].values.astype(float)

    matrix = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(users), len(movies))
    )

    row_sums = np.array(matrix.sum(axis=1)).flatten()
    row_counts = np.diff(matrix.indptr)
    user_means = np.where(row_counts > 0, row_sums / row_counts, 0.0)

    return matrix, user_index, movie_index, user_means


def mean_centre_matrix(matrix: sp.csr_matrix, user_means: np.ndarray) -> sp.csr_matrix:
    """
    Subtract each user's mean rating from their observed ratings.
    Operates only on non-zero entries — does not fill missing values.
    """
    matrix_centred = matrix.copy().astype(float)
    for i, mean in enumerate(user_means):
        start = matrix_centred.indptr[i]
        end = matrix_centred.indptr[i + 1]
        matrix_centred.data[start:end] -= mean
    return matrix_centred


# ─────────────────────────────────────────────────────────── #
#  Training
# ─────────────────────────────────────────────────────────── #

def train_svd(
    ratings_df: pd.DataFrame,
    k: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Train SVD collaborative filtering model.

    Steps:
      1. Split ratings into train/test
      2. Build sparse matrix from train set
      3. Mean-centre the matrix
      4. Run truncated SVD (top-k factors)
      5. Store U, sigma, Vt + mappings for inference

    Args:
        ratings_df   : full ratings DataFrame (userId, movieId, rating)
        k            : number of latent factors (try 50, 100, 150)
        test_size    : fraction held out for evaluation
        random_state : reproducibility seed

    Returns:
        model dict with keys: U, sigma, Vt, user_index, movie_index,
                              user_means, k, train_df, test_df
    """
    train_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(train_df):,} ratings | Test: {len(test_df):,} ratings")

    matrix, user_index, movie_index, user_means = build_sparse_matrix(train_df)
    matrix_centred = mean_centre_matrix(matrix, user_means)

    print(f"Matrix shape: {matrix.shape[0]} users × {matrix.shape[1]} movies")
    print(f"Running truncated SVD with k={k} factors...")

    U, sigma, Vt = svds(matrix_centred.astype(float), k=k)
    U = U[:, ::-1]
    sigma = sigma[::-1]
    Vt = Vt[::-1, :]

    print(f"SVD complete. Top singular value: {sigma[0]:.2f} | k={k}")

    model = {
        "U": U,
        "sigma": sigma,
        "Vt": Vt,
        "user_index": user_index,
        "movie_index": movie_index,
        "user_means": user_means,
        "k": k,
        "train_df": train_df,
        "test_df": test_df,
    }
    return model


# ─────────────────────────────────────────────────────────── #
#  Prediction
# ─────────────────────────────────────────────────────────── #

def predict_rating(model: dict, user_id: int, movie_id: int) -> float:
    """
    Predict a single user's rating for a single movie.

    Returns:
        Predicted rating clipped to [0.5, 5.0].
        Returns the global mean (3.5) if user/movie not in training data.
    """
    user_index = model['user_index']
    movie_index = model['movie_index']

    if user_id not in user_index or movie_id not in movie_index:
        return 3.5

    u = user_index[user_id]
    m = movie_index[movie_id]

    pred = model['U'][u] @ np.diag(model['sigma']) @ model['Vt'][:, m]
    pred += model['user_means'][u]

    return float(np.clip(pred, 0.5, 5.0))


# ─────────────────────────────────────────────────────────── #
#  Evaluation
# ─────────────────────────────────────────────────────────── #

def evaluate_model(model: dict, ratings_df: pd.DataFrame) -> dict:
    """
    Evaluate RMSE and MAE on the test set.
    Also computes the naive baseline (always predict global mean).

    Prints a comparison table. Returns metrics dict.
    """
    test_df = model['test_df']
    global_mean = ratings_df['rating'].mean()

    actual = []
    predicted = []
    baseline = []

    for _, row in test_df.iterrows():
        pred = predict_rating(model, int(row['userId']), int(row['movieId']))
        actual.append(row['rating'])
        predicted.append(pred)
        baseline.append(global_mean)

    actual = np.array(actual)
    predicted = np.array(predicted)
    baseline = np.array(baseline)

    rmse_model = np.sqrt(np.mean((actual - predicted) ** 2))
    mae_model = np.mean(np.abs(actual - predicted))
    rmse_baseline = np.sqrt(np.mean((actual - baseline) ** 2))
    mae_baseline = np.mean(np.abs(actual - baseline))

    improvement = (rmse_baseline - rmse_model) / rmse_baseline * 100

    print("\n" + "=" * 52)
    print("MODEL EVALUATION")
    print("=" * 52)
    print(f"{'Metric':<20} {'SVD Model':<16} {'Baseline (mean)'}")
    print("-" * 52)
    print(f"{'RMSE':<20} {rmse_model:<16.4f} {rmse_baseline:.4f}")
    print(f"{'MAE':<20} {mae_model:<16.4f} {mae_baseline:.4f}")
    print("=" * 52)
    print(f"RMSE improvement over baseline: {improvement:.1f}%")
    print(f"Global mean rating: {global_mean:.4f}")

    return {
        "rmse": rmse_model,
        "mae": mae_model,
        "baseline_rmse": rmse_baseline,
        "baseline_mae": mae_baseline,
        "improvement_pct": improvement,
    }


# ─────────────────────────────────────────────────────────── #
#  Recommendations
# ─────────────────────────────────────────────────────────── #

def get_recommendations(
    user_id: int,
    model: dict,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Return top-N movie recommendations for a user.

    Strategy:
      1. Find all movies this user has NOT rated.
      2. Predict their rating for each using SVD.
      3. Return top-N sorted by predicted rating descending.

    Args:
        user_id    : integer userId from the dataset
        model      : trained model dict from train_svd()
        ratings_df : full ratings DataFrame
        movies_df  : movies DataFrame (movieId, title, genres)
        n          : number of recommendations

    Returns:
        DataFrame with columns: movieId, title, genres, predicted_rating
    """
    if user_id not in model['user_index']:
        raise ValueError(
            f"User {user_id} was not in the training set. "
            f"Use the cold-start handler (Phase 4) for new users."
        )

    rated_movie_ids = set(
        ratings_df[ratings_df['userId'] == user_id]['movieId'].values
    )
    all_movie_ids = list(model['movie_index'].keys())
    unrated = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    u = model['user_index'][user_id]
    u_vec = model['U'][u] * model['sigma']

    col_indices = [model['movie_index'][mid] for mid in unrated]
    movie_vecs = model['Vt'][:, col_indices]

    scores = u_vec @ movie_vecs + model['user_means'][u]
    scores = np.clip(scores, 0.5, 5.0)

    top_indices = np.argsort(scores)[::-1][:n]
    top_movie_ids = [unrated[i] for i in top_indices]
    top_scores = {mid: float(scores[i]) for mid, i in zip(top_movie_ids, top_indices)}

    result = movies_df[movies_df['movieId'].isin(top_movie_ids)].copy()
    result['predicted_rating'] = result['movieId'].map(top_scores)
    result = result.sort_values('predicted_rating', ascending=False).reset_index(drop=True)

    return result[['movieId', 'title', 'genres', 'predicted_rating']]


# ─────────────────────────────────────────────────────────── #
#  Persistence
# ─────────────────────────────────────────────────────────── #

def save_model(model: dict, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {k: v for k, v in model.items() if k not in ('train_df', 'test_df')}
    joblib.dump(save_dict, path)
    print(f"Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path)
