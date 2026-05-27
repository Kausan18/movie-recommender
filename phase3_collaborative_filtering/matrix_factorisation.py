"""
SVD from scratch using numpy — for understanding only.
Not used in the production pipeline.
Shows the raw decomposition on a small subset of the matrix.
"""
import numpy as np
import pandas as pd

def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot ratings into a dense user × movie matrix.
    Missing ratings are filled with 0.
    Returns DataFrame of shape (n_users, n_movies).
    """
    matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    return matrix


def compute_sparsity(matrix: pd.DataFrame) -> float:
    """
    Prints and returns the density (fraction of filled cells) of the matrix.
    For MovieLens 100K, expect ~1.7% density.
    """
    total = matrix.shape[0] * matrix.shape[1]
    non_zero = np.count_nonzero(matrix.values)
    density = non_zero / total
    print(f"Matrix shape:  {matrix.shape[0]} users × {matrix.shape[1]} movies")
    print(f"Total cells:   {total:,}")
    print(f"Filled cells:  {non_zero:,}")
    print(f"Density:       {density:.2%}  |  Sparsity: {1-density:.2%}")
    return density


def numpy_svd_decompose(matrix: pd.DataFrame, k: int = 50):
    """
    Decompose the user-item matrix using full numpy SVD.
    Keep only the top-k latent factors.
    EDUCATIONAL ONLY — do not use in production (too slow on full matrix).

    Returns U_k, sigma_k, Vt_k, R_hat (reconstructed matrix with predictions filled in).
    """
    R = matrix.values.astype(float)

    user_means = np.where(
        (R != 0).sum(axis=1) > 0,
        R.sum(axis=1) / np.maximum((R != 0).sum(axis=1), 1),
        0
    )
    R_centered = R.copy()
    for i, mean in enumerate(user_means):
        mask = R[i] != 0
        R_centered[i][mask] -= mean

    U, sigma, Vt = np.linalg.svd(R_centered, full_matrices=False)

    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k = Vt[:k, :]

    R_hat = U_k @ sigma_k @ Vt_k
    R_hat += user_means[:, np.newaxis]

    print(f"SVD complete. Kept {k} of {len(sigma)} factors.")
    print(f"Variance explained: {sigma[:k].sum() / sigma.sum():.2%}")

    return U_k, sigma_k, Vt_k, R_hat


if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    top_users = ratings['userId'].value_counts().head(100).index
    top_movies = ratings['movieId'].value_counts().head(500).index
    subset = ratings[ratings['userId'].isin(top_users) & ratings['movieId'].isin(top_movies)]

    matrix = build_user_item_matrix(subset)
    compute_sparsity(matrix)
    _, _, _, R_hat = numpy_svd_decompose(matrix, k=20)
    print("\nR_hat shape:", R_hat.shape)
    print("Sample predicted ratings (row 0, first 10 movies):", R_hat[0, :10].round(2))
