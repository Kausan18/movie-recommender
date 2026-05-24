"""
Phase 4 — User Clustering
Clusters users by their genre preferences using K-Means on a user-genre rating matrix.
Provides cluster assignment, PCA visualisation, and elbow curve diagnostics.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ────────────────────────────────────────────────────────────────────────────
#  Function 1 — build_user_genre_matrix
# ────────────────────────────────────────────────────────────────────────────

def build_user_genre_matrix(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a matrix of (userId × genre) with mean ratings.

    Steps:
        1. Explode pipe-separated genres into individual rows
        2. Merge with ratings to get per-user-per-genre ratings
        3. Pivot to wide format with NaN filled by column means

    Returns:
        DataFrame with index=userId, columns=genre names, values=mean ratings.
        Shape is approximately (610, 19) for the full MovieLens dataset.
    """
    # 1. Copy and filter out "(no genres listed)"
    genre_df = movies_df[["movieId", "genres"]].copy()
    genre_df = genre_df[genre_df["genres"] != "(no genres listed)"]

    # 2. Split and explode genres
    genre_df = genre_df.assign(genre=genre_df["genres"].str.split("|")).explode("genre")
    genre_df = genre_df[["movieId", "genre"]]

    # 3. Merge with ratings
    merged = genre_df.merge(
        ratings_df[["userId", "movieId", "rating"]],
        on="movieId",
        how="inner"
    )

    # 4. Group by userId and genre, compute mean rating
    grouped = merged.groupby(["userId", "genre"])["rating"].mean()

    # 5. Pivot to wide format
    matrix = grouped.unstack(fill_value=np.nan)

    # 6. Fill NaN with column means
    matrix = matrix.fillna(matrix.mean())

    return matrix


# ────────────────────────────────────────────────────────────────────────────
#  Function 2 — train_kmeans
# ────────────────────────────────────────────────────────────────────────────

def train_kmeans(
    user_genre_matrix: pd.DataFrame,
    k: int = 5,
    random_state: int = 42
) -> tuple:
    """
    Train K-Means on the user-genre matrix after standard scaling.

    Returns:
        (scaler, kmeans, labels) — the fitted StandardScaler, KMeans model,
        and cluster labels for each user.
    """
    # 1. Scale features
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(user_genre_matrix.values)

    # 2. Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(scaled_matrix)

    # 3. Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/kmeans_scaler.pkl")
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    return scaler, kmeans, kmeans.labels_


# ────────────────────────────────────────────────────────────────────────────
#  Function 3 — plot_clusters_pca
# ────────────────────────────────────────────────────────────────────────────

def plot_clusters_pca(
    user_genre_matrix: pd.DataFrame,
    labels: np.ndarray,
    scaler: StandardScaler,
    save_path: str = "phase4_hybrid_coldstart/cluster_plot.png"
) -> None:
    """
    Visualise user clusters in 2D using PCA on the scaled genre matrix.
    Saves the plot to disk (no plt.show()).
    """
    # 1. Scale the matrix
    scaled = scaler.transform(user_genre_matrix.values)

    # 2. PCA to 2 components
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)

    # 3. Plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab10", s=15, alpha=0.7
    )
    plt.title("User clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved cluster plot to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
#  Function 4 — describe_clusters
# ────────────────────────────────────────────────────────────────────────────

def describe_clusters(
    user_genre_matrix: pd.DataFrame,
    labels: np.ndarray,
    k: int = 5
) -> dict:
    """
    Describe each cluster by its top 3 genres (most above global average).

    Returns:
        dict {cluster_id: [genre1, genre2, genre3]}
    """
    df = user_genre_matrix.copy()
    df["cluster"] = labels

    global_means = user_genre_matrix.mean()

    result = {}
    for i in range(k):
        sub_df = df[df["cluster"] == i]
        cluster_means = sub_df.drop(columns="cluster").mean()
        delta = cluster_means - global_means
        top3 = delta.nlargest(3)

        genre_names = list(top3.index)
        result[i] = genre_names

        genre_str = ", ".join(
            f"{g} ({top3[g]:+.2f})" for g in genre_names
        )
        print(f"  Cluster {i} ({len(sub_df)} users): {genre_str}")

    return result


# ────────────────────────────────────────────────────────────────────────────
#  Function 5 — plot_elbow_curve
# ────────────────────────────────────────────────────────────────────────────

def plot_elbow_curve(
    user_genre_matrix: pd.DataFrame,
    scaler: StandardScaler,
    k_range: range = range(2, 11),
    save_path: str = "phase4_hybrid_coldstart/elbow_curve.png"
) -> None:
    """
    Plot the elbow curve (inertia vs k) for K-Means cluster selection.
    This is a diagnostic tool — it refits K-Means multiple times.
    """
    scaled = scaler.transform(user_genre_matrix.values)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", linewidth=2)
    plt.title("Elbow curve")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved elbow curve to {save_path}")


# ────────────────────────────────────────────────────────────────────────────
#  Function 6 — assign_user_to_cluster
# ────────────────────────────────────────────────────────────────────────────

def assign_user_to_cluster(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans
) -> int:
    """
    Assign a single user to a cluster.

    Builds a genre vector for the user, scales it using the training scaler,
    and predicts the cluster label.

    Returns:
        Cluster ID (int), or -1 if the user has no ratings.
    """
    # Build the full matrix to get the reference columns
    full_matrix = build_user_genre_matrix(ratings_df, movies_df)

    # Build single-user matrix
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    if user_ratings.empty:
        return -1

    user_matrix = build_user_genre_matrix(user_ratings, movies_df)

    if user_matrix.empty:
        return -1

    # Reindex to match full matrix columns, fill missing with column means
    user_vector = user_matrix.reindex(columns=full_matrix.columns, fill_value=np.nan)
    user_vector = user_vector.fillna(full_matrix.mean())

    # Scale and predict
    scaled = scaler.transform(user_vector.values)
    return int(kmeans.predict(scaled)[0])


# ────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ratings = pd.read_csv("data/raw/ratings.csv")
    movies = pd.read_csv("data/raw/movies.csv")

    print("Building user-genre matrix...")
    matrix = build_user_genre_matrix(ratings, movies)
    print(f"  Matrix shape: {matrix.shape}")

    print("Running elbow curve (saves to phase4_hybrid_coldstart/elbow_curve.png)...")
    # Need a scaler first just for the elbow plot
    temp_scaler = StandardScaler()
    temp_scaler.fit(matrix.values)
    plot_elbow_curve(matrix, temp_scaler)

    print("Training K-Means (k=5)...")
    scaler, kmeans, labels = train_kmeans(matrix, k=5)
    print(f"  Cluster counts: {dict(zip(*np.unique(labels, return_counts=True)))}")

    print("Cluster descriptions:")
    describe_clusters(matrix, labels, k=5)

    print("Saving PCA plot...")
    plot_clusters_pca(matrix, labels, scaler)

    print("Assigning user 42 to cluster...")
    c = assign_user_to_cluster(42, ratings, movies, scaler, kmeans)
    print(f"  User 42 -> Cluster {c}")
