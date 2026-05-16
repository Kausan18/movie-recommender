import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 

DATA_DIR   = ".."         
PLOTS_DIR  = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
 
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
AMBER  = "#BA7517"
BLUE   = "#378ADD"
 
 
# LOAD DATA
def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all four MovieLens CSV files into DataFrames.
    
    Why separate DataFrames?
      Each CSV represents a different *entity* — movies, ratings, tags, links.
      Keeping them separate mirrors good database design (normalised tables)
      and lets us join only what we need, when we need it.
    """
    paths = {
        "movies" : os.path.join(data_dir, "data/raw/movies.csv"),
        "ratings": os.path.join(data_dir, "data/raw/ratings.csv"),
        "tags"   : os.path.join(data_dir, "data/raw/tags.csv"),
        "links"  : os.path.join(data_dir, "data/raw/links.csv"),
    }
    dfs = {}
    for name, path in paths.items():
        dfs[name] = pd.read_csv(path)
        print(f"  Loaded {name:8s} → {dfs[name].shape[0]:>6,} rows × {dfs[name].shape[1]} cols")
    return dfs
 
 
# DATA QUALITY REPORT
def data_quality_report(dfs: dict) -> None:

    print("\n" + "="*60)
    print("  DATA QUALITY REPORT")
    print("="*60)
 
    for name, df in dfs.items():
        nulls   = df.isnull().sum()
        dupes   = df.duplicated().sum()
        has_null = nulls[nulls > 0]
 
        print(f"\n── {name.upper()} ──")
        print(f"  Shape       : {df.shape}")
        print(f"  Duplicates  : {dupes}")
        if has_null.empty:
            print("  Nulls       : none ✓")
        else:
            print(f"  Nulls       :\n{has_null.to_string()}")
        print(f"  dtypes      :\n{df.dtypes.to_string()}")
 
    # ── ratings-specific sanity checks ──────────────────────────
    ratings = dfs["ratings"]
    r_min, r_max = ratings["rating"].min(), ratings["rating"].max()
    print(f"\n  Rating range : {r_min} – {r_max}  (expected 0.5–5.0)")
    assert r_min >= 0.5 and r_max <= 5.0, "Unexpected rating values!"
 
    # ── genres sanity ─────────────────────────────────────────────
    movies = dfs["movies"]
    no_genre = (movies["genres"] == "(no genres listed)").sum()
    print(f"  Movies with no genre : {no_genre}")
 
 
# RATING DISTRIBUTION
 
def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    # --- histogram ---
    counts = ratings["rating"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, width=0.4, color=TEAL, edgecolor="white")
    axes[0].set_title("Rating Distribution", fontweight="bold")
    axes[0].set_xlabel("Star rating")
    axes[0].set_ylabel("Number of ratings")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
    # --- cumulative % ---
    sorted_ratings = ratings["rating"].sort_values()
    cdf = np.arange(1, len(sorted_ratings)+1) / len(sorted_ratings)
    axes[1].plot(sorted_ratings, cdf, color=CORAL, linewidth=2)
    axes[1].set_title("Cumulative Distribution of Ratings", fontweight="bold")
    axes[1].set_xlabel("Star rating")
    axes[1].set_ylabel("Cumulative proportion")
    axes[1].axvline(x=ratings["rating"].median(), color=PURPLE,
                    linestyle="--", linewidth=1.5, label=f"Median = {ratings['rating'].median()}")
    axes[1].legend()
    print("\n" + "="*60)
    print("  RATING DISTRIBUTION INSIGHTS")
    print("="*60)
    print(f"  Total ratings   : {len(ratings):,}")
    print(f"  Mean rating     : {ratings['rating'].mean():.3f}")
    print(f"  Median rating   : {ratings['rating'].median():.1f}")
    print(f"  Std deviation   : {ratings['rating'].std():.3f}")
    print(f"  Most common     : {ratings['rating'].mode()[0]} stars")
    pct_high = (ratings["rating"] >= 4).mean() * 100
    print(f"  Ratings ≥ 4★    : {pct_high:.1f}%  ← 'positive' signal for Precision@K")
 
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_rating_distribution.png", dpi=150)
    plt.close()
    print(f"\n  [saved] {PLOTS_DIR}/01_rating_distribution.png")
 
 
def plot_top_movies(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """
    Most-rated movies and highest-rated movies (with a vote minimum).
    
    Why a vote minimum for 'highest-rated'?
      Without a minimum, obscure films with 1 rating of 5★ would top
      the chart.  We use the IMDb-style filter: only films with at least
      the 70th-percentile number of ratings qualify.  This is the same
      logic we'll use in Phase 4 for the cold-start popularity fallback.
    """
    # ── merge ratings with movie titles ──────────────────────────
    merged = ratings.merge(movies[["movieId", "title"]], on="movieId")
 
    # most-rated (count)
    most_rated = (merged.groupby("title")
                         .agg(num_ratings=("rating", "count"),
                              avg_rating=("rating", "mean"))
                         .sort_values("num_ratings", ascending=False)
                         .head(20))
 
    # highest-rated (filtered)
    MIN_VOTES = merged.groupby("title")["rating"].count().quantile(0.70)
    highest_rated = (merged.groupby("title")
                            .agg(num_ratings=("rating", "count"),
                                 avg_rating=("rating", "mean"))
                            .query("num_ratings >= @MIN_VOTES")
                            .sort_values("avg_rating", ascending=False)
                            .head(20))
 
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
 
    # plot most-rated
    axes[0].barh(most_rated.index[::-1], most_rated["num_ratings"][::-1],
                 color=BLUE, edgecolor="white")
    axes[0].set_title("Top 20 Most-Rated Movies", fontweight="bold")
    axes[0].set_xlabel("Number of ratings")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ["top", "right"]: axes[0].spines[spine].set_visible(False)
 
    # plot highest-rated
    axes[1].barh(highest_rated.index[::-1], highest_rated["avg_rating"][::-1],
                 color=CORAL, edgecolor="white")
    axes[1].set_title(f"Top 20 Highest-Rated Movies\n(min {int(MIN_VOTES)} ratings)", fontweight="bold")
    axes[1].set_xlabel("Average rating")
    axes[1].set_xlim(3.5, 5.0)
    for spine in ["top", "right"]: axes[1].spines[spine].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_top_movies.png", dpi=150)
    plt.close()
    print(f"  [saved] {PLOTS_DIR}/02_top_movies.png")
 
    print("\n" + "="*60)
    print("  TOP 5 MOST-RATED MOVIES")
    print("="*60)
    print(most_rated.head(5).to_string())
    print(f"\n  (vote threshold for 'highest-rated' chart: ≥ {int(MIN_VOTES)} ratings)")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  STEP 5 — GENRE ANALYSIS                                     ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def plot_genre_frequency(movies: pd.DataFrame) -> None:
    """
    Count how often each genre appears across all movies.
    
    Why does genre distribution matter?
      In Phase 2, TF-IDF will *down-weight* genres that appear in too
      many movies (high IDF = rare and distinctive; low IDF = common and
      uninformative).  Knowing that 'Drama' dominates helps us anticipate
      that TF-IDF will naturally penalise it — which is exactly what we want.
    """
    # each movie can have multiple pipe-separated genres
    genre_series = (movies["genres"]
                    .str.split("|")
                    .explode()
                    .str.strip())
    genre_series = genre_series[genre_series != "(no genres listed)"]
    genre_counts = genre_series.value_counts()
 
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                   color=TEAL, edgecolor="white")
    ax.set_title("Genre Frequency Across All Movies", fontweight="bold")
    ax.set_xlabel("Number of movies")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_genre_frequency.png", dpi=150)
    plt.close()
    print(f"  [saved] {PLOTS_DIR}/03_genre_frequency.png")
 
    print("\n" + "="*60)
    print("  TOP 5 GENRES")
    print("="*60)
    print(genre_counts.head(5).to_string())
    print(f"\n  Total unique genres : {len(genre_counts)}")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  STEP 6 — USER ACTIVITY ANALYSIS                             ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def plot_user_activity(ratings: pd.DataFrame) -> None:
    """
    Distribution of ratings-per-user and ratings-per-movie.
    
    Why does this matter?
      Collaborative filtering needs users with *enough* ratings to find
      neighbours.  If 40 % of users rated only 1 movie, the model will
      struggle.  This analysis tells us whether we need a minimum-activity
      filter before training SVD in Phase 3.
      
      The ratings-per-movie distribution also reveals the 'long tail':
      most movies get very few ratings, but a small number of blockbusters
      dominate — this creates the popularity bias we'll measure in Phase 5.
    """
    ratings_per_user  = ratings.groupby("userId")["rating"].count()
    ratings_per_movie = ratings.groupby("movieId")["rating"].count()
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # --- per-user histogram (log y-axis to handle the long tail) ---
    axes[0].hist(ratings_per_user, bins=50, color=PURPLE, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Ratings per User (log scale)", fontweight="bold")
    axes[0].set_xlabel("Number of ratings given")
    axes[0].set_ylabel("Number of users (log scale)")
    axes[0].axvline(ratings_per_user.median(), color=CORAL, linestyle="--",
                    linewidth=1.5, label=f"Median = {int(ratings_per_user.median())}")
    axes[0].legend()
 
    # --- per-movie histogram ---
    axes[1].hist(ratings_per_movie, bins=50, color=AMBER, edgecolor="white")
    axes[1].set_yscale("log")
    axes[1].set_title("Ratings per Movie (log scale)", fontweight="bold")
    axes[1].set_xlabel("Number of ratings received")
    axes[1].set_ylabel("Number of movies (log scale)")
    axes[1].axvline(ratings_per_movie.median(), color=CORAL, linestyle="--",
                    linewidth=1.5, label=f"Median = {int(ratings_per_movie.median())}")
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_user_activity.png", dpi=150)
    plt.close()
    print(f"  [saved] {PLOTS_DIR}/04_user_activity.png")
 
    print("\n" + "="*60)
    print("  USER ACTIVITY INSIGHTS")
    print("="*60)
    print(f"  Total unique users   : {ratings['userId'].nunique():,}")
    print(f"  Total unique movies  : {ratings['movieId'].nunique():,}")
    print(f"  Avg ratings/user     : {ratings_per_user.mean():.1f}")
    print(f"  Median ratings/user  : {ratings_per_user.median():.0f}")
    print(f"  Max ratings by user  : {ratings_per_user.max():,}")
    pct_low = (ratings_per_user < 20).mean() * 100
    print(f"  Users with < 20 ratings : {pct_low:.1f}%  ← potential cold-start users")
    pct_single = (ratings_per_movie == 1).mean() * 100
    print(f"  Movies with only 1 rating : {pct_single:.1f}%  ← long tail")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  STEP 7 — MATRIX SPARSITY                                    ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def analyse_sparsity(ratings: pd.DataFrame) -> None:
    """
    Calculate the sparsity of the user-item rating matrix.
    
    The sparsity metric is central to recommender systems.
    
    The user-item matrix has shape (n_users × n_movies).
    Most cells are empty — a user can't have seen every movie.
    
      Sparsity = 1 - (number of ratings / total possible ratings)
    
    A sparsity of 98.3% means only 1.7% of cells are filled.
    This is *typical* and *expected* for real datasets.  It's why
    plain matrix operations fail (memory + accuracy) and why we need
    matrix factorisation (SVD) — it finds low-rank approximations
    that can fill in the gaps.
    """
    n_users  = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    possible  = n_users * n_movies
    density   = n_ratings / possible
    sparsity  = 1 - density
 
    print("\n" + "="*60)
    print("  MATRIX SPARSITY")
    print("="*60)
    print(f"  Users             : {n_users:,}")
    print(f"  Movies            : {n_movies:,}")
    print(f"  Possible cells    : {possible:,}")
    print(f"  Actual ratings    : {n_ratings:,}")
    print(f"  Density           : {density*100:.2f}%")
    print(f"  Sparsity          : {sparsity*100:.2f}%")
    print()
    print("  Interpretation:")
    print(f"  → {sparsity*100:.1f}% of the matrix is EMPTY.")
    print("  → This is why collaborative filtering can't just look up a table.")
    print("  → SVD (Phase 3) learns to 'fill in' these missing values.")
 
    # ── visual: sample 200x200 corner of the matrix ───────────────
    sample_users  = ratings["userId"].unique()[:200]
    sample_movies = ratings["movieId"].unique()[:200]
    sub = ratings[ratings["userId"].isin(sample_users) &
                  ratings["movieId"].isin(sample_movies)]
    matrix = sub.pivot_table(index="userId", columns="movieId",
                              values="rating", fill_value=0)
    # pad to 200×200 for a clean visual
    matrix = matrix.reindex(index=range(200), columns=range(200), fill_value=0)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.spy(matrix, markersize=1.5, color=TEAL)
    ax.set_title(f"User–Item Matrix (200×200 sample)\nSparsity = {sparsity*100:.1f}%",
                 fontweight="bold")
    ax.set_xlabel("Movie index")
    ax.set_ylabel("User index")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_sparsity_map.png", dpi=150)
    plt.close()
    print(f"\n  [saved] {PLOTS_DIR}/05_sparsity_map.png")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  STEP 8 — RATINGS OVER TIME                                  ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def plot_ratings_over_time(ratings: pd.DataFrame) -> None:
    """
    How did rating activity evolve over time?
    
    Why check this?
      Temporal patterns can reveal:
        - Whether the dataset has a recency bias (newer movies rated more)
        - Natural train/test split boundaries (train on older, test on newer)
        - Seasonal effects (more ratings in winter months)
      In Phase 3, a time-based train/test split is more realistic than
      a random split, because it mimics a real deployment scenario.
    """
    r = ratings.copy()
    r["date"] = pd.to_datetime(r["timestamp"], unit="s")
    r["year_month"] = r["date"].dt.to_period("M")
    monthly = r.groupby("year_month").size()
 
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(monthly.index.astype(str), monthly.values,
                    alpha=0.4, color=BLUE)
    ax.plot(monthly.index.astype(str), monthly.values,
            color=BLUE, linewidth=1.5)
    ax.set_title("Monthly Rating Activity Over Time", fontweight="bold")
    ax.set_ylabel("Number of ratings")
    # only show every 12th tick label to avoid crowding
    ticks = [i for i in range(len(monthly)) if i % 12 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(monthly.index[i]) for i in ticks], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_ratings_over_time.png", dpi=150)
    plt.close()
    print(f"  [saved] {PLOTS_DIR}/06_ratings_over_time.png")
 
    earliest = r["date"].min().strftime("%b %Y")
    latest   = r["date"].max().strftime("%b %Y")
    print("\n" + "="*60)
    print("  TEMPORAL INSIGHTS")
    print("="*60)
    print(f"  Dataset spans : {earliest} → {latest}")
    print(f"  Busiest month : {monthly.idxmax()}  ({monthly.max():,} ratings)")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  STEP 9 — KEY QUESTIONS ANSWERED                             ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def answer_key_questions(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """
    Directly answer the 5 key EDA questions from the project brief.
    """
    print("\n" + "="*60)
    print("  KEY QUESTIONS — ANSWERED")
    print("="*60)
 
    # Q1: Most common rating
    q1 = ratings["rating"].mode()[0]
    print(f"\n  Q1. Most common rating given by users?")
    print(f"      → {q1} ★")
 
    # Q2: % of matrix filled
    n_users  = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    density  = len(ratings) / (n_users * n_movies) * 100
    print(f"\n  Q2. What percentage of the user-item matrix is filled?")
    print(f"      → {density:.2f}%  (sparsity = {100-density:.2f}%)")
 
    # Q3: Top 3 genres
    genre_counts = (movies["genres"]
                    .str.split("|")
                    .explode()
                    .value_counts())
    genre_counts = genre_counts[genre_counts.index != "(no genres listed)"]
    print(f"\n  Q3. Which genres appear most frequently?")
    for g, c in genre_counts.head(3).items():
        print(f"      → {g}: {c:,} movies")
 
    # Q4: Users who rated only 1 movie
    single_raters = (ratings.groupby("userId")["rating"].count() == 1).sum()
    print(f"\n  Q4. Users who rated only 1 movie?")
    print(f"      → {single_raters:,} users")
    print(f"      Recommendation: filter them out for SVD training in Phase 3.")
    print(f"      They provide no collaborative signal — can't find 'similar users'.")
 
    # Q5: Distribution of movie rating counts
    movie_counts = ratings.groupby("movieId")["rating"].count()
    print(f"\n  Q5. Distribution of movie rating counts?")
    print(f"      → Min      : {movie_counts.min()}")
    print(f"      → Median   : {movie_counts.median():.0f}")
    print(f"      → Mean     : {movie_counts.mean():.1f}")
    print(f"      → Max      : {movie_counts.max():,}")
    print(f"      → 75th pct : {movie_counts.quantile(0.75):.0f}")
    print(f"      Interpretation: classic long-tail distribution.")
    print(f"      A small number of popular films get most ratings.")
 
 
# ╔═══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                        ║
# ╚═══════════════════════════════════════════════════════════════╝
 
def main():
    print("="*60)
    print("  PHASE 1 — EDA  |  MovieLens Latest Small")
    print("="*60)
 
    # 1. Load
    print("\n[1/8] Loading CSVs...")
    dfs = load_data(DATA_DIR)
 
    # 2. Quality
    print("\n[2/8] Running data quality checks...")
    data_quality_report(dfs)
 
    # 3. Rating distribution
    print("\n[3/8] Plotting rating distribution...")
    plot_rating_distribution(dfs["ratings"])
 
    # 4. Top movies
    print("\n[4/8] Plotting top movies...")
    plot_top_movies(dfs["movies"], dfs["ratings"])
 
    # 5. Genre frequency
    print("\n[5/8] Plotting genre frequency...")
    plot_genre_frequency(dfs["movies"])
 
    # 6. User activity
    print("\n[6/8] Plotting user activity...")
    plot_user_activity(dfs["ratings"])
 
    # 7. Sparsity
    print("\n[7/8] Analysing matrix sparsity...")
    analyse_sparsity(dfs["ratings"])
 
    # 8. Temporal
    print("\n[8/8] Plotting ratings over time...")
    plot_ratings_over_time(dfs["ratings"])
 
    # Key questions
    answer_key_questions(dfs["movies"], dfs["ratings"])
 
    print("\n" + "="*60)
    print("  EDA COMPLETE  →  6 charts saved to ./eda_plots/")
    print("  Next: Phase 2 — Content-Based Filtering")
    print("="*60)
 
 
if __name__ == "__main__":
    main()