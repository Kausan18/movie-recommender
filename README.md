# 🎬 Movie Recommender System

A full-stack, production-ready movie recommendation engine built in Python — featuring content-based filtering, collaborative filtering via SVD, a hybrid model, cold-start handling, LLM-generated explanations, a FastAPI backend, and a Streamlit frontend. Built on the **MovieLens 100K dataset** (100,836 ratings, 9,742 movies, 610 users).

> Built as an end-to-end ML systems project to demonstrate skills in recommendation algorithms, NLP, API design, and cloud deployment.

---

## 🔗 Live Demo

| Service | URL |
|---|---|
| App URL| https://movie-recommender-kausty.streamlit.app |


## 🗂️ Project Structure

```
movie-recommender/
│
├── phase1_eda/                      # Exploratory Data Analysis
├── phase2_content_based/            # TF-IDF content-based filtering
├── phase3_collaborative_filtering/  # SVD matrix factorisation
├── phase4_hybrid_coldstart/         # Hybrid model + cold-start routing
├── phase5_evaluation/               # RMSE, Precision@K, A/B test
├── phase6_api/                      # FastAPI REST backend
├── phase7_frontend/                 # Streamlit UI
├── phase8_deploy/                   # Docker + deployment config
│
├── data/
│   ├── raw/                         # MovieLens source files
│   └── processed/                   # Cleaned CSVs, popularity scores
│
├── models/                          # Saved .pkl artefacts
├── utils/                           # API client, LLM explainer, metrics
└── requirements.txt
```

---

## 🧠 How the Recommendation System Works

This project implements **three distinct recommendation strategies** that are automatically selected based on how much is known about the user.

### 1. Content-Based Filtering (Phase 2)

**The idea:** Recommend movies that are *similar* to ones you already like, based on their content — genres, tags, and title.

**How it works:**
1. A "soup" string is built for every movie by combining its genres (weighted ×3 since every movie has them), title tokens, release year, and user-written community tags. The genres are repeated three times to ensure they dominate the similarity signal — a movie's genre is the most reliable content descriptor.
2. A **TF-IDF vectoriser** (Term Frequency–Inverse Document Frequency) converts these soup strings into sparse numerical vectors with up to 8,000 features. TF-IDF gives higher weight to terms that appear often in a movie but rarely across the whole catalogue — so a genre like "Documentary" means more for a niche film than "Drama" does for a mainstream one. Bigrams (two-word phrases like "science fiction") are also captured.
3. At query time, **cosine similarity** is computed between one query movie's vector and all others using `linear_kernel` — mathematically identical to cosine similarity on L2-normalised vectors, but faster. The full pairwise similarity matrix is never stored; similarity is computed on-the-fly per request.
4. The top-N most similar movies are returned, excluding the query movie itself.

**Key design choice:** The cosine similarity matrix is never pre-computed and saved. Even at 9,700 movies this would be a dense 9,700 × 9,700 matrix that grows quadratically. Instead, only the sparse TF-IDF matrix (~340 KB) is saved, and a single row is compared against all others at request time — taking ~50ms.

**Fuzzy title search:** The recommender uses word-boundary normalisation (strips year, punctuation, and case) so queries like `"avengers age of ultron"` correctly match `"Avengers: Age of Ultron (2015)"` without returning false matches like `"Scavengers"`.

---

### 2. Collaborative Filtering via SVD (Phase 3)

**The idea:** Recommend movies that users *similar to you* enjoyed — without needing to know anything about the movies themselves. The algorithm learns hidden patterns purely from the history of who rated what.

**How it works:**
1. All 100,836 ratings are arranged into a **sparse user-item matrix** (610 users as rows, 9,742 movies as columns, ratings as values). With only 100,836 entries in a 610 × 9,742 = ~5.9 million cell grid, the matrix is ~98.3% empty — this is called **sparsity**, and it's the core challenge of collaborative filtering.
2. Each user's **mean rating is subtracted** from their entries (mean-centering). This removes individual rating bias — a user who rates everything 4–5 stars and one who rates everything 1–2 stars can still have the same *relative* preferences. Without this step, SVD would model "generous rater" vs "harsh rater" rather than actual taste.
3. **Truncated SVD** (Singular Value Decomposition) decomposes the centred matrix into three matrices: U (users × latent factors), Σ (singular values), and Vᵀ (latent factors × movies). With `k=100` latent factors, the algorithm compresses the entire rating history into 100 hidden dimensions — things like "this user likes 90s sci-fi" or "this user prefers critically acclaimed dramas" — without ever needing explicit labels for those patterns.
4. A predicted rating for any user-movie pair is reconstructed as: `U[user] · Σ · Vᵀ[:, movie] + user_mean`, clipped to [0.5, 5.0].
5. For a given user, all movies they *haven't* rated are scored in a single vectorised operation and the top-N are returned.

**Why scipy's `svds()` instead of `numpy.linalg.svd()`?** `svds()` computes only the top-k singular values/vectors (truncated decomposition). Full SVD on even the 100K matrix would compute all 610 factors — wasteful and slow. `svds()` keeps only the k=100 most important ones.

**Educational reference:** `phase3_collaborative_filtering/matrix_factorisation.py` contains a from-scratch numpy implementation of the same algorithm on a small subset — written purely for understanding, not used in production.

**Model evaluation results:**

| Metric | Baseline (predict global mean) | SVD Model |
|---|---|---|
| RMSE | 1.0425 | **0.9483** |
| MAE | — | **0.7353** |
| Precision@10 | — | **51.4%** |
| Recall@10 | — | **65.1%** |

The SVD model achieves a **9.0% RMSE improvement** over the global mean baseline.

---

### 3. Hybrid Model (Phase 4)

**The idea:** Combine content-based and collaborative filtering scores into one unified ranking to get the best of both worlds — content diversity and personalised relevance.

**How it works:**
1. **Content candidates**: The user's top-10 rated movies are used as seeds. For each seed, `get_similar_movies()` returns 200 content-similar movies, weighted by the seed's rating (a 5-star seed contributes more than a 3-star one) and aggregated by mean across all seeds.
2. **CF candidates**: SVD predicts ratings for 200 movies the user hasn't seen yet.
3. Both score sets are independently **sigmoid-normalised** to [0, 1] using mean-centering and z-score scaling. This is necessary because cosine similarity scores (roughly 0.0–0.3) and SVD predicted ratings (0.5–5.0) are on completely different scales — direct blending would let one dominate the other.
4. Scores are **blended** with equal weight: `hybrid = 0.5 × content_norm + 0.5 × cf_norm`. Movies that appear in only one candidate set are penalised by 10% to favour movies both models agree on.
5. The top-N blended results are returned with all three sub-scores (content, CF, hybrid) exposed in the API response for transparency.

---

### 4. Cold-Start Handling & Routing (Phase 4)

The **cold-start problem** is when you have little or no information about a new user. The system handles it with a three-branch routing strategy in `cold_start.route_recommendation()`:

| User rating count | Strategy | Reasoning |
|---|---|---|
| 0 ratings | Popularity (IMDb weighted formula) | No signal at all — show crowd favourites |
| 1–4 ratings | Content-based (seeded from highest-rated movie) | Too sparse for CF; content features are reliable with even one data point |
| 5+ ratings | Full hybrid (content + SVD) | Enough rating history for both models to contribute meaningfully |

The **popularity formula** uses the IMDb Bayesian weighted rating:

```
weighted_score = (v / (v + m)) × R + (m / (v + m)) × C
```

where `v` = movie's vote count, `R` = movie's average rating, `m` = 50th-percentile vote threshold, and `C` = global mean rating. This prevents a movie with 1 five-star rating from outranking a classic with thousands of ratings.

---

### 5. User Clustering (Phase 4)

Users are clustered by their genre preferences using K-Means on a **user-genre rating matrix**:

1. Each user's mean rating per genre is computed, forming a vector of ~19 genre dimensions.
2. Features are **StandardScaler**-normalised so no genre dominates by scale.
3. K-Means with `k=5` (selected via elbow curve) groups users into taste profiles — e.g. action/thriller fans, drama fans, animation enthusiasts.
4. PCA reduces the genre space to 2D for visualisation of cluster separation.

This enables cluster-aware cold-start: a new user can be assigned to the nearest cluster and shown movies popular within that group.

---

### 6. LLM-Generated Explanations (Phase 7)

After recommendations are generated, the app optionally calls the **Groq API** (running `llama-3.3-70b-versatile`) to generate a one-sentence natural language explanation for each recommendation. The prompt is engineered to:

- Reference the user's actual watch history and the movie's genres
- Stay under 25 words
- Avoid generic filler phrases like "you might enjoy" or "based on your history"
- Distinguish between content-based, hybrid, and popularity recommendation methods

Explanations are generated after recommendations arrive and never block the UI. If the Groq API is unavailable, a genre-based fallback string is shown instead.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────┐
│              Streamlit Frontend              │
│  (phase7_frontend/)                          │
│  ┌───────────────────┐  ┌─────────────────┐  │
│  │    Home Page      │  │   Dashboard     │  │
│  │  (title search    │  │  (EDA plots +   │  │
│  │   + user ID)      │  │   metrics)      │  │
│  └────────┬──────────┘  └────────┬────────┘  │
│           └──────────┬───────────┘           │
│                      │ HTTP (requests)       │
└──────────────────────┼───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│              FastAPI Backend                 │
│  (phase6_api/)        PORT 8000              │
│                                              │
│  GET /recommend/popular                      │
│  GET /recommend/movie/{title}                │
│  GET /recommend/user/{user_id}               │
│  GET /recommend/search?q=                    │
│  GET /health                                 │
└───────┬──────────────────┬───────────────────┘
        │                  │
┌───────▼──────┐  ┌────────▼──────────────────┐
│ Content      │  │ Hybrid + Cold Start        │
│ Model        │  │ (SVD + Popularity +        │
│ (TF-IDF +    │  │  K-Means Clustering)       │
│ cosine sim)  │  └────────────────────────────┘
└──────────────┘
        │
┌───────▼──────────────────────────────────────┐
│           MovieLens 100K Dataset             │
│  movies.csv · ratings.csv · tags.csv        │
│  genome-tags.csv · links.csv                │
└──────────────────────────────────────────────┘
```

---

## 📊 Dataset

**MovieLens 100K** — published by GroupLens Research, University of Minnesota.

| Stat | Value |
|---|---|
| Ratings | 100,836 |
| Movies | 9,742 |
| Users | 610 |
| Tags | ~3,683 |
| Rating scale | 0.5 – 5.0 stars |
| Time span | March 1996 – September 2018 |

> The project was architected to also support the full **MovieLens 25M** dataset (25M ratings, 62K movies) — the content model uses genome tag integration and `svds()` truncated SVD specifically to handle that scale. The 100K version is used for deployment due to GitHub file size limits.

Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4).

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML / NLP | scikit-learn (TF-IDF, K-Means, PCA, StandardScaler)|
| Data | pandas, numpy |
| API | FastAPI, uvicorn, pydantic |
| Frontend | Streamlit, plotly |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Movie posters | TMDB API |
| Serialisation | joblib |
| Deployment | Render (backend), Streamlit Cloud (frontend), Docker |

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)
- A [TMDB API key](https://www.themoviedb.org/settings/api) (free, for movie posters)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
# Fill in GROQ_API_KEY and TMDB_KEY in .env
```

### Run

```bash
# Terminal 1 — FastAPI backend
uvicorn phase6_api.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run phase7_frontend/app.py
```

Open `http://localhost:8501` in your browser. The FastAPI Swagger UI is at `http://localhost:8000/docs`.

---

## 🌐 Deployment

**Backend → Render**, **Frontend → Streamlit Cloud**. Both services pull from the same GitHub repo.

See `phase8_deploy/DEPLOY.md` for step-by-step instructions.

| Variable | Where to set | What it does |
|---|---|---|
| `GROQ_API_KEY` | Render + Streamlit Cloud secrets | Powers LLM explanations |
| `TMDB_KEY` | Render + Streamlit Cloud secrets | Fetches movie poster images |
| `API_BASE_URL` | Streamlit Cloud secrets only | Points frontend to Render backend URL |

---

## 📈 Model Performance

### Evaluation Results

| Metric | Baseline (global mean) | SVD Collaborative Filtering |
|---|---|---|
| RMSE | 1.0425 | **0.9483** (↓ 9.0%) |
| MAE | — | **0.7353** |
| Precision@10 | — | **51.4%** |
| Recall@10 | — | **65.1%** |

### A/B Test: Content-Based vs. SVD Collaborative

A simulated A/B test on 304 users per group compared Precision@10:

| Model | Mean Precision@10 | Std Dev |
|---|---|---|
| Content-Based | 0.69% | ±2.78% |
| SVD Collaborative | **18.85%** | ±14.11% |

Statistically significant difference (t = −21.98, p ≈ 0.0). SVD wins decisively for users with rating history, which is why the hybrid model uses SVD as its collaborative filtering backbone.

### Why use a Hybrid at all?

Content-based alone suffers from **over-specialisation** — it keeps recommending the same genre because it can only see what a movie *is*, not what people *feel* about it. SVD alone suffers from **popularity bias** — it gravitates toward widely-rated movies because they have more signal in the matrix. The hybrid addresses both: content diversity keeps recommendations varied, SVD personalisation keeps them relevant to the individual.

---

## 🔍 Key Engineering Decisions

**On-the-fly cosine similarity instead of a pre-computed matrix** — Storing the full pairwise similarity matrix grows as O(n²). Even at 9,700 movies it's a ~375 MB dense float matrix. Instead, the sparse TF-IDF matrix (~340 KB) is stored and `linear_kernel` computes one row's similarities in ~50ms at request time. This approach was designed to scale cleanly to 62K movies (MovieLens 25M) without any architectural change.

**scipy `svds()` instead of `numpy.linalg.svd()`** — Truncated SVD computes only the top-k latent factors needed for prediction. Full SVD computes all factors — on a 610 × 9,742 matrix that's wasteful; on the 25M dataset (162K × 62K) it would be computationally impossible. `svds()` is the production-correct choice at any scale.

**Mean-centering before SVD** — Without subtracting each user's mean rating, the decomposition models rating *magnitude* (generous vs. harsh raters) rather than *relative preference*. Mean-centering shifts the focus to what a user liked more or less than their own average, which is the actual signal useful for recommendations.

**Sigmoid normalisation for hybrid blending** — Cosine similarity scores (typically 0.0–0.3 for this dataset) and SVD predicted ratings (0.5–5.0) are on completely different scales. Direct blending would mean CF scores dominate. Sigmoid normalisation with z-score centering maps both to [0, 1] around their respective means without hard-clamping outliers.

**FastAPI lifespan for model loading** — All ML artefacts (TF-IDF matrix, SVD model, K-Means model, DataFrames) are loaded once at server startup via the lifespan context manager. Every recommendation endpoint is served from in-memory objects — zero file I/O per request.

**Thread-safe lazy loading in content recommender** — The `ContentRecommender` class uses a `threading.Lock()` to ensure the TF-IDF matrix is loaded only once even under concurrent Streamlit requests. This is important because Streamlit reruns scripts on every user interaction.

---

## 📁 Notable Files

| File | What it does |
|---|---|
| `phase2_content_based/content_model.py` | Builds TF-IDF soup, trains and saves vectoriser + sparse matrix |
| `phase2_content_based/recommend.py` | Thread-safe lazy-loading recommender class with fuzzy title search |
| `phase3_collaborative_filtering/svd_model.py` | Sparse matrix construction, mean-centering, truncated SVD, prediction |
| `phase3_collaborative_filtering/matrix_factorisation.py` | From-scratch numpy SVD — educational reference only |
| `phase4_hybrid_coldstart/cold_start.py` | IMDb weighted popularity + 3-branch routing logic |
| `phase4_hybrid_coldstart/hybrid_model.py` | Score blending with sigmoid normalisation |
| `phase4_hybrid_coldstart/clustering.py` | K-Means user clustering on genre-preference vectors |
| `phase6_api/main.py` | FastAPI entry point with lifespan model loading |
| `phase6_api/routes/recommend.py` | All recommendation endpoints with dependency injection |
| `utils/llm_explainer.py` | Groq API calls for natural language explanations |
| `utils/api_client.py` | All HTTP calls from Streamlit to FastAPI (no direct requests in UI) |

---

## 🧪 Evaluation Metrics Explained

**RMSE (Root Mean Squared Error)** — How far predicted ratings are from actual ratings on average. Penalises large errors more than small ones. Lower is better. The SVD model achieved 0.9483 vs. 1.0425 for the naive baseline.

**MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual ratings. Less sensitive to outliers than RMSE. Achieved 0.7353 — meaning predictions are off by less than 0.75 stars on average.

**Precision@K** — Of the top-K movies recommended, what fraction did the user actually rate ≥ 4.0? Measures recommendation quality. Achieved 51.4% at K=10.

**Recall@K** — Of all movies a user rated ≥ 4.0, what fraction appeared in the top-K recommendations? Measures how much of the user's actual taste the system captures. Achieved 65.1% at K=10.

---

## 🤝 Acknowledgements

- [GroupLens Research](https://grouplens.org) for the MovieLens dataset
- [TMDB](https://www.themoviedb.org) for movie poster images
- [Groq](https://groq.com) for fast LLM inference
