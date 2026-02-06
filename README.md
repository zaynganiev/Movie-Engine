# Netflix-style Collaborative Filtering Demo

This repo contains a tiny, self-contained recommender to illustrate core ideas behind movie recommendations:

- **Collaborative filtering**: learn from user–item interactions rather than content.
- **Matrix factorization & embeddings**: derive latent vectors for users and movies to predict preferences.
- **Ranking**: produce top-N recommendations by scoring unseen movies.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python recommender_demo.py
```

Expected output: training RMSE every 10 epochs and top-3 movie picks for each demo user.

## Run the mini website

```bash
source .venv/bin/activate
python app.py
# open http://127.0.0.1:5000
```

The page lets you pick a demo user and returns top-N recommendations from the matrix-factorization model, alongside their existing ratings.

## What the demo does

1. Builds a small synthetic dataset of users and movie ratings.
2. Trains a **matrix factorization** model with stochastic gradient descent to learn user and item **embeddings**.
3. Uses dot-product scores to **rank** unseen movies per user and returns the top-N.

## Files

- `recommender_demo.py` — executable script with model + demo data.
- `requirements.txt` — lightweight dependency list (numpy only).

## Next ideas to extend

- Swap in real data (e.g., MovieLens), split train/validation, and add evaluation metrics (MAP@K, NDCG).
- Try implicit-feedback losses (BPR, WARP) or ALS for implicit data.
- Add content-based features (genres, cast) for hybrid recommendations.
- Export embeddings and serve via a simple API for live recommendations.
