"""
Simple Netflix-style collaborative filtering demo.

Features:
- Matrix factorization with SGD to learn user/item embeddings.
- Embedding-based scoring for ranking items a user hasn't watched.
- Small synthetic movie dataset for quick experimentation.

Run:
    python recommender_demo.py
"""
from __future__ import annotations

from recommender_core import list_users, train_demo_model


def demo():
    model, id_to_item, id_to_user, known = train_demo_model()
    # Show top recommendations for each user
    for uid in range(len(list_users())):
        user = id_to_user[uid]
        recs = model.recommend(uid, known[uid], k=3)
        pretty = ", ".join(f"{id_to_item[i]} (score={s:.2f})" for i, s in recs)
        print(f"\nRecommendations for {user}: {pretty}")


if __name__ == "__main__":
    demo()
