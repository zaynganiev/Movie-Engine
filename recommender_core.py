"""Shared recommender logic for demo web app and CLI."""
from __future__ import annotations
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

Rating = Tuple[int, int, float]  # (user_id, item_id, rating)


@dataclass
class MFConfig:
    factors: int = 16
    lr: float = 0.05
    reg: float = 0.01
    epochs: int = 40
    seed: int = 42


class MatrixFactorizationRecommender:
    """Basic matrix factorization for explicit feedback."""

    def __init__(self, num_users: int, num_items: int, config: MFConfig | None = None):
        self.cfg = config or MFConfig()
        self.num_users = num_users
        self.num_items = num_items
        rng = np.random.default_rng(self.cfg.seed)
        self.P = rng.normal(scale=0.1, size=(num_users, self.cfg.factors))
        self.Q = rng.normal(scale=0.1, size=(num_items, self.cfg.factors))

    def fit(self, ratings: Sequence[Rating]) -> None:
        rng = random.Random(self.cfg.seed)
        ratings = list(ratings)
        for epoch in range(self.cfg.epochs):
            rng.shuffle(ratings)
            total_loss = 0.0
            for u, i, r in ratings:
                pred = float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                self.P[u] += self.cfg.lr * (err * self.Q[i] - self.cfg.reg * self.P[u])
                self.Q[i] += self.cfg.lr * (err * self.P[u] - self.cfg.reg * self.Q[i])
                total_loss += err * err
            yield (epoch + 1, (total_loss / len(ratings)) ** 0.5)

    def predict(self, user_id: int, item_id: int) -> float:
        return float(np.dot(self.P[user_id], self.Q[item_id]))

    def recommend(self, user_id: int, known_items: set[int], k: int = 5) -> List[Tuple[int, float]]:
        scores: List[Tuple[int, float]] = []
        for item in range(self.num_items):
            if item in known_items:
                continue
            scores.append((item, self.predict(user_id, item)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


MOVIES = [
    "The Matrix",
    "The Matrix Reloaded",
    "John Wick",
    "John Wick: Chapter 2",
    "Speed",
    "Point Break",
    "Toy Story",
    "Finding Nemo",
    "Inside Out",
    "Up",
]

USER_RATINGS: Dict[str, Dict[str, float]] = {
    "alice": {
        "The Matrix": 5,
        "The Matrix Reloaded": 4,
        "John Wick": 5,
        "Toy Story": 2,
    },
    "bob": {
        "Toy Story": 5,
        "Finding Nemo": 5,
        "Inside Out": 4,
        "Up": 4,
    },
    "carol": {
        "The Matrix": 4,
        "Speed": 5,
        "Point Break": 4,
    },
    "dave": {
        "John Wick": 4,
        "John Wick: Chapter 2": 5,
        "Speed": 3,
    },
    "erin": {
        "Toy Story": 4,
        "Up": 5,
        "Finding Nemo": 4,
    },
    "frank": {
        "The Matrix": 5,
        "John Wick": 5,
        "Point Break": 3,
        "Inside Out": 1,
    },
}


def _build_id_maps(items: List[str], users: List[str]):
    item_to_id = {name: idx for idx, name in enumerate(items)}
    user_to_id = {name: idx for idx, name in enumerate(users)}
    id_to_item = {idx: name for name, idx in item_to_id.items()}
    id_to_user = {idx: name for name, idx in user_to_id.items()}
    return item_to_id, user_to_id, id_to_item, id_to_user


def _build_ratings(user_ratings: Dict[str, Dict[str, float]], item_to_id, user_to_id) -> List[Rating]:
    ratings: List[Rating] = []
    for user, items in user_ratings.items():
        for item, score in items.items():
            ratings.append((user_to_id[user], item_to_id[item], float(score)))
    return ratings


def _known_items_by_user(user_ratings: Dict[str, Dict[str, float]], item_to_id, user_to_id) -> Dict[int, set[int]]:
    known: Dict[int, set[int]] = {uid: set() for uid in user_to_id.values()}
    for user, items in user_ratings.items():
        uid = user_to_id[user]
        for item in items:
            known[uid].add(item_to_id[item])
    return known


@lru_cache(maxsize=1)
def train_demo_model(cfg: MFConfig | None = None):
    users = list(USER_RATINGS.keys())
    item_to_id, user_to_id, id_to_item, id_to_user = _build_id_maps(MOVIES, users)
    ratings = _build_ratings(USER_RATINGS, item_to_id, user_to_id)
    known = _known_items_by_user(USER_RATINGS, item_to_id, user_to_id)

    model = MatrixFactorizationRecommender(
        num_users=len(users), num_items=len(MOVIES), config=cfg or MFConfig(factors=8, lr=0.05, reg=0.02, epochs=60)
    )
    # Run fit generator to completion without printing here; caller can display metrics if desired.
    for _ in model.fit(ratings):
        pass
    return model, id_to_item, id_to_user, known


def recommend_for_username(username: str, k: int = 5):
    model, id_to_item, id_to_user, known = train_demo_model()
    users = {v: k for k, v in id_to_user.items()}
    if username not in users:
        raise KeyError(f"Unknown user '{username}'")
    uid = users[username]
    recs = model.recommend(uid, known[uid], k=k)
    return [(id_to_item[i], score) for i, score in recs]


def list_users():
    return list(USER_RATINGS.keys())


def list_movies():
    return MOVIES


def user_profile(username: str) -> Dict[str, float]:
    if username not in USER_RATINGS:
        raise KeyError(f"Unknown user '{username}'")
    return USER_RATINGS[username]
