from __future__ import annotations

from flask import Flask, render_template, request

from recommender_core import list_users, recommend_for_username, user_profile

app = Flask(__name__)

# Warm the model at startup so first request is fast
# (train_demo_model is cached inside recommender_core)
from recommender_core import train_demo_model  # noqa: E402
train_demo_model()


@app.route("/", methods=["GET", "POST"])
def index():
    users = list_users()
    selected_user = request.values.get("user", users[0])
    top_k = int(request.values.get("k", 5))
    top_k = min(max(top_k, 1), 10)

    try:
        recs = recommend_for_username(selected_user, k=top_k)
        profile = user_profile(selected_user)
        error = None
    except KeyError as e:
        recs = []
        profile = {}
        error = str(e)

    return render_template(
        "index.html",
        users=users,
        selected_user=selected_user,
        recs=recs,
        profile=profile,
        top_k=top_k,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
