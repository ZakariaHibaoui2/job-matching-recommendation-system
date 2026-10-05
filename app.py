from flask import Flask, render_template, request, jsonify
from ml_engine import engine
import threading

app = Flask(__name__)

# ── Train in background on startup ───────────────────────
def _train():
    try:
        info = engine.train()
        print(f"\n✓ Model ready  —  accuracy {info['accuracy']}%  |  {info['users']} users  |  {info['jobs']} jobs\n")
    except Exception as e:
        print(f"\n✗ Training failed: {e}\n")

threading.Thread(target=_train, daemon=True).start()


# ── Pages ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── API ───────────────────────────────────────────────────
@app.route("/api/status")
def status():
    if not engine.ready:
        return jsonify(ready=False, message="Training in progress…")
    s = engine.stats()
    return jsonify(
        ready      = True,
        users      = s["total_users"],
        jobs       = s["total_jobs"],
        hard_skills= s["total_hard"],
        soft_skills= s["total_soft"],
        accuracy   = s["accuracy"],
        categories = s["categories"],
    )


@app.route("/api/recommend", methods=["POST"])
def recommend():
    if not engine.ready:
        return jsonify(error="Model still training, please wait…"), 503

    body = request.get_json(force=True)
    hard = [s.strip() for s in body.get("hard_skills", "").split(",") if s.strip()]
    soft = [s.strip() for s in body.get("soft_skills", "").split(",") if s.strip()]

    if not hard and not soft:
        return jsonify(error="Enter at least one skill"), 400

    results = engine.recommend(
        hard_skills=hard,
        soft_skills=soft,
        top_n=11,
    )
    return jsonify(results=results, total=len(results))


@app.route("/api/autocomplete/hard")
def ac_hard():
    q = request.args.get("q", "").strip()
    return jsonify(engine.suggest_hard(q) if len(q) >= 2 else [])


@app.route("/api/autocomplete/soft")
def ac_soft():
    q = request.args.get("q", "").strip()
    return jsonify(engine.suggest_soft(q) if len(q) >= 2 else [])


@app.route("/api/metrics")
def metrics():
    if not engine.ready:
        return jsonify(error="Not ready"), 503
    return jsonify(engine.stats())


if __name__ == "__main__":
    app.run(debug=False, port=5000)