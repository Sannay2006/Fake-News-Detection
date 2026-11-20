from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from detector import DatasetError, FakeNewsDetector


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "Datasets" / "datasets.zip"
MODEL_PATH = ROOT / "model" / "fake_news_pipeline.joblib"
MAX_ARTICLE_LENGTH = 20_000
MIN_ARTICLE_LENGTH = 60


def create_app(detector: Any | None = None) -> Flask:
    app = Flask(__name__, static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024
    app.extensions["news_detector"] = detector or FakeNewsDetector(
        DATASET_PATH, MODEL_PATH
    )

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        model = app.extensions["news_detector"]
        return jsonify({"status": "ok", "model_loaded": model.is_loaded})

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("article"), str):
            return jsonify({"error": "Send a JSON object with an article string."}), 400

        article = payload["article"].strip()
        if len(article) < MIN_ARTICLE_LENGTH:
            return jsonify(
                {"error": f"Enter at least {MIN_ARTICLE_LENGTH} characters to analyze."}
            ), 400
        if len(article) > MAX_ARTICLE_LENGTH:
            return jsonify(
                {"error": f"Articles must be {MAX_ARTICLE_LENGTH:,} characters or fewer."}
            ), 413

        try:
            result = app.extensions["news_detector"].analyze(article)
        except (DatasetError, OSError, ValueError, EOFError, KeyError):
            app.logger.exception("Article analysis failed")
            return jsonify(
                {"error": "The article could not be analyzed. Check the local dataset and try again."}
            ), 500
        return jsonify(result)

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Request is too large."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
