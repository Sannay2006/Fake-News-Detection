"""TF-IDF and Logistic Regression detector trained from the bundled news data."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import re
import threading
import zipfile
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TAG_RE = re.compile(r"<.*?>")
_BRACKET_RE = re.compile(r"\[.*?\]")
_NON_WORD_RE = re.compile(r"\W")
_DIGIT_WORD_RE = re.compile(r"\w*\d\w*")


class DatasetError(RuntimeError):
    """Raised when the bundled dataset cannot be used to train the model."""


def normalize_text(text: str) -> str:
    """Apply the notebook's article cleanup before vectorizing text."""
    text = text.lower()
    text = _BRACKET_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _DIGIT_WORD_RE.sub("", text)


class FakeNewsDetector:
    def __init__(
        self,
        dataset_path: Path,
        model_path: Path,
        max_articles_per_class: int = 6000,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.max_articles_per_class = max_articles_per_class
        self._pipeline: Pipeline | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _load_or_train(self) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            if self.model_path.is_file():
                saved = joblib.load(self.model_path)
                self._pipeline = saved["pipeline"]
                return self._pipeline

            self._pipeline = self._train()
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.model_path.with_suffix(
                self.model_path.suffix + ".tmp"
            )
            joblib.dump({"pipeline": self._pipeline}, temporary_path)
            os.replace(temporary_path, self.model_path)
            return self._pipeline

    def _read_articles(self) -> tuple[list[str], list[int]]:
        if not self.dataset_path.is_file():
            raise DatasetError(f"Dataset archive not found: {self.dataset_path}")

        articles: list[str] = []
        labels: list[int] = []
        found: set[str] = set()
        randomizer = random.Random(42)

        try:
            with zipfile.ZipFile(self.dataset_path) as archive:
                for member in archive.namelist():
                    filename = Path(member).name.lower()
                    if filename not in {"fake.csv", "true.csv"}:
                        continue

                    label = 0 if filename == "fake.csv" else 1
                    found.add(filename)
                    samples: list[str] = []
                    seen: set[bytes] = set()
                    with archive.open(member) as binary_file:
                        text_file = io.TextIOWrapper(
                            binary_file, encoding="utf-8-sig", newline=""
                        )
                        reader = csv.DictReader(text_file)
                        if not reader.fieldnames or "text" not in {
                            name.strip().lower() for name in reader.fieldnames
                        }:
                            raise DatasetError(
                                f"{member} is missing the required text column"
                            )
                        text_column = next(
                            name for name in reader.fieldnames
                            if name.strip().lower() == "text"
                        )
                        for row in reader:
                            article = normalize_text((row.get(text_column) or "").strip())
                            if len(article) < 60:
                                continue
                            digest = hashlib.sha256(article.encode("utf-8")).digest()
                            if digest in seen:
                                continue
                            seen.add(digest)
                            if len(samples) < self.max_articles_per_class:
                                samples.append(article)
                            else:
                                index = randomizer.randrange(len(seen))
                                if index < self.max_articles_per_class:
                                    samples[index] = article

                    articles.extend(samples)
                    labels.extend([label] * len(samples))
        except DatasetError:
            raise
        except (OSError, UnicodeError, zipfile.BadZipFile, csv.Error) as error:
            raise DatasetError(f"Could not read dataset archive: {error}") from error

        if found != {"fake.csv", "true.csv"}:
            missing = ", ".join(sorted({"fake.csv", "true.csv"} - found))
            raise DatasetError(f"Dataset archive is missing {missing}")
        if len(set(labels)) != 2 or min(labels.count(0), labels.count(1)) < 2:
            raise DatasetError("Dataset must contain at least two usable articles per class")
        return articles, labels

    def _train(self) -> Pipeline:
        articles, labels = self._read_articles()
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=40_000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipeline.fit(articles, labels)

    def analyze(self, article: str) -> dict[str, Any]:
        pipeline = self._load_or_train()
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        vector = vectorizer.transform([normalize_text(article)])
        predicted_label = int(classifier.predict(vector)[0])
        class_index = list(classifier.classes_).index(predicted_label)
        confidence = float(classifier.predict_proba(vector)[0][class_index])

        coefficients = classifier.coef_[0]
        direction = 1 if predicted_label == 1 else -1
        contributions = vector.toarray()[0] * coefficients * direction
        feature_names = vectorizer.get_feature_names_out()
        ranked_indices = contributions.argsort()[::-1]
        signals = [
            {
                "term": str(feature_names[index]),
                "effect": "supports_real" if predicted_label == 1 else "supports_fake",
            }
            for index in ranked_indices
            if contributions[index] > 0
        ][:5]

        return {
            "label": "real" if predicted_label == 1 else "fake",
            "confidence": round(confidence, 4),
            "signals": signals,
            "caveat": (
                "This is a text-pattern estimate from historical news, "
                "not a fact-check or a judgment of an article's source."
            ),
        }


def load_dataset_for_training(
    dataset_path: Path, max_articles_per_class: int = 6000
) -> tuple[list[str], list[int]]:
    """Expose the dataset reader for focused training checks and tests."""
    detector = FakeNewsDetector(dataset_path, Path(), max_articles_per_class)
    return detector._read_articles()
