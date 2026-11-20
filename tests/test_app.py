import csv
import io
import zipfile

from app import create_app
from detector import FakeNewsDetector, load_dataset_for_training


def create_dataset(path):
    fake = [
        "Anonymous sources reveal a shocking secret conspiracy that officials deny.",
        "A shocking secret plot was exposed by anonymous sources and officials deny.",
        "The shocking conspiracy was hidden by officials, anonymous sources reveal.",
        "An anonymous source reveals officials deny the shocking secret plot, according to records.",
    ]
    real = [
        "The committee published its report after a public hearing on Tuesday.",
        "Officials released the committee report following Tuesday's public hearing.",
        "A public hearing took place Tuesday before the committee published its report.",
        "The report was published by the committee after Tuesday's public hearing.",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name, texts in (("Fake.csv", fake), ("True.csv", real)):
            content = io.StringIO(newline="")
            writer = csv.writer(content)
            writer.writerow(["title", "text"])
            writer.writerows([["Story", text] for text in texts])
            archive.writestr(f"news-data/{name}", content.getvalue())


def test_loads_fake_and_true_csvs_from_nested_zip(tmp_path):
    archive_path = tmp_path / "datasets.zip"
    create_dataset(archive_path)

    articles, labels = load_dataset_for_training(archive_path)

    assert len(articles) == 8
    assert labels.count(0) == 4
    assert labels.count(1) == 4


def test_detector_trains_saves_and_explains_prediction(tmp_path):
    archive_path = tmp_path / "datasets.zip"
    model_path = tmp_path / "model.joblib"
    create_dataset(archive_path)
    detector = FakeNewsDetector(archive_path, model_path, max_articles_per_class=20)

    result = detector.analyze(
        "Anonymous sources reveal a shocking secret conspiracy that officials deny."
    )

    assert result["label"] == "fake"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["signals"]
    assert all(signal["effect"] == "supports_fake" for signal in result["signals"])
    assert model_path.is_file()
    reloaded = FakeNewsDetector(archive_path, model_path)
    assert reloaded.analyze(
        "The committee published its report after a public hearing on Tuesday."
    )["label"] == "real"


class StubDetector:
    is_loaded = True

    def analyze(self, article):
        return {
            "label": "real",
            "confidence": 0.91,
            "signals": [{"term": "committee", "effect": "supports_real"}],
            "caveat": "This is not a fact-check.",
        }


def test_analyze_endpoint_returns_result():
    client = create_app(StubDetector()).test_client()

    response = client.post(
        "/api/analyze",
        json={"article": "A sufficiently long news article that is more than sixty characters in length."},
    )

    assert response.status_code == 200
    assert response.json["label"] == "real"
    assert response.json["signals"][0]["term"] == "committee"


def test_analyze_endpoint_rejects_invalid_input():
    client = create_app(StubDetector()).test_client()

    response = client.post("/api/analyze", json={"article": "too short"})

    assert response.status_code == 400
    assert "60 characters" in response.json["error"]


def test_health_and_home_routes():
    client = create_app(StubDetector()).test_client()

    assert client.get("/api/health").json == {"status": "ok", "model_loaded": True}
    assert b"Read between" in client.get("/").data
