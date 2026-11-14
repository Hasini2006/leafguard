"""End-to-end tests against the running Docker Compose stack.

These go through the same doors a real user does: the nginx frontend on port 8080, which
proxies /api to the backend, which calls the MLflow model server. They need the stack up
and a champion model promoted, so they are marked 'e2e' and kept out of the unit-test CI
stage. Run them with:  make test-e2e

Every test is skipped, not failed, when the stack is not reachable at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

from conftest import make_leaf

pytestmark = pytest.mark.e2e

FRONTEND = os.environ.get("E2E_FRONTEND_URL", "http://localhost:8080")
BACKEND = os.environ.get("E2E_BACKEND_URL", "http://localhost:8000")
MODEL_SERVER = os.environ.get("E2E_MODEL_URL", "http://localhost:5001")
PROMETHEUS = os.environ.get("E2E_PROMETHEUS_URL", "http://localhost:9090")
ROOT = Path(__file__).resolve().parents[2]
CLASSES = {"bacterial_spot", "early_blight", "healthy", "late_blight", "leaf_mold", "septoria"}


def reachable(url: str) -> bool:
    try:
        return requests.get(url, timeout=3).status_code < 500
    except requests.RequestException:
        return False


@pytest.fixture(scope="module", autouse=True)
def stack_is_up() -> None:
    if not reachable(f"{BACKEND}/health"):
        pytest.skip("the Docker Compose stack is not running")


def sample_photo() -> tuple[str, bytes]:
    """A real test-set photo when the dataset is present, otherwise a drawn leaf."""
    folder = ROOT / "data" / "raw" / "val" / "healthy"
    photos = sorted(folder.glob("*.JPG")) if folder.exists() else []
    if photos:
        return photos[0].name, photos[0].read_bytes()
    return "leaf.jpg", make_leaf(seed=5)


def test_the_frontend_serves_all_three_pages() -> None:
    for page in ("index.html", "pipeline.html", "help.html"):
        reply = requests.get(f"{FRONTEND}/{page}", timeout=5)
        assert reply.status_code == 200
        assert "LeafGuard" in reply.text


def test_the_frontend_receives_its_runtime_config() -> None:
    reply = requests.get(f"{FRONTEND}/config.js", timeout=5)
    assert reply.status_code == 200
    assert "LEAFGUARD_API" in reply.text


def test_the_model_server_is_up_and_the_backend_is_ready() -> None:
    assert requests.get(f"{MODEL_SERVER}/ping", timeout=5).status_code == 200
    ready = requests.get(f"{BACKEND}/ready", timeout=5).json()
    assert ready["ready"] is True
    assert ready["model_version"] != "unknown"


def test_a_real_photo_is_diagnosed_through_the_frontend_proxy() -> None:
    name, raw = sample_photo()
    reply = requests.post(
        f"{FRONTEND}/api/v1/predict", files={"file": (name, raw, "image/jpeg")}, timeout=30
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["predicted_class"] in CLASSES
    assert body["status"] in {"ok", "uncertain"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 3
    assert body["inference_ms"] < 2000
    assert reply.headers["X-Request-ID"] == body["request_id"]


def test_feedback_on_a_prediction_is_stored() -> None:
    name, raw = sample_photo()
    prediction = requests.post(
        f"{FRONTEND}/api/v1/predict", files={"file": (name, raw, "image/jpeg")}, timeout=30
    ).json()
    reply = requests.post(
        f"{FRONTEND}/api/v1/feedback",
        json={
            "request_id": prediction["request_id"],
            "correct": False,
            "correct_class": "healthy",
            "comment": "e2e test",
        },
        timeout=10,
    )
    assert reply.status_code == 200
    assert reply.json()["stored"] is True


def test_a_non_image_is_refused_with_the_standard_error_shape() -> None:
    reply = requests.post(
        f"{FRONTEND}/api/v1/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        timeout=10,
    )
    assert reply.status_code == 415
    assert reply.json()["error"]["code"] == "UNSUPPORTED_TYPE"


def test_the_serving_model_version_is_published() -> None:
    info = requests.get(f"{FRONTEND}/api/v1/model/info", timeout=10).json()
    assert info["alias"] == "champion"
    assert info["version"] != "unknown"
    assert set(info["classes"]) == CLASSES


def test_the_console_lists_registered_versions_with_the_champion() -> None:
    info = requests.get(f"{FRONTEND}/api/v1/model/info", timeout=10).json()
    versions = requests.get(f"{FRONTEND}/api/v1/models", timeout=10).json()
    assert versions, "no registered versions listed"
    champions = [row["version"] for row in versions if "champion" in row["aliases"]]
    assert champions == [info["version"]]


def test_the_pipeline_console_has_data_to_show() -> None:
    summary = requests.get(f"{FRONTEND}/api/v1/pipeline/summary", timeout=10).json()
    stages = {run["stage"] for run in summary["stages"]}
    assert {"validate", "split", "preprocess", "train", "evaluate", "gate"} <= stages
    assert summary["gate"] is not None


def test_prometheus_scrapes_every_target() -> None:
    if not reachable(f"{PROMETHEUS}/-/ready"):
        pytest.skip("Prometheus is not running")
    targets = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=10).json()
    health = {t["labels"]["job"]: t["health"] for t in targets["data"]["activeTargets"]}
    assert health, "Prometheus has no targets"
    down = [job for job, state in health.items() if state != "up"]
    assert not down, f"targets down: {down}"
