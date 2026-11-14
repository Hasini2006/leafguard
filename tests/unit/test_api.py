"""Unit tests for the backend.

The model server is replaced by a stub, so these tests check the gateway's own behaviour:
validation, the error contract, the uncertain state, metrics and the pipeline endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import ValidationFailed, check_upload, describe
from conftest import make_leaf

CLASSES = ["bacterial_spot", "early_blight", "healthy", "late_blight", "leaf_mold", "septoria"]


class StubModel:
    """Stands in for the MLflow model server."""

    def __init__(self, scores: dict[str, float] | None = None, broken: bool = False) -> None:
        self.scores = scores or {
            "healthy": 0.91,
            "early_blight": 0.04,
            "late_blight": 0.02,
            "bacterial_spot": 0.01,
            "leaf_mold": 0.01,
            "septoria": 0.01,
        }
        self.broken = broken

    def alive(self) -> bool:
        return not self.broken

    def predict(self, raw: bytes):
        if self.broken:
            raise ValidationFailed("MODEL_UNAVAILABLE", "The diagnosis service is down.", 503)
        return self.scores, 12.5


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    with TestClient(main.app) as started:
        monkeypatch.setattr(main, "model_client", StubModel())
        yield started


def upload(raw: bytes, name: str = "leaf.jpg", content_type: str = "image/jpeg") -> dict:
    return {"file": (name, raw, content_type)}


# ----------------------------------------------------------------- upload validation


def test_a_normal_photo_passes_validation(leaf: bytes) -> None:
    checked = check_upload(leaf, "image/jpeg")
    assert checked.looks_like_leaf
    assert set(checked.stats) == {"brightness", "saturation", "green_ratio", "blur"}


@pytest.mark.parametrize(
    ("raw", "content_type", "code", "status"),
    [
        (b"", "image/jpeg", "NO_FILE", 400),
        (b"not an image at all", "image/jpeg", "INVALID_IMAGE", 400),
        (b"%PDF-1.4 fake", "application/pdf", "UNSUPPORTED_TYPE", 415),
    ],
)
def test_bad_uploads_are_rejected_with_a_useful_code(raw, content_type, code, status) -> None:
    with pytest.raises(ValidationFailed) as caught:
        check_upload(raw, content_type)
    assert caught.value.code == code
    assert caught.value.status_code == status
    assert caught.value.message  # a person must be told what to do about it


def test_a_tiny_image_is_rejected() -> None:
    with pytest.raises(ValidationFailed) as caught:
        check_upload(make_leaf(size=32), "image/jpeg")
    assert caught.value.code == "IMAGE_TOO_SMALL"


def test_an_oversized_upload_is_rejected() -> None:
    with pytest.raises(ValidationFailed) as caught:
        check_upload(b"x" * (6 * 1024 * 1024), "image/jpeg")
    assert caught.value.code == "FILE_TOO_LARGE"
    assert caught.value.status_code == 413


def test_every_known_class_has_plain_language_text() -> None:
    for name in CLASSES:
        words = describe(name)
        assert words["display_name"] and words["description"] and words["advice"]
    assert describe("something_else")["display_name"]  # unknown names still get a label


# ----------------------------------------------------------------- endpoints


def test_health_is_always_available(client: TestClient) -> None:
    reply = client.get("/health")
    assert reply.status_code == 200
    assert reply.json() == {"status": "ok"}


def test_ready_reports_503_when_the_model_server_is_down(client, monkeypatch) -> None:
    monkeypatch.setattr(main, "model_client", StubModel(broken=True))
    reply = client.get("/ready")
    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_a_confident_prediction_comes_back_in_full(client: TestClient, leaf: bytes) -> None:
    reply = client.post("/api/v1/predict", files=upload(leaf))
    assert reply.status_code == 200

    body = reply.json()
    assert body["status"] == "ok"
    assert body["predicted_class"] == "healthy"
    assert body["confidence"] == pytest.approx(0.91)
    assert body["display_name"] == "Healthy"
    assert len(body["top_k"]) == 3
    assert body["top_k"][0]["confidence"] >= body["top_k"][1]["confidence"]
    assert body["advice"] and body["disclaimer"]
    assert body["hint"] is None
    assert reply.headers["X-Request-ID"]


def test_a_low_score_is_reported_as_uncertain(client, leaf, monkeypatch) -> None:
    flat = {name: 1 / len(CLASSES) for name in CLASSES}
    monkeypatch.setattr(main, "model_client", StubModel(scores=flat))

    body = client.post("/api/v1/predict", files=upload(leaf)).json()
    assert body["status"] == "uncertain"
    assert body["hint"]
    assert body["advice"] == ""  # no treatment advice when we are not sure


def test_a_picture_that_is_not_a_leaf_is_reported_as_uncertain(client, monkeypatch) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 40, 40)).save(buffer, format="JPEG")
    body = client.post("/api/v1/predict", files=upload(buffer.getvalue())).json()
    assert body["status"] == "uncertain"


def test_the_model_being_down_returns_503_in_the_standard_error_shape(
    client, leaf, monkeypatch
) -> None:
    monkeypatch.setattr(main, "model_client", StubModel(broken=True))
    reply = client.post("/api/v1/predict", files=upload(leaf))
    assert reply.status_code == 503

    error = reply.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == "MODEL_UNAVAILABLE"


def test_a_missing_file_is_a_422_in_the_standard_error_shape(client: TestClient) -> None:
    reply = client.post("/api/v1/predict")
    assert reply.status_code == 422
    error = reply.json()["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert "file" in error["message"]


def test_feedback_needs_a_class_when_the_prediction_was_wrong(client: TestClient) -> None:
    reply = client.post("/api/v1/feedback", json={"request_id": "abc", "correct": False})
    assert reply.status_code == 422
    assert reply.json()["error"]["code"] == "MISSING_CLASS"

    unknown = client.post(
        "/api/v1/feedback",
        json={"request_id": "abc", "correct": False, "correct_class": "banana_rot"},
    )
    assert unknown.json()["error"]["code"] == "UNKNOWN_CLASS"


def test_agreeing_with_a_prediction_is_accepted(client: TestClient) -> None:
    reply = client.post("/api/v1/feedback", json={"request_id": "abc", "correct": True})
    assert reply.status_code == 200
    assert reply.json()["stored"] is True


def test_the_class_list_is_published_for_the_ui(client: TestClient) -> None:
    body = client.get("/api/v1/classes").json()
    assert {item["name"] for item in body} == set(CLASSES)


def test_metrics_are_exposed_in_the_prometheus_format(client, leaf) -> None:
    client.post("/api/v1/predict", files=upload(leaf))
    text = client.get("/metrics").text
    for metric in (
        "leafguard_requests_total",
        "leafguard_predictions_total",
        "leafguard_prediction_confidence",
        "leafguard_request_duration_seconds",
    ):
        assert metric in text


def test_the_pipeline_endpoints_cope_with_no_reports_yet(client: TestClient) -> None:
    assert client.get("/api/v1/pipeline/runs").json() == []

    summary = client.get("/api/v1/pipeline/summary").json()
    assert summary["stages"] == []
    assert summary["gate"] is None
    assert "flowchart" in client.get("/api/v1/pipeline/dag").text


def test_the_pipeline_summary_shows_the_newest_run_of_each_stage(
    client, tmp_path, monkeypatch
) -> None:
    from app import config, services

    runs = [
        {
            "run_id": "a",
            "stage": "validate",
            "status": "success",
            "started_at": "2026-01-01T10:00:00+00:00",
            "duration_s": 5.0,
            "items": 100,
            "items_per_s": 20.0,
            "error": None,
            "git_commit": "abc1234",
        },
        {
            "run_id": "b",
            "stage": "validate",
            "status": "failed",
            "started_at": "2026-01-01T11:00:00+00:00",
            "duration_s": 1.0,
            "items": 0,
            "items_per_s": 0.0,
            "error": "DataError: bad",
            "git_commit": "abc1234",
        },
    ]
    (tmp_path / "pipeline_runs.jsonl").write_text(
        "\n".join(json.dumps(run) for run in runs), encoding="utf-8"
    )
    monkeypatch.setattr(config.settings, "reports_dir", str(tmp_path))
    monkeypatch.setattr(services.settings, "reports_dir", str(tmp_path))

    summary = client.get("/api/v1/pipeline/summary").json()
    assert len(summary["stages"]) == 1
    assert summary["stages"][0]["status"] == "failed"  # the newest run wins
    assert summary["stages"][0]["error"] == "DataError: bad"
    assert len(client.get("/api/v1/pipeline/runs").json()) == 2


def test_reports_are_read_when_they_exist(client, tmp_path, monkeypatch) -> None:
    from app import config, services

    Path(tmp_path / "gate.json").write_text(
        json.dumps({"promoted": True, "candidate_version": "3"}), encoding="utf-8"
    )
    monkeypatch.setattr(config.settings, "reports_dir", str(tmp_path))
    monkeypatch.setattr(services.settings, "reports_dir", str(tmp_path))

    gate = client.get("/api/v1/pipeline/summary").json()["gate"]
    assert gate["promoted"] is True
    assert gate["candidate_version"] == "3"


def test_the_serving_version_is_looked_up_again_after_the_cache_expires(monkeypatch) -> None:
    from app import services

    reader = services.RegistryReader()  # MLflow is unreachable in tests: version "unknown"
    first = reader.champion()
    assert first["version"] == "unknown"
    stamp = reader.cached_at

    assert reader.champion() is first  # still fresh, so no new lookup
    monkeypatch.setattr(services.settings, "registry_ttl_s", 0)
    reader.champion()
    assert reader.cached_at > stamp  # expired, so MLflow was asked again


def test_the_model_list_is_empty_not_an_error_when_mlflow_is_down(client: TestClient) -> None:
    reply = client.get("/api/v1/models")
    assert reply.status_code == 200
    assert reply.json() == []


def test_model_versions_carry_their_aliases_and_gate_result(monkeypatch) -> None:
    from app import services

    class Reply:
        def __init__(self, body: dict) -> None:
            self.status_code = 200
            self.body = body

        def json(self) -> dict:
            return self.body

    model = {"registered_model": {"aliases": [{"alias": "champion", "version": "2"}]}}
    found = {
        "model_versions": [
            {
                "version": "1",
                "run_id": "r1",
                "creation_timestamp": 1_700_000_000_000,
                "tags": [{"key": "gate", "value": "passed"}, {"key": "accuracy", "value": "0.7"}],
            },
            {"version": "2", "run_id": "r2", "tags": [{"key": "gate", "value": "passed"}]},
        ]
    }
    reader = services.RegistryReader()
    monkeypatch.setattr(
        reader.http, "get", lambda url, params=None: Reply(model if url.endswith("/get") else found)
    )

    rows = reader.versions()
    assert [row["version"] for row in rows] == ["2", "1"]  # newest first
    assert rows[0]["aliases"] == ["champion"]
    assert rows[1]["aliases"] == []
    assert rows[1]["accuracy"] == "0.7"
    assert rows[1]["created_at"].startswith("2023-11-14")
