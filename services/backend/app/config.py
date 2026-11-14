from __future__ import annotations

import os
from dataclasses import dataclass, field


def _text(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def _number(name: str, fallback: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return fallback


@dataclass
class Settings:
    """Everything the backend needs to know about its neighbours and its limits."""

    model_server_url: str = field(
        default_factory=lambda: _text("MODEL_SERVER_URL", "http://model-server:5001")
    )
    mlflow_url: str = field(
        default_factory=lambda: _text("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    model_name: str = field(default_factory=lambda: _text("MODEL_NAME", "leafguard"))

    confidence_threshold: float = field(
        default_factory=lambda: _number("CONFIDENCE_THRESHOLD", 0.60)
    )
    min_green_ratio: float = field(default_factory=lambda: _number("MIN_GREEN_RATIO", 0.05))
    min_side_px: int = field(default_factory=lambda: int(_number("MIN_SIDE_PX", 64)))
    max_upload_mb: float = field(default_factory=lambda: _number("MAX_UPLOAD_MB", 5))
    request_timeout_s: float = field(default_factory=lambda: _number("REQUEST_TIMEOUT_S", 20))

    reports_dir: str = field(default_factory=lambda: _text("REPORTS_DIR", "/app/reports"))
    feedback_dir: str = field(default_factory=lambda: _text("FEEDBACK_DIR", "/app/feedback"))
    baseline_path: str = field(
        default_factory=lambda: _text("BASELINE_PATH", "/app/models/baseline.json")
    )
    fernet_key: str = field(default_factory=lambda: _text("FERNET_KEY", ""))
    drift_window: int = field(default_factory=lambda: int(_number("DRIFT_WINDOW", 50)))
    registry_ttl_s: float = field(default_factory=lambda: _number("REGISTRY_TTL_S", 30))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


settings = Settings()
