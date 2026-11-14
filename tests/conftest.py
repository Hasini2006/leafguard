"""Shared test fixtures.

The tests draw their own leaf-like images instead of reading the dataset, so they run
anywhere, stay fast, and never depend on data that is tracked by DVC rather than git.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

# Tests must never reach the Pushgateway or write to the real run log, and the backend
# settings are read when app.config is first imported, so point them at a closed port.
# Connection refused comes back immediately, while a bad hostname would wait on DNS.
# These are forced, not defaulted: `dvc repro` runs the tests with MLFLOW_TRACKING_URI
# pointing at the live server, and unit tests must never depend on what is running.
os.environ["LEAFGUARD_NO_PUSH"] = "1"
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:1"
os.environ["MODEL_SERVER_URL"] = "http://127.0.0.1:1"
os.environ.setdefault("LEAFGUARD_LOG_DIR", "/tmp/leafguard-test-logs")


def make_leaf(
    seed: int = 0, size: int = 200, spots: int = 5, blur: float = 0.0, image_format: str = "JPEG"
) -> bytes:
    """Draw a green leaf shape with brown spots and return the encoded bytes."""
    rng = np.random.default_rng(seed)
    canvas = Image.new("RGB", (size, size), (225, 225, 215))
    pen = ImageDraw.Draw(canvas)

    green = (40 + int(rng.integers(0, 40)), 130 + int(rng.integers(0, 60)), 40)
    left, top = (int(v) for v in rng.integers(0, size // 4, size=2))
    right, bottom = (int(v) for v in rng.integers(3 * size // 4, size, size=2))
    pen.ellipse([left, top, right, bottom], fill=green)

    for _ in range(spots):
        x, y = rng.integers(size // 4, 3 * size // 4, size=2)
        radius = int(rng.integers(3, max(4, size // 15)))
        pen.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(110, 70, 30))

    noise = rng.integers(-12, 12, size=(size, size, 3))
    canvas = Image.fromarray(np.clip(np.asarray(canvas, dtype=int) + noise, 0, 255).astype("uint8"))
    if blur:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))

    buffer = io.BytesIO()
    canvas.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def leaf() -> bytes:
    return make_leaf(seed=1)


@pytest.fixture
def runs_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the stage run log at a temporary file for the duration of one test."""
    path = tmp_path / "runs.jsonl"
    monkeypatch.setenv("LEAFGUARD_RUNS_LOG", str(path))
    return path
