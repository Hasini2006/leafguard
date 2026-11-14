"""Integration test: validate, split and preprocess on a small synthetic dataset.

This starts a real local Spark session, so it is marked 'spark' and kept out of the fast
unit-test CI stage. Run it with:  make test-spark
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import make_leaf
from leafguard.data import preprocess, split, validate

pytestmark = pytest.mark.spark

CLASSES = ["healthy", "septoria"]


def build_raw_folder(raw: Path) -> None:
    """A miniature copy of the real dataset layout, with three deliberate problems."""
    for class_index, label in enumerate(CLASSES):
        for source, count in (("train", 12), ("val", 4)):
            folder = raw / source / label
            folder.mkdir(parents=True)
            for index in range(count):
                seed = class_index * 1000 + (500 if source == "val" else 0) + index
                (folder / f"img_{index}.jpg").write_bytes(
                    make_leaf(seed=seed, spots=3 + class_index * 8)
                )

    healthy = raw / "train" / "healthy"
    (healthy / "broken.jpg").write_bytes(b"this file is not a jpeg")
    (healthy / "tiny.jpg").write_bytes(make_leaf(seed=99, size=30))
    (healthy / "copy.jpg").write_bytes((healthy / "img_0.jpg").read_bytes())


def build_params(tmp_path: Path, raw: Path) -> dict:
    return {
        "seed": 42,
        "data": {
            "raw_dir": str(raw),
            "manifest_path": str(tmp_path / "interim" / "manifest.parquet"),
            "splits_path": str(tmp_path / "interim" / "splits.parquet"),
            "processed_dir": str(tmp_path / "processed"),
            "baseline_path": str(tmp_path / "models" / "baseline.json"),
            "train_source": "train",
            "test_source": "val",
            "classes": CLASSES,
            "max_per_class": None,
            "val_fraction": 0.25,
            "image_size": 64,
            "min_side_px": 64,
            "blur_threshold": 1.0,
            "min_images_per_class": 3,
            "n_bins": 5,
        },
        "spark": {"master": "local[2]", "driver_memory": "1g", "partitions": 2},
    }


def test_the_data_pipeline_runs_end_to_end(
    tmp_path: Path, runs_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    build_raw_folder(raw)
    params = build_params(tmp_path, raw)
    monkeypatch.chdir(tmp_path)  # reports are written relative to the working folder

    validate.run(params)
    quality = json.loads((tmp_path / "reports" / "data_quality.json").read_text())
    assert quality["total_images"] == 35
    assert quality["rejected_by_reason"] == {"corrupt": 1, "duplicate": 1, "too_small": 1}
    assert quality["valid_per_split_and_class"]["val"] == {"healthy": 4, "septoria": 4}
    assert quality["images_per_second"] > 0

    split.run(params)
    summary = json.loads((tmp_path / "reports" / "split_summary.json").read_text())
    assert summary["per_split"] == {"test": 8, "train": 18, "val": 6}

    preprocess.run(params)
    throughput = json.loads((tmp_path / "reports" / "throughput.json").read_text())
    assert throughput["processed_images"] == 32
    assert throughput["failed_images"] == 0

    sample = next((tmp_path / "processed" / "train" / "healthy").glob("*.jpg"))
    assert Image.open(sample).size == (64, 64)

    baseline = json.loads(Path(params["data"]["baseline_path"]).read_text())
    assert baseline["image_count"] == 18
    assert set(baseline["features"]) == {"brightness", "saturation", "green_ratio", "blur"}

    stages = [json.loads(line)["stage"] for line in runs_log.read_text().splitlines()]
    assert stages == ["validate", "split", "preprocess"]
