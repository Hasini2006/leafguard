"""Unit tests for the image helpers, drift maths, splitting and stage tracking."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import make_leaf
from leafguard import drift, images
from leafguard.common import (
    ConfigError,
    DataError,
    ImageError,
    StageResult,
    load_params,
    need,
    track_stage,
)
from leafguard.data.split import make_splits, summarise
from leafguard.data.validate import check_gate

ROOT = Path(__file__).resolve().parents[2]


def open_image(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ----------------------------------------------------------------- image helpers


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file:/data/raw/train/healthy/a.jpg", "/data/raw/train/healthy/a.jpg"),
        ("file:///data/raw/val/leaf_mold/x%20y.jpg", "/data/raw/val/leaf_mold/x y.jpg"),
        ("/already/plain.jpg", "/already/plain.jpg"),
    ],
)
def test_local_path_removes_the_spark_uri_prefix(uri: str, expected: str) -> None:
    assert images.local_path(uri) == expected


def test_split_and_label_read_from_the_folder_names() -> None:
    assert images.split_and_label("file:/x/data/raw/val/septoria/img.JPG") == ("val", "septoria")


def test_decode_rejects_rubbish_and_truncated_files(leaf: bytes) -> None:
    with pytest.raises(ImageError):
        images.decode(b"this is not an image")
    with pytest.raises(ImageError):
        images.decode(leaf[: len(leaf) // 2])


def test_perceptual_hash_is_stable_and_tells_images_apart() -> None:
    one = open_image(make_leaf(seed=1))
    other = open_image(make_leaf(seed=2, spots=14))
    assert images.perceptual_hash(one) == images.perceptual_hash(one.copy())
    assert images.perceptual_hash(one) != images.perceptual_hash(other)


def test_blur_score_drops_when_an_image_is_blurred() -> None:
    sharp = images.blur_score(open_image(make_leaf(seed=3)))
    soft = images.blur_score(open_image(make_leaf(seed=3, blur=4)))
    assert soft < sharp


def test_green_ratio_separates_a_leaf_from_a_plain_red_picture(leaf: bytes) -> None:
    assert images.green_ratio(open_image(leaf)) > 0.1
    assert images.green_ratio(Image.new("RGB", (64, 64), (220, 30, 30))) == 0.0


def test_inspect_flags_each_kind_of_problem() -> None:
    small = images.inspect(make_leaf(size=40), min_side=64, blur_limit=0)
    assert small["too_small"] and not small["corrupt"]

    broken = images.inspect(b"\x00\x01\x02", min_side=64, blur_limit=0)
    assert broken["corrupt"] and broken["width"] is None and broken["error"]

    good = images.inspect(make_leaf(), min_side=64, blur_limit=1.0)
    assert not any(good[flag] for flag in ("corrupt", "too_small", "too_blurry", "bad_format"))
    assert good["format"] == "JPEG" and good["width"] == 200


def test_resize_square_produces_an_rgb_jpeg_of_the_asked_size() -> None:
    result = Image.open(
        io.BytesIO(images.resize_square(make_leaf(size=150, image_format="PNG"), 64))
    )
    assert result.size == (64, 64)
    assert result.mode == "RGB"
    assert result.format == "JPEG"


def test_model_array_has_the_shape_the_network_expects(leaf: bytes) -> None:
    array = images.to_model_array(open_image(leaf), 128)
    assert array.shape == (3, 128, 128)
    assert array.dtype == np.float32


# ----------------------------------------------------------------- drift


def test_psi_is_small_for_the_same_distribution_and_large_after_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = drift.build_reference(rng.normal(0.5, 0.1, 2000), n_bins=10)
    assert len(reference["shares"]) == len(reference["edges"]) + 1
    assert drift.psi(reference, rng.normal(0.5, 0.1, 2000)) < 0.05
    assert drift.psi(reference, rng.normal(0.9, 0.1, 2000)) > 0.5


def test_psi_returns_zero_when_there_is_nothing_to_compare() -> None:
    reference = drift.build_reference(np.linspace(0, 1, 100), n_bins=5)
    assert drift.psi(reference, np.array([])) == 0.0


# ----------------------------------------------------------------- splitting


def build_manifest(
    train_count: int = 20, test_count: int = 5, labels=("healthy", "septoria")
) -> pd.DataFrame:
    rows = []
    for label in labels:
        for source, count in (("train", train_count), ("val", test_count)):
            for index in range(count):
                key = f"{label}-{source}-{index}"
                rows.append(
                    {
                        "path": f"/raw/{source}/{label}/{index}.jpg",
                        "file_name": f"{index}.jpg",
                        "label": label,
                        "source_split": source,
                        "file_md5": key,
                        "phash": key,
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows)


def test_split_is_stratified_and_keeps_the_val_folder_as_the_test_set() -> None:
    splits = make_splits(build_manifest(), val_fraction=0.15, seed=42)
    counts = splits.groupby(["split", "label"]).size()
    assert counts[("val", "healthy")] == 3
    assert counts[("train", "healthy")] == 17
    assert counts[("test", "septoria")] == 5
    assert set(splits[splits["split"] == "test"]["source_split"]) == {"val"}


def test_the_same_seed_always_gives_the_same_split() -> None:
    first = make_splits(build_manifest(), val_fraction=0.2, seed=7)
    again = make_splits(build_manifest(), val_fraction=0.2, seed=7)
    other = make_splits(build_manifest(), val_fraction=0.2, seed=8)
    pd.testing.assert_frame_equal(first, again)

    def val_files(frame: pd.DataFrame) -> set[str]:
        return set(frame[frame["split"] == "val"]["path"])

    assert val_files(first) != val_files(other)


def test_invalid_images_are_dropped_and_the_cap_is_respected() -> None:
    manifest = build_manifest()
    manifest.loc[0, "valid"] = False
    splits = make_splits(manifest, val_fraction=0.1, seed=1, cap=10)
    assert manifest.loc[0, "path"] not in set(splits["path"])
    per_class = splits[splits["split"] != "test"].groupby("label").size()
    assert (per_class == 10).all()


def test_near_duplicates_stay_together_and_never_leak_into_the_test_set() -> None:
    manifest = build_manifest(train_count=10, test_count=3, labels=("healthy",))
    manifest.loc[manifest.index[:4], "phash"] = "shared"
    first_test = manifest.index[manifest["source_split"] == "val"][0]
    manifest.loc[first_test, "phash"] = "shared"

    splits = make_splits(manifest, val_fraction=0.3, seed=3)
    group = splits[splits["phash"] == "shared"]
    assert group["split"].nunique() == 1
    assert "test" not in set(group["split"])


def test_asking_for_a_class_that_does_not_exist_is_an_error() -> None:
    with pytest.raises(DataError):
        make_splits(build_manifest(), val_fraction=0.1, seed=1, classes=["healthy", "late_blight"])


def test_summary_counts_every_split() -> None:
    summary = summarise(make_splits(build_manifest(), val_fraction=0.15, seed=42))
    assert summary["total"] == 50
    assert summary["per_split"] == {"test": 10, "train": 34, "val": 6}


# ----------------------------------------------------------------- config and tracking


def test_the_project_params_file_loads_and_has_every_section() -> None:
    params = load_params(ROOT / "params.yaml")
    for section in ("data", "spark", "train", "gate", "serving"):
        assert section in params
    assert params["data"]["image_size"] > 0


def test_config_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_params(tmp_path / "nope.yaml")

    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("- a\n- list\n")
    with pytest.raises(ConfigError):
        load_params(wrong)

    with pytest.raises(ConfigError):
        need({}, "image_size", "data")


def test_a_successful_stage_is_recorded(runs_log: Path) -> None:
    @track_stage("demo")
    def stage() -> StageResult:
        return StageResult(items=10, extra={"note": "fine"})

    assert stage().items == 10
    record = json.loads(runs_log.read_text().splitlines()[-1])
    assert record["stage"] == "demo"
    assert record["status"] == "success"
    assert record["items"] == 10
    assert record["error"] is None


def test_a_failing_stage_is_recorded_and_the_error_still_propagates(runs_log: Path) -> None:
    @track_stage("broken")
    def stage() -> StageResult:
        raise DataError("the data was no good")

    with pytest.raises(DataError):
        stage()
    record = json.loads(runs_log.read_text().splitlines()[-1])
    assert record["status"] == "failed"
    assert "the data was no good" in record["error"]


def test_the_quality_gate_rejects_a_missing_or_tiny_class() -> None:
    report = {"valid_images": 30, "valid_per_split_and_class": {"train": {"healthy": 25}}}
    check_gate(report, ["healthy"], min_per_class=20)  # passes, so no error

    with pytest.raises(DataError):
        check_gate(report, ["healthy", "septoria"], min_per_class=20)
    with pytest.raises(DataError):
        check_gate(report, ["healthy"], min_per_class=50)
