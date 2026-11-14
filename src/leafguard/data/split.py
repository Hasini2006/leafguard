"""Stage 'split': a reproducible, stratified, leakage-free train / val / test assignment.

  test   every valid image from the raw 'val' folder. This is the golden set: it is never
         used for training or tuning, so comparing two model versions on it is fair.
  train  the raw 'train' folder, minus the validation share.
  val    data.val_fraction of the raw 'train' folder, taken per class.

Two rules stop the same leaf appearing in two splits, which would make the scores look
better than they are: images sharing a perceptual hash always land in the same split, and
test images that duplicate a training image are dropped.

data.max_per_class caps how many training images each class contributes, which is the dial
for a faster run and for the "new data arrives" retraining demo.

Run: python -m leafguard.data.split
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pandas as pd

from leafguard.common import (
    DataError,
    StageResult,
    full_path,
    get_logger,
    load_params,
    log_info,
    need,
    run_stage,
    track_stage,
    write_json,
)

logger = get_logger("split")

OUTPUT_COLUMNS = ["path", "file_name", "label", "split", "source_split", "file_md5", "phash"]


def class_rng(seed: int, label: str) -> np.random.Generator:
    """One generator per class, so adding a class never reshuffles the others."""
    return np.random.default_rng([seed, zlib.crc32(label.encode())])


def split_one_class(
    frame: pd.DataFrame, val_fraction: float, seed: int, cap: int | None
) -> pd.DataFrame:
    """Pick the images for one class and label each of them train or val."""
    label = str(frame["label"].iloc[0])
    frame = frame.sort_values("path").copy()
    # Near-identical images share a group and therefore always share a split.
    frame["group"] = frame["phash"].fillna(frame["file_md5"])
    groups = sorted(frame["group"].unique())
    shuffled = [groups[i] for i in class_rng(seed, label).permutation(len(groups))]
    group_sizes = frame.groupby("group").size()

    kept, kept_count = [], 0
    for group in shuffled:
        if cap is not None and kept_count >= cap:
            break
        kept.append(group)
        kept_count += int(group_sizes[group])

    wanted_val = round(kept_count * val_fraction) if kept_count > 1 else 0
    val_groups, val_count = set(), 0
    for group in kept:
        if val_count >= wanted_val or kept_count - val_count - int(group_sizes[group]) < 1:
            break
        val_groups.add(group)
        val_count += int(group_sizes[group])

    chosen = frame[frame["group"].isin(kept)].copy()
    chosen["split"] = np.where(chosen["group"].isin(val_groups), "val", "train")
    return chosen


def make_splits(
    manifest: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    cap: int | None = None,
    classes: list[str] | None = None,
    train_source: str = "train",
    test_source: str = "val",
) -> pd.DataFrame:
    """Assign every valid image to train, val or test."""
    if not 0 <= val_fraction < 1:
        raise DataError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    frame = manifest[manifest["valid"].astype(bool)].copy()
    if classes:
        missing = sorted(set(classes) - set(frame["label"]))
        if missing:
            raise DataError(f"no valid images for these classes: {missing}")
        frame = frame[frame["label"].isin(classes)]
    if frame.empty:
        raise DataError("no valid images to split")

    pool = frame[frame["source_split"] == train_source]
    test = frame[frame["source_split"] == test_source].copy()
    if pool.empty:
        raise DataError(f"no images found in the '{train_source}' source folder")

    leaked = test["file_md5"].isin(pool["file_md5"]) | (
        test["phash"].notna() & test["phash"].isin(pool["phash"].dropna())
    )
    if leaked.any():
        log_info(
            logger, "dropping test images that also appear in training", count=int(leaked.sum())
        )
    test = test[~leaked]
    test["split"] = "test"

    parts = [
        split_one_class(group, val_fraction, seed, cap)
        for _, group in pool.groupby("label", sort=True)
    ]
    result = pd.concat([*parts, test], ignore_index=True)
    return result[OUTPUT_COLUMNS].sort_values(["split", "label", "path"]).reset_index(drop=True)


def summarise(splits: pd.DataFrame) -> dict[str, Any]:
    counts = splits.groupby(["split", "label"]).size().unstack(fill_value=0)
    return {
        "total": int(len(splits)),
        "per_split": {k: int(v) for k, v in splits["split"].value_counts().sort_index().items()},
        "per_split_and_class": {
            split: {label: int(n) for label, n in row.items()}
            for split, row in counts.to_dict("index").items()
        },
    }


@track_stage("split")
def run(params: dict[str, Any] | None = None) -> StageResult:
    params = params or load_params()
    data = need(params, "data", "root")
    manifest = pd.read_parquet(full_path(need(data, "manifest_path", "data")))

    splits = make_splits(
        manifest,
        val_fraction=float(data["val_fraction"]),
        seed=int(params.get("seed", 42)),
        cap=data.get("max_per_class"),
        classes=data.get("classes"),
        train_source=data.get("train_source", "train"),
        test_source=data.get("test_source", "val"),
    )
    out_path = full_path(need(data, "splits_path", "data"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out_path, index=False)

    summary = summarise(splits)
    write_json("reports/split_summary.json", summary)
    log_info(logger, "split finished", per_split=summary["per_split"])
    return StageResult(items=summary["total"], extra=summary["per_split"])


if __name__ == "__main__":
    run_stage(run, "split")
