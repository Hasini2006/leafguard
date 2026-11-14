"""Stage 'preprocess': parallel Spark resize plus the drift reference baseline.

Each Spark partition resizes its share of the images to data.image_size and writes them to
data/processed/<split>/<class>/, a layout torchvision's ImageFolder reads directly.
The same workers return the four drift features per image, so the baseline for the
training split is built here instead of reading every image a second time.

Outputs data/processed/ (resized images plus index.csv), models/baseline.json and
reports/throughput.json (images per second).

Run: python -m leafguard.data.preprocess
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from leafguard.common import (
    ImageError,
    StageError,
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
from leafguard.drift import build_reference
from leafguard.images import decode, image_stats, resize_square
from leafguard.spark_session import make_spark

logger = get_logger("preprocess")

SCHEMA = (
    "source_path string, out_path string, split string, label string, status string, "
    "error string, out_bytes long, brightness double, saturation double, "
    "green_ratio double, blur double"
)
COLUMNS = [part.strip().split(" ")[0] for part in SCHEMA.split(",")]
FEATURES = ["brightness", "saturation", "green_ratio", "blur"]


def resize_batches(
    batches: Iterator[pd.DataFrame], image_size: int, out_root: str
) -> Iterator[pd.DataFrame]:
    """Spark worker function: resize one image, write it, and report its drift features."""
    for batch in batches:
        rows = []
        columns = (batch["path"], batch["label"], batch["split"], batch["file_name"])
        for source_path, label, split, file_name in zip(*columns, strict=True):
            target = Path(out_root) / split / label / f"{Path(file_name).stem}.jpg"
            try:
                raw = Path(source_path).read_bytes()
                resized = resize_square(raw, image_size)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resized)
                stats = image_stats(decode(resized))
                rows.append(
                    [
                        source_path,
                        str(target),
                        split,
                        label,
                        "ok",
                        None,
                        len(resized),
                        *[stats[name] for name in FEATURES],
                    ]
                )
            except (OSError, ImageError) as err:
                rows.append(
                    [
                        source_path,
                        None,
                        split,
                        label,
                        "failed",
                        str(err)[:200],
                        0,
                        None,
                        None,
                        None,
                        None,
                    ]
                )
        frame = pd.DataFrame(rows, columns=COLUMNS)
        frame["out_bytes"] = frame["out_bytes"].astype("int64")
        yield frame


def make_baseline(train_rows: pd.DataFrame, image_size: int, n_bins: int) -> dict[str, Any]:
    """Build the drift reference from the processed training images."""
    return {
        "image_size": image_size,
        "image_count": int(len(train_rows)),
        "class_counts": {
            k: int(v) for k, v in train_rows["label"].value_counts().sort_index().items()
        },
        "features": {
            name: build_reference(train_rows[name].to_numpy(), n_bins) for name in FEATURES
        },
    }


@track_stage("preprocess")
def run(params: dict[str, Any] | None = None) -> StageResult:
    params = params or load_params()
    data = need(params, "data", "root")
    spark_config = need(params, "spark", "root")
    splits_path = full_path(need(data, "splits_path", "data"))
    out_root = full_path(need(data, "processed_dir", "data"))
    image_size = int(need(data, "image_size", "data"))

    if out_root.exists():
        shutil.rmtree(out_root)  # always rebuild from scratch, so the output is reproducible
    out_root.mkdir(parents=True)

    spark = make_spark("leafguard-preprocess", spark_config)
    try:
        started = time.perf_counter()
        results = (
            spark.read.parquet(str(splits_path))
            .select("path", "label", "split", "file_name")
            .repartition(int(spark_config.get("partitions", 8)))
            .mapInPandas(
                partial(resize_batches, image_size=image_size, out_root=str(out_root)),
                schema=SCHEMA,
            )
            .toPandas()
        )
        seconds = time.perf_counter() - started
    finally:
        spark.stop()

    done = results[results["status"] == "ok"].copy()
    failed = results[results["status"] != "ok"]
    total = len(results)
    if total == 0:
        raise StageError("there were no images to preprocess")
    if len(failed) > 0:
        raise StageError(f"{len(failed)} of {total} images could not be processed")

    done["out_path"] = done["out_path"].map(lambda p: str(Path(p).relative_to(out_root)))
    index = done.sort_values(["split", "label", "out_path"])
    index[["split", "label", "out_path", "source_path"]].to_csv(out_root / "index.csv", index=False)

    baseline = make_baseline(
        done[done["split"] == "train"], image_size, int(data.get("n_bins", 10))
    )
    write_json(need(data, "baseline_path", "data"), baseline)

    report = {
        "image_size": image_size,
        "total_images": total,
        "processed_images": len(done),
        "failed_images": len(failed),
        "per_split": {k: int(v) for k, v in done["split"].value_counts().sort_index().items()},
        "output_megabytes": round(float(done["out_bytes"].sum()) / 1e6, 2),
        "elapsed_seconds": round(seconds, 3),
        "images_per_second": round(total / seconds, 2) if seconds > 0 else 0.0,
        "spark_master": spark_config.get("master"),
        "partitions": int(spark_config.get("partitions", 8)),
    }
    write_json("reports/throughput.json", report)
    log_info(
        logger,
        "preprocess finished",
        processed=len(done),
        images_per_second=report["images_per_second"],
    )
    return StageResult(items=total, extra={"images_per_second": report["images_per_second"]})


if __name__ == "__main__":
    run_stage(run, "preprocess")
