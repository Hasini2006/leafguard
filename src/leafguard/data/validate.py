"""Stage 'validate': a Spark data-quality pass over every raw image.

Spark reads all image files with the binaryFile source and decodes them in parallel
Python workers. For each image it records size, format, perceptual hash, blur and colour
statistics, then flags:

  corrupt        the file cannot be decoded at all
  bad_format     not a JPEG, PNG or WEBP
  too_small      shorter side below data.min_side_px
  too_blurry     variance of Laplacian below data.blur_threshold
  label_conflict the same file content appears under two different class folders
  duplicate      the same file content appears twice (only the first copy is kept)

Writes data/interim/manifest.parquet (one row per image) and reports/data_quality.json
(counts, reject reasons and images per second).

Run: python -m leafguard.data.validate
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

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
from leafguard.images import file_hash, inspect, local_path, split_and_label
from leafguard.spark_session import make_spark

logger = get_logger("validate")

IMAGE_GLOB = "*.{jpg,jpeg,png,webp,JPG,JPEG,PNG,WEBP}"

# Schema of what each Spark worker hands back. Keep this in step with inspect().
SCHEMA = (
    "path string, file_name string, source_split string, label string, file_bytes long, "
    "file_md5 string, width long, height long, mode string, format string, phash string, "
    "blur double, green_ratio double, brightness double, saturation double, "
    "corrupt boolean, too_small boolean, too_blurry boolean, bad_format boolean, error string"
)
COLUMNS = [part.strip().split(" ")[0] for part in SCHEMA.split(",")]
INT_COLUMNS = ["file_bytes", "width", "height"]
BOOL_COLUMNS = ["corrupt", "too_small", "too_blurry", "bad_format"]


def inspect_batches(
    batches: Iterator[pd.DataFrame], min_side: int, blur_limit: float
) -> Iterator[pd.DataFrame]:
    """Spark worker function: turn raw file bytes into one quality row per image."""
    for batch in batches:
        rows = []
        columns = (batch["path"], batch["content"], batch["length"])
        for path, content, length in zip(*columns, strict=True):
            data = bytes(content)
            source_split, label = split_and_label(path)
            plain = local_path(path)
            rows.append(
                {
                    "path": plain,
                    "file_name": Path(plain).name,
                    "source_split": source_split,
                    "label": label,
                    "file_bytes": int(length),
                    "file_md5": file_hash(data),
                    **inspect(data, min_side, blur_limit),
                }
            )
        frame = pd.DataFrame(rows, columns=COLUMNS)
        frame[INT_COLUMNS] = frame[INT_COLUMNS].astype("Int64")
        frame[BOOL_COLUMNS] = frame[BOOL_COLUMNS].astype("boolean")
        yield frame


def flag_duplicates(frame: DataFrame) -> DataFrame:
    """Add duplicate and conflict flags, then the first matching reject reason."""
    same_file = Window.partitionBy("file_md5")
    frame = (
        frame.withColumn("copy_number", F.row_number().over(same_file.orderBy("path")))
        .withColumn("duplicate", F.col("copy_number") > 1)
        .withColumn("label_conflict", F.size(F.collect_set("label").over(same_file)) > 1)
        .withColumn(
            "similar_group_size",
            F.when(
                F.col("phash").isNotNull(), F.count("*").over(Window.partitionBy("phash"))
            ).otherwise(F.lit(0)),
        )
        .drop("copy_number")
    )
    reason = (
        F.when(F.col("corrupt"), "corrupt")
        .when(F.col("bad_format"), "bad_format")
        .when(F.col("too_small"), "too_small")
        .when(F.col("label_conflict"), "label_conflict")
        .when(F.col("duplicate"), "duplicate")
        .when(F.col("too_blurry"), "too_blurry")
    )
    return frame.withColumn("reject_reason", reason).withColumn(
        "valid", F.col("reject_reason").isNull()
    )


def build_report(manifest: pd.DataFrame, seconds: float, train_source: str) -> dict[str, Any]:
    """Summarise the manifest into the data-quality report, also used as DVC metrics."""
    total = len(manifest)
    good = manifest[manifest["valid"]]
    readable = manifest[~manifest["corrupt"].astype(bool)]
    per_class = good.groupby(["source_split", "label"]).size().unstack(fill_value=0)
    train_counts = good[good["source_split"] == train_source]["label"].value_counts()
    return {
        "total_images": total,
        "valid_images": len(good),
        "rejected_images": total - len(good),
        "rejected_by_reason": {
            str(k): int(v) for k, v in manifest["reject_reason"].value_counts().sort_index().items()
        },
        "valid_per_split_and_class": {
            split: {label: int(n) for label, n in row.items()}
            for split, row in per_class.to_dict("index").items()
        },
        "class_imbalance_ratio": (
            round(float(train_counts.max() / train_counts.min()), 3) if len(train_counts) else None
        ),
        "similar_image_groups": int(
            readable[readable["similar_group_size"] > 1]["phash"].nunique() if len(readable) else 0
        ),
        "formats": {str(k): int(v) for k, v in readable["format"].value_counts().items()},
        "width_range": (
            [int(readable["width"].min()), int(readable["width"].max())] if len(readable) else None
        ),
        "blur_median": round(float(readable["blur"].median()), 2) if len(readable) else None,
        "elapsed_seconds": round(seconds, 3),
        "images_per_second": round(total / seconds, 2) if seconds > 0 else 0.0,
    }


def check_gate(report: dict[str, Any], classes: list[str], min_per_class: int) -> None:
    """Stop the pipeline if any requested class is missing or nearly empty."""
    if report["valid_images"] == 0:
        raise DataError("no valid images found in the raw dataset")
    problems = []
    for split, counts in report["valid_per_split_and_class"].items():
        for label in classes:
            found = counts.get(label, 0)
            if found < min_per_class:
                problems.append(f"{split}/{label} has {found} valid images (need {min_per_class})")
    if problems:
        raise DataError("quality gate failed: " + "; ".join(problems))


@track_stage("validate")
def run(params: dict[str, Any] | None = None) -> StageResult:
    params = params or load_params()
    data = need(params, "data", "root")
    raw_dir = full_path(need(data, "raw_dir", "data"))
    manifest_path = full_path(need(data, "manifest_path", "data"))

    if not raw_dir.exists() or not any(raw_dir.rglob("*")):
        raise DataError(f"raw data folder is missing or empty: {raw_dir}")

    spark = make_spark("leafguard-validate", need(params, "spark", "root"))
    try:
        started = time.perf_counter()
        files = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", IMAGE_GLOB)
            .option("recursiveFileLookup", "true")
            .load(str(raw_dir))
            .select("path", "content", "length")
            .repartition(int(params["spark"].get("partitions", 8)))
        )
        inspected = files.mapInPandas(
            partial(
                inspect_batches,
                min_side=int(data["min_side_px"]),
                blur_limit=float(data["blur_threshold"]),
            ),
            schema=SCHEMA,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        flag_duplicates(inspected).coalesce(1).write.mode("overwrite").parquet(str(manifest_path))
        seconds = time.perf_counter() - started
    finally:
        spark.stop()

    manifest = pd.read_parquet(manifest_path)
    report = build_report(manifest, seconds, data.get("train_source", "train"))
    write_json("reports/data_quality.json", report)
    log_info(
        logger,
        "validation finished",
        total=report["total_images"],
        valid=report["valid_images"],
        rejected=report["rejected_by_reason"],
        images_per_second=report["images_per_second"],
    )
    check_gate(report, list(data.get("classes") or []), int(data.get("min_images_per_class", 1)))
    return StageResult(
        items=report["total_images"],
        extra={"valid": report["valid_images"], "images_per_second": report["images_per_second"]},
    )


if __name__ == "__main__":
    run_stage(run, "validate")
