"""Stage 'evaluate': score the trained model on the golden test set.

The test split comes from the dataset's own val folder and is never used for training or
tuning, so the numbers here are what the promotion gate compares between model versions.

Writes reports/metrics.json (accuracy, macro F1, per-class scores, CPU latency) and
reports/confusion_matrix.png, and logs both back to the training run in MLflow.

Run: python -m leafguard.training.evaluate
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Matplotlib must draw to a file, not a screen: there is no display in the VM shell.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from leafguard.common import (
    StageError,
    StageResult,
    full_path,
    get_logger,
    load_params,
    log_info,
    mlflow_uri,
    need,
    read_json,
    run_stage,
    track_stage,
    write_json,
)
from leafguard.images import IMAGENET_MEAN, IMAGENET_STD

logger = get_logger("evaluate")


def load_test_set(processed_dir: Path, batch_size: int = 32) -> tuple[DataLoader, list[str]]:
    steps = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
        ]
    )
    test_set = datasets.ImageFolder(processed_dir / "test", transform=steps)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False), test_set.classes


def measure_latency(
    net: torch.jit.ScriptModule, image_size: int, rounds: int = 30
) -> dict[str, float]:
    """Time single-image CPU inference, which is how the service is actually used."""
    sample = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(5):  # warm up
            net(sample)
        timings = []
        for _ in range(rounds):
            started = time.perf_counter()
            net(sample)
            timings.append((time.perf_counter() - started) * 1000)
    values = np.array(timings)
    return {
        "latency_p50_ms": round(float(np.percentile(values, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(values, 95)), 2),
    }


def save_confusion_matrix(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.imshow(matrix, cmap="Greens")
    axes.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(classes)), classes, fontsize=8)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")
    axes.set_title("LeafGuard confusion matrix (golden test set)")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axes.text(
                column,
                row,
                int(matrix[row][column]),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


@track_stage("evaluate")
def run(params: dict[str, Any] | None = None) -> StageResult:
    params = params or load_params()
    data = need(params, "data", "root")
    processed_dir = full_path(need(data, "processed_dir", "data"))
    info = read_json("models/run.json")
    if info is None:
        raise StageError("models/run.json is missing: run the train stage first")

    net = torch.jit.load(str(full_path("models/model.pt")))
    net.eval()
    loader, classes = load_test_set(processed_dir)
    if classes != info["classes"]:
        raise StageError("the test folder classes do not match the trained model classes")

    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images)
            predicted.extend(outputs.argmax(1).tolist())
            actual.extend(labels.tolist())

    report = classification_report(
        actual, predicted, target_names=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(actual, predicted)
    matrix_path = full_path("reports/confusion_matrix.png")
    save_confusion_matrix(matrix, classes, matrix_path)

    metrics = {
        "run_id": info["run_id"],
        "test_images": len(actual),
        "accuracy": round(float(accuracy_score(actual, predicted)), 5),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 5),
        "per_class_f1": {name: round(float(report[name]["f1-score"]), 5) for name in classes},
        "model_size_mb": info["model_size_mb"],
        **measure_latency(net, int(info["image_size"])),
    }
    write_json("reports/metrics.json", metrics)

    mlflow.set_tracking_uri(mlflow_uri())
    with mlflow.start_run(run_id=info["run_id"]):
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "latency_p50_ms": metrics["latency_p50_ms"],
                "latency_p95_ms": metrics["latency_p95_ms"],
                **{f"f1_{name}": score for name, score in metrics["per_class_f1"].items()},
            }
        )
        mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")
        mlflow.log_artifact(str(full_path("reports/metrics.json")), artifact_path="evaluation")

    log_info(
        logger,
        "evaluation finished",
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        latency_p95_ms=metrics["latency_p95_ms"],
    )
    return StageResult(items=len(actual), extra={"accuracy": metrics["accuracy"]})


if __name__ == "__main__":
    run_stage(run, "evaluate")
