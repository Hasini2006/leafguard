"""Stage 'train': transfer learning on MobileNetV3-Small, tracked in MLflow.

The model is deliberately small. Accuracy is not what this project is marked on, so the
defaults (128 px images, a frozen backbone, 3 epochs) are chosen to finish in a few
minutes on 4 CPU cores rather than to squeeze out the last percent.

Everything needed to reproduce a run is logged to MLflow: parameters, the git commit, the
DVC hash of the dataset, per-epoch metrics and the artifacts. The weights are exported as
TorchScript, so the serving container needs torch but not torchvision.

Run: python -m leafguard.training.train [--epochs 3] [--lr 0.001] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from leafguard.common import (
    StageError,
    StageResult,
    data_version,
    full_path,
    get_logger,
    git_commit,
    git_dirty,
    load_params,
    log_info,
    log_warn,
    mlflow_uri,
    need,
    run_stage,
    track_stage,
    write_json,
)
from leafguard.images import IMAGENET_MEAN, IMAGENET_STD

logger = get_logger("train")

MEAN = IMAGENET_MEAN.tolist()
STD = IMAGENET_STD.tolist()


def build_loaders(
    processed_dir: Path, batch_size: int, workers: int
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create the train and validation loaders from the processed image folders.

    The images on disk are already the right size, so the only training-time augmentation
    is a horizontal flip and a small rotation. Validation stays deterministic.
    """
    train_steps = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    plain_steps = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_set = datasets.ImageFolder(processed_dir / "train", transform=train_steps)
    val_set = datasets.ImageFolder(processed_dir / "val", transform=plain_steps)
    if train_set.classes != val_set.classes:
        raise StageError("train and val folders hold different classes")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, train_set.classes


def build_model(class_count: int, freeze: bool) -> nn.Module:
    """MobileNetV3-Small with a fresh classifier head sized to our classes."""
    try:
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        pretrained = True
    except Exception as err:  # noqa: BLE001 - offline VM, carry on from scratch
        log_warn(
            logger, "pretrained weights unavailable, training from scratch", reason=str(err)[:150]
        )
        net = models.mobilenet_v3_small(weights=None)
        pretrained = False

    if freeze and pretrained:
        for parameter in net.features.parameters():
            parameter.requires_grad = False

    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, class_count)
    net.pretrained = pretrained  # remembered so it can be logged as a parameter
    return net


def run_epoch(
    net: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimiser: Any | None
) -> tuple[float, float]:
    """One pass over a loader. Passing an optimiser trains, passing None evaluates."""
    training = optimiser is not None
    net.train() if training else net.eval()
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimiser.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            if training:
                loss.backward()
                optimiser.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((outputs.argmax(1) == labels).sum().item())
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def export_torchscript(net: nn.Module, image_size: int, path: Path) -> None:
    """Freeze the trained network into a standalone TorchScript file."""
    net.eval()
    example = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        traced = torch.jit.trace(net, example)
    path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path))


@track_stage("train")
def run(
    params: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None
) -> StageResult:
    params = params or load_params()
    data = need(params, "data", "root")
    config = dict(need(params, "train", "root"))
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})

    seed = int(params.get("seed", 42))
    image_size = int(need(data, "image_size", "data"))
    processed_dir = full_path(need(data, "processed_dir", "data"))

    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("num_threads", 4)))

    train_loader, val_loader, classes = build_loaders(
        processed_dir, int(config["batch_size"]), int(config.get("num_workers", 2))
    )
    net = build_model(len(classes), bool(config.get("freeze_backbone", True)))
    loss_fn = nn.CrossEntropyLoss()
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=float(config["lr"]))

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("leafguard")
    # Autolog first. mlflow.pytorch.autolog only hooks PyTorch Lightning, so for this plain
    # training loop it records little by itself; that is why everything below is also
    # logged explicitly. log_system_metrics adds the automatic CPU / memory / disk
    # sampling for the run, which plain autolog does not.
    mlflow.autolog(log_models=False, silent=True)

    with mlflow.start_run(log_system_metrics=True) as active:
        run_id = active.info.run_id
        log_dataset_lineage(data)
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "git_dirty": str(git_dirty()),
                "data_version": data_version(),
                "stage": "train",
                "host": f"{platform.system()} {platform.machine()}",
            }
        )
        mlflow.log_params(
            {
                "backbone": config.get("backbone", "mobilenet_v3_small"),
                "pretrained": getattr(net, "pretrained", False),
                "freeze_backbone": config.get("freeze_backbone", True),
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["lr"],
                "image_size": image_size,
                "seed": seed,
                "class_count": len(classes),
                "classes": ",".join(classes),
                "train_images": len(train_loader.dataset),
                "val_images": len(val_loader.dataset),
                "num_threads": config.get("num_threads", 4),
            }
        )

        best_accuracy, history = 0.0, []
        for epoch in range(1, int(config["epochs"]) + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(net, train_loader, loss_fn, optimiser)
            val_loss, val_accuracy = run_epoch(net, val_loader, loss_fn, None)
            seconds = time.perf_counter() - started
            best_accuracy = max(best_accuracy, val_accuracy)

            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 5),
                    "train_accuracy": round(train_accuracy, 5),
                    "val_loss": round(val_loss, 5),
                    "val_accuracy": round(val_accuracy, 5),
                    "epoch_seconds": round(seconds, 2),
                    "train_images_per_second": round(len(train_loader.dataset) / seconds, 2),
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 5),
                    "val_accuracy": round(val_accuracy, 5),
                    "seconds": round(seconds, 2),
                }
            )
            log_info(
                logger,
                "epoch done",
                epoch=epoch,
                val_accuracy=round(val_accuracy, 4),
                seconds=round(seconds, 1),
            )

        model_path = full_path("models/model.pt")
        export_torchscript(net, image_size, model_path)
        size_mb = round(model_path.stat().st_size / 1e6, 3)

        classes_file = write_json(
            "models/classes.json", {"classes": classes, "image_size": image_size}
        )
        info = {
            "run_id": run_id,
            "experiment": "leafguard",
            "git_commit": git_commit(),
            "data_version": data_version(),
            "classes": classes,
            "image_size": image_size,
            "best_val_accuracy": round(best_accuracy, 5),
            "model_size_mb": size_mb,
            "history": history,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_json("models/run.json", info)
        mlflow.log_dict(info, "metadata/run.json")

        mlflow.log_metrics(
            {
                "best_val_accuracy": round(best_accuracy, 5),
                "model_size_mb": size_mb,
                "parameter_count": sum(p.numel() for p in net.parameters()),
            }
        )
        mlflow.log_artifact(str(classes_file), artifact_path="metadata")
        mlflow.log_artifact(str(full_path("params.yaml")), artifact_path="metadata")
        baseline = full_path(data.get("baseline_path", "models/baseline.json"))
        if baseline.exists():
            mlflow.log_artifact(str(baseline), artifact_path="metadata")
        quality = full_path("reports/data_quality.json")
        if quality.exists():
            mlflow.log_artifact(str(quality), artifact_path="metadata")

    log_info(logger, "training finished", run_id=run_id, best_val_accuracy=round(best_accuracy, 4))
    return StageResult(
        items=len(train_loader.dataset),
        extra={"run_id": run_id, "val_accuracy": round(best_accuracy, 4)},
    )


def log_dataset_lineage(data: dict[str, Any]) -> None:
    """Record the exact train/val/test split as an MLflow dataset input (beyond autolog).

    The run page then shows which images, by digest, the model was trained on. Failure is
    logged and ignored: lineage is useful, but never worth failing a training run over.
    """
    try:
        import pandas as pd

        splits_path = full_path(data.get("splits_path", "data/interim/splits.parquet"))
        frame = pd.read_parquet(splits_path)
        dataset = mlflow.data.from_pandas(frame, source=str(splits_path), name="leafguard-splits")
        mlflow.log_input(dataset, context="training")
        mlflow.log_dict(
            (
                {str(k): int(v) for k, v in frame.groupby("split").size().items()}
                if "split" in frame.columns
                else {"rows": len(frame)}
            ),
            "metadata/split_counts.json",
        )
    except Exception as err:  # lineage must never break a training run
        log_info(logger, "dataset lineage not logged", reason=str(err)[:150])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LeafGuard classifier")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    args = parser.parse_args()
    overrides = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}
    run_stage(lambda: run(overrides=overrides), "train")


if __name__ == "__main__":
    main()
