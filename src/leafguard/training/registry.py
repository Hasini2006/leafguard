"""Stage 'gate': register the candidate model and decide whether it becomes champion.

A new model is never served just because training finished. This stage logs the candidate
as an MLflow pyfunc model, registers it as a new version, marks it 'challenger', and only
moves the 'champion' alias when every rule in the gate section of params.yaml passes:

  accuracy on the golden test set is at least gate.min_accuracy
  accuracy has not dropped more than gate.max_drop_vs_champion below the current champion
  p95 CPU latency is within gate.max_latency_ms

The decision and the reasons behind it are written to reports/gate.json. The model server
loads models:/leafguard@champion, so promoting or rolling back is an alias change plus a
container restart, with no code change anywhere.

Commands:
  python -m leafguard.training.registry promote            run the gate (the DVC stage)
  python -m leafguard.training.registry rollback --to 1    move champion to an older version
  python -m leafguard.training.registry list               show versions and aliases
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from leafguard.common import (
    StageError,
    StageResult,
    data_version,
    full_path,
    get_logger,
    git_commit,
    load_params,
    log_info,
    log_warn,
    mlflow_uri,
    need,
    read_json,
    run_stage,
    track_stage,
    write_json,
)

logger = get_logger("registry")

CHAMPION = "champion"
CHALLENGER = "challenger"


def client() -> MlflowClient:
    mlflow.set_tracking_uri(mlflow_uri())
    return MlflowClient()


def alias_version(api: MlflowClient, name: str, alias: str) -> Any | None:
    """Return the model version behind an alias, or None when the alias is not set yet."""
    try:
        return api.get_model_version_by_alias(name, alias)
    except Exception:  # noqa: BLE001 - MLflow raises different errors for "not found"
        return None


def log_candidate(name: str, run_id: str, classes: list[str]) -> str:
    """Log the pyfunc model into the training run and register a new version.

    The wrapper is logged as a model from code (its file path, not a pickled object), so
    the model server never has to unpickle a class built by a different Python version.
    """
    example = pd.DataFrame({"image_b64": ["aGVsbG8="]})
    output = pd.DataFrame([[0.0] * len(classes)], columns=classes)

    mlflow.set_tracking_uri(mlflow_uri())
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(full_path("src/leafguard/training/wrapper.py")),
            artifacts={
                "weights": str(full_path("models/model.pt")),
                "classes": str(full_path("models/classes.json")),
            },
            signature=mlflow.models.infer_signature(example, output),
            registered_model_name=name,
            pip_requirements=["torch", "numpy", "pandas", "pillow", "mlflow"],
        )
    return info.registered_model_version


def decide(
    metrics: dict[str, Any], rules: dict[str, Any], champion_accuracy: float | None
) -> tuple[bool, list[str]]:
    """Apply the gate rules and return the verdict with a readable reason for each."""
    reasons, passed = [], True

    accuracy = float(metrics["accuracy"])
    floor = float(rules["min_accuracy"])
    if accuracy >= floor:
        reasons.append(f"accuracy {accuracy:.3f} meets the minimum of {floor:.3f}")
    else:
        passed = False
        reasons.append(f"accuracy {accuracy:.3f} is below the minimum of {floor:.3f}")

    if champion_accuracy is None:
        reasons.append("no champion exists yet, so there is nothing to regress against")
    else:
        allowed = float(rules["max_drop_vs_champion"])
        drop = champion_accuracy - accuracy
        if drop <= allowed:
            reasons.append(
                f"accuracy is within {allowed:.3f} of the champion ({champion_accuracy:.3f})"
            )
        else:
            passed = False
            reasons.append(
                f"accuracy dropped {drop:.3f} below the champion ({champion_accuracy:.3f}), "
                f"more than the allowed {allowed:.3f}"
            )

    latency = float(metrics["latency_p95_ms"])
    budget = float(rules["max_latency_ms"])
    if latency <= budget:
        reasons.append(f"p95 latency {latency:.1f} ms is within the {budget:.0f} ms budget")
    else:
        passed = False
        reasons.append(f"p95 latency {latency:.1f} ms exceeds the {budget:.0f} ms budget")

    return passed, reasons


def push_model_gauge(name: str, version: str, accuracy: float) -> None:
    """Publish the champion's accuracy so Grafana can show which version is serving."""
    if os.environ.get("LEAFGUARD_NO_PUSH") == "1":
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

        registry = CollectorRegistry()
        Gauge(
            "leafguard_champion_accuracy",
            "Golden-set accuracy of the champion",
            ["version"],
            registry=registry,
        ).labels(version=version).set(accuracy)
        pushadd_to_gateway(
            os.environ.get("PUSHGATEWAY_URL", "localhost:9091"),
            job="leafguard_model",
            grouping_key={"model": name},
            registry=registry,
            timeout=2,
        )
    except Exception as err:  # noqa: BLE001
        log_warn(logger, "could not publish the model gauge", reason=str(err)[:150])


@track_stage("gate")
def run(params: dict[str, Any] | None = None) -> StageResult:
    params = params or load_params()
    rules = need(params, "gate", "root")
    name = need(params, "serving", "root")["model_name"]

    info = read_json("models/run.json")
    metrics = read_json("reports/metrics.json")
    if info is None or metrics is None:
        raise StageError("run the train and evaluate stages before the gate")

    api = client()
    version = log_candidate(name, info["run_id"], info["classes"])
    api.set_registered_model_alias(name, CHALLENGER, version)
    api.set_model_version_tag(name, version, "accuracy", f"{metrics['accuracy']:.5f}")
    api.set_model_version_tag(name, version, "macro_f1", f"{metrics['macro_f1']:.5f}")
    api.set_model_version_tag(name, version, "git_commit", git_commit()[:8])
    api.set_model_version_tag(name, version, "data_version", data_version()[:8])

    champion = alias_version(api, name, CHAMPION)
    champion_accuracy = None
    if champion is not None and champion.version != version:
        champion_accuracy = float(champion.tags.get("accuracy", 0.0))

    passed, reasons = decide(metrics, rules, champion_accuracy)
    if passed:
        api.set_registered_model_alias(name, CHAMPION, version)
        push_model_gauge(name, version, float(metrics["accuracy"]))
    api.set_model_version_tag(name, version, "gate", "passed" if passed else "failed")

    decision = {
        "model": name,
        "candidate_version": version,
        "promoted": passed,
        "reasons": reasons,
        "candidate_accuracy": metrics["accuracy"],
        "candidate_macro_f1": metrics["macro_f1"],
        "candidate_latency_p95_ms": metrics["latency_p95_ms"],
        "previous_champion_version": champion.version if champion else None,
        "previous_champion_accuracy": champion_accuracy,
        "serving_version": version if passed else (champion.version if champion else None),
        "run_id": info["run_id"],
        "git_commit": git_commit()[:8],
    }
    write_json("reports/gate.json", decision)
    log_info(
        logger, "gate decision", promoted=passed, version=version, accuracy=metrics["accuracy"]
    )
    return StageResult(items=1, extra={"promoted": passed, "version": version})


def rollback(target: str, params: dict[str, Any] | None = None) -> None:
    """Point the champion alias back at an earlier version."""
    params = params or load_params()
    name = need(params, "serving", "root")["model_name"]
    api = client()
    api.get_model_version(name, target)  # raises if the version does not exist
    api.set_registered_model_alias(name, CHAMPION, target)
    write_json(
        "reports/gate.json",
        {
            "model": name,
            "promoted": False,
            "serving_version": target,
            "reasons": [f"manual rollback: champion moved to version {target}"],
        },
    )
    print(f"champion is now version {target}. Restart the model server to load it:")
    print("  docker compose restart model-server")


def show(params: dict[str, Any] | None = None) -> None:
    """Print every registered version with its aliases and accuracy."""
    params = params or load_params()
    name = need(params, "serving", "root")["model_name"]
    api = client()
    # Version search results do not carry aliases in MLflow 3, so read them from the model.
    by_version: dict[str, list[str]] = {}
    for alias, number in api.get_registered_model(name).aliases.items():
        by_version.setdefault(str(number), []).append(alias)
    print(f"{'version':<9}{'aliases':<24}{'accuracy':<10}{'gate':<9}commit")
    for version in sorted(
        api.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)
    ):
        aliases = ",".join(sorted(by_version.get(str(version.version), []))) or "-"
        print(
            f"{version.version:<9}{aliases:<24}{version.tags.get('accuracy', '-'):<10}"
            f"{version.tags.get('gate', '-'):<9}{version.tags.get('git_commit', '-')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LeafGuard model registry operations")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("promote", help="run the promotion gate (used by the DVC stage)")
    back = commands.add_parser("rollback", help="move champion to an earlier version")
    back.add_argument("--to", required=True, help="version number to serve")
    commands.add_parser("list", help="show registered versions and aliases")
    args = parser.parse_args()

    if args.command == "promote":
        run_stage(run, "gate")
    elif args.command == "rollback":
        rollback(args.to)
    else:
        show()


if __name__ == "__main__":
    main()
