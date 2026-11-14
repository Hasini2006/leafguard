"""Shared building blocks: errors, config, logging, git info and stage tracking.

Everything the pipeline stages need that is not specific to images or models lives here,
so each stage file stays short and readable.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- errors


class LeafGuardError(Exception):
    """Base class for every error LeafGuard raises on purpose."""


class ConfigError(LeafGuardError):
    """params.yaml is missing, unreadable or incomplete."""


class DataError(LeafGuardError):
    """Input data failed a quality rule."""


class ImageError(LeafGuardError):
    """A single image could not be decoded or transformed."""


class StageError(LeafGuardError):
    """A pipeline stage could not finish."""


# ---------------------------------------------------------------- config


def project_root() -> Path:
    """Folder that holds params.yaml. DVC runs every stage from there."""
    return Path(os.environ.get("LEAFGUARD_ROOT", Path.cwd())).resolve()


def full_path(path: str | Path) -> Path:
    """Turn a project-relative path into an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Read params.yaml and return it as a dictionary."""
    file = full_path(path)
    if not file.exists():
        raise ConfigError(f"params file not found: {file}")
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"invalid YAML in {file}: {err}") from err
    if not isinstance(data, dict):
        raise ConfigError(f"{file} must hold a mapping at the top level")
    return data


def need(section: dict[str, Any], key: str, where: str) -> Any:
    """Return section[key], or raise an error that names the missing key."""
    if key not in section:
        raise ConfigError(f"missing parameter '{where}.{key}' in params.yaml")
    return section[key]


def write_json(path: str | Path, data: Any) -> Path:
    """Write pretty JSON, creating parent folders as needed."""
    out = full_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning `default` when it does not exist."""
    file = full_path(path)
    if not file.exists():
        return default
    return json.loads(file.read_text(encoding="utf-8"))


def mlflow_uri() -> str:
    """Tracking server address. The MLflow container publishes port 5000 on the VM."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ---------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, so logs can be filtered and parsed."""

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            line.update(fields)
        if record.exc_info:
            line["error"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


_logging_ready = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger under the 'leafguard' namespace."""
    global _logging_ready
    root = logging.getLogger("leafguard")
    if not _logging_ready:
        root.setLevel(os.environ.get("LEAFGUARD_LOG_LEVEL", "INFO").upper())
        root.propagate = False  # keep Spark / py4j noise out of our output
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        root.addHandler(console)
        log_dir = Path(os.environ.get("LEAFGUARD_LOG_DIR", "logs"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                log_dir / "leafguard.log", maxBytes=5_000_000, backupCount=3
            )
            handler.setFormatter(JsonFormatter())
            root.addHandler(handler)
        except OSError:
            root.warning("file logging disabled: cannot write to %s", log_dir)
        _logging_ready = True
    return logging.getLogger(f"leafguard.{name}")


def log_info(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a message together with extra structured fields."""
    logger.info(message, extra={"fields": fields})


def log_warn(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.warning(message, extra={"fields": fields})


# ---------------------------------------------------------------- git info


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def git_commit() -> str:
    """Current commit hash, or 'unknown' outside a git repository."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes."""
    return bool(_git("status", "--porcelain"))


def data_version(dvc_file: str = "data/raw.dvc") -> str:
    """md5 that DVC recorded for the raw dataset: the dataset's version id."""
    meta = full_path(dvc_file)
    if not meta.exists():
        return "unknown"
    try:
        return yaml.safe_load(meta.read_text(encoding="utf-8"))["outs"][0]["md5"]
    except (yaml.YAMLError, KeyError, IndexError, TypeError):
        return "unknown"


# ---------------------------------------------------------------- stage tracking


@dataclass
class StageResult:
    """What a stage hands back: how many items it handled, plus summary values."""

    items: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def runs_log() -> Path:
    return Path(os.environ.get("LEAFGUARD_RUNS_LOG", "reports/pipeline_runs.jsonl"))


def push_metrics(record: dict[str, Any]) -> None:
    """Send stage gauges to the Prometheus Pushgateway. Never raises.

    The Pushgateway only runs once the monitoring stack is up, so a failure here is
    logged and ignored: monitoring must never break the pipeline.
    """
    if os.environ.get("LEAFGUARD_NO_PUSH") == "1":
        return
    logger = get_logger("pipeline")
    try:
        from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

        registry = CollectorRegistry()
        prefix = "leafguard_stage"
        ok = record["status"] == "success"
        gauges = {
            f"{prefix}_duration_seconds": record["duration_s"],
            f"{prefix}_items": record["items"],
            f"{prefix}_throughput": record["items_per_s"],
            f"{prefix}_last_status": 1 if ok else 0,
            f"{prefix}_last_run_timestamp": time.time(),
        }
        if ok:
            gauges[f"{prefix}_last_success_timestamp"] = time.time()
        for name, value in gauges.items():
            Gauge(name, name.replace("_", " "), registry=registry).set(value)
        pushadd_to_gateway(
            os.environ.get("PUSHGATEWAY_URL", "localhost:9091"),
            job="leafguard_pipeline",
            grouping_key={"stage": record["stage"]},
            registry=registry,
            timeout=2,
        )
    except Exception as err:  # noqa: BLE001 - monitoring is best effort
        log_warn(logger, "pushgateway not reachable, skipping", reason=str(err)[:150])


def track_stage(stage: str):
    """Decorate a stage's run() so every execution is timed, logged and recorded.

    Writes one line per run to reports/pipeline_runs.jsonl (success or failure, with the
    error text) and pushes the same numbers to the Pushgateway. The Pipeline Console page
    and the Grafana pipeline dashboard both read from these two places.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            logger = get_logger("pipeline")
            run_id = uuid.uuid4().hex[:12]
            started = time.time()
            status, error, result = "success", None, None
            log_info(logger, "stage started", stage=stage, run_id=run_id)
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as err:
                status = "failed"
                error = f"{type(err).__name__}: {err}"
                logger.error(
                    "stage failed",
                    extra={"fields": {"stage": stage, "run_id": run_id, "error": error}},
                )
                raise
            finally:
                seconds = time.time() - started
                items = result.items if isinstance(result, StageResult) else 0
                record = {
                    "run_id": run_id,
                    "stage": stage,
                    "status": status,
                    "started_at": datetime.fromtimestamp(started, tz=UTC).isoformat(),
                    "duration_s": round(seconds, 3),
                    "items": items,
                    "items_per_s": round(items / seconds, 2) if seconds > 0 else 0.0,
                    "error": error,
                    "git_commit": git_commit()[:8],
                    "extra": result.extra if isinstance(result, StageResult) else {},
                }
                path = runs_log()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
                push_metrics(record)
                if status == "success":
                    log_info(
                        logger,
                        "stage finished",
                        stage=stage,
                        run_id=run_id,
                        duration_s=record["duration_s"],
                        items=items,
                        items_per_s=record["items_per_s"],
                    )

        return wrapper

    return decorate


def run_stage(func, name: str) -> None:
    """Entry point helper: run a stage and turn an expected error into exit code 1."""
    import sys

    try:
        func()
    except LeafGuardError as err:
        print(f"{name} stage failed: {err}", file=sys.stderr)
        sys.exit(1)
