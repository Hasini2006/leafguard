from __future__ import annotations

import base64
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from leafguard.common import ImageError, get_logger, log_warn
from leafguard.drift import psi
from leafguard.images import decode, image_stats, resize_square

logger = get_logger("backend")
CLASS_INFO: dict[str, dict[str, str]] = {
    "healthy": {
        "display_name": "Healthy",
        "description": "No signs of the diseases this model was trained to recognise.",
        "advice": "Keep watering at the base of the plant and check new growth weekly.",
    },
    "early_blight": {
        "display_name": "Early blight",
        "description": "Dark spots with rings, usually starting on the older lower leaves.",
        "advice": "Remove affected lower leaves, avoid wetting the foliage, and mulch the soil.",
    },
    "late_blight": {
        "display_name": "Late blight",
        "description": "Large greasy grey-brown patches that spread quickly in cool damp weather.",
        "advice": "Act quickly: remove affected plants, improve airflow, do not compost them.",
    },
    "bacterial_spot": {
        "display_name": "Bacterial spot",
        "description": "Many small dark water-soaked spots on leaves and sometimes fruit.",
        "advice": "Avoid overhead watering, remove affected leaves, and use clean seed.",
    },
    "leaf_mold": {
        "display_name": "Leaf mould",
        "description": "Pale yellow patches above, with olive-brown fuzzy growth underneath.",
        "advice": "Lower the humidity, increase ventilation, and space plants further apart.",
    },
    "septoria": {
        "display_name": "Septoria leaf spot",
        "description": "Many small circular spots with pale centres and dark edges.",
        "advice": "Remove affected leaves, mulch to stop soil splash, and rotate crops.",
    },
}

DISCLAIMER = (
    "This is an automated suggestion from an educational prototype. "
    "Confirm with an agricultural expert before treating your crop."
)
UNCLEAR_HINT = (
    "We are not confident about this one. Try a closer, sharper photo of a "
    "single leaf on a plain background in daylight."
)


def describe(name: str) -> dict[str, str]:
    """Look up the plain-language text for a class, with a safe fallback."""
    return CLASS_INFO.get(
        name,
        {
            "display_name": name.replace("_", " ").capitalize(),
            "description": "No description available for this class.",
            "advice": "Check the plant and confirm with an expert.",
        },
    )


class ValidationFailed(Exception):
    """An upload was rejected before it ever reached the model."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class CheckedImage:
    """An upload that passed every check, together with its measured statistics."""

    raw: bytes
    stats: dict[str, float]
    looks_like_leaf: bool


def check_upload(raw: bytes, content_type: str | None) -> CheckedImage:
    """Reject anything that is not a usable leaf photo, with a reason a person can act on."""
    if not raw:
        raise ValidationFailed("NO_FILE", "No file was uploaded. Choose an image and try again.")

    if len(raw) > settings.max_upload_bytes:
        raise ValidationFailed(
            "FILE_TOO_LARGE",
            f"That image is larger than {settings.max_upload_mb:.0f} MB. "
            "Please use a smaller photo.",
            status_code=413,
        )

    if content_type and not content_type.startswith("image/"):
        raise ValidationFailed(
            "UNSUPPORTED_TYPE",
            "That file is not an image. Please upload a JPG, PNG or WEBP photo.",
            status_code=415,
        )

    try:
        image = decode(raw)
    except ImageError as err:
        raise ValidationFailed(
            "INVALID_IMAGE",
            "That file could not be read as an image. It may be damaged.",
        ) from err

    width, height = image.size
    if min(width, height) < settings.min_side_px:
        raise ValidationFailed(
            "IMAGE_TOO_SMALL",
            f"That image is only {width} by {height} pixels. "
            f"Please use one at least {settings.min_side_px} pixels on each side.",
        )

    stats = image_stats(image)
    return CheckedImage(
        raw=raw,
        stats=stats,
        looks_like_leaf=stats["green_ratio"] >= settings.min_green_ratio,
    )


class ModelClient:
    """Talks to the MLflow model server over REST. The backend never loads the model."""

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or settings.model_server_url).rstrip("/")
        self.http = httpx.Client(timeout=settings.request_timeout_s)

    def alive(self) -> bool:
        try:
            return self.http.get(f"{self.url}/ping", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def predict(self, raw: bytes) -> tuple[dict[str, float], float]:
        """Send one image and get back {class: score} plus how long the call took."""
        payload = {
            "dataframe_split": {
                "columns": ["image_b64"],
                "data": [[base64.b64encode(raw).decode()]],
            }
        }
        started = time.perf_counter()
        try:
            reply = self.http.post(f"{self.url}/invocations", json=payload)
        except httpx.HTTPError as err:
            raise ValidationFailed(
                "MODEL_UNAVAILABLE",
                "The diagnosis service is starting up or unavailable. Please try again shortly.",
                status_code=503,
            ) from err
        elapsed_ms = (time.perf_counter() - started) * 1000

        if reply.status_code != 200:
            raise ValidationFailed(
                "MODEL_ERROR",
                "The diagnosis service could not process that image.",
                status_code=502,
            )

        body = reply.json()
        rows = body["predictions"] if isinstance(body, dict) else body
        if not rows:
            raise ValidationFailed(
                "MODEL_ERROR", "The diagnosis service returned no result.", status_code=502
            )
        return {str(k): float(v) for k, v in rows[0].items()}, round(elapsed_ms, 2)


class RegistryReader:
    """Reads the serving model's version and tags from the MLflow REST API."""

    def __init__(self) -> None:
        self.url = settings.mlflow_url.rstrip("/")
        self.http = httpx.Client(timeout=5)
        self.cached: dict[str, Any] | None = None
        self.cached_at = 0.0

    def champion(self, refresh: bool = False) -> dict[str, Any]:
        """Details of the model behind the champion alias.

        Cached for settings.registry_ttl_s seconds, so a promotion or rollback shows up in
        responses shortly after it happens without asking MLflow on every request.
        """
        fresh = time.monotonic() - self.cached_at < settings.registry_ttl_s
        if self.cached and fresh and not refresh:
            return self.cached
        details = {
            "name": settings.model_name,
            "version": "unknown",
            "alias": "champion",
            "accuracy": None,
            "git_commit": None,
        }
        try:
            reply = self.http.get(
                f"{self.url}/api/2.0/mlflow/registered-models/alias",
                params={"name": settings.model_name, "alias": "champion"},
            )
            if reply.status_code == 200:
                version = reply.json()["model_version"]
                tags = {tag["key"]: tag["value"] for tag in version.get("tags", [])}
                details.update(
                    version=str(version["version"]),
                    accuracy=tags.get("accuracy"),
                    git_commit=tags.get("git_commit"),
                )
        except (httpx.HTTPError, KeyError, ValueError) as err:
            log_warn(logger, "could not read the model version from MLflow", reason=str(err)[:150])
        self.cached = details
        self.cached_at = time.monotonic()
        return details

    def versions(self) -> list[dict[str, Any]]:
        name = settings.model_name
        try:
            model = self.http.get(
                f"{self.url}/api/2.0/mlflow/registered-models/get", params={"name": name}
            )
            found = self.http.get(
                f"{self.url}/api/2.0/mlflow/model-versions/search",
                params={"filter": f"name='{name}'", "max_results": 200},
            )
            if model.status_code != 200 or found.status_code != 200:
                return []
            aliases: dict[str, list[str]] = {}
            for entry in model.json().get("registered_model", {}).get("aliases", []):
                aliases.setdefault(str(entry["version"]), []).append(entry["alias"])
            rows = []
            for version in found.json().get("model_versions", []):
                tags = {tag["key"]: tag["value"] for tag in version.get("tags", [])}
                created_ms = version.get("creation_timestamp")
                created = (
                    datetime.fromtimestamp(int(created_ms) / 1000, tz=UTC).isoformat()
                    if created_ms
                    else None
                )
                rows.append(
                    {
                        "version": str(version["version"]),
                        "aliases": sorted(aliases.get(str(version["version"]), [])),
                        "accuracy": tags.get("accuracy"),
                        "gate": tags.get("gate"),
                        "git_commit": tags.get("git_commit"),
                        "run_id": version.get("run_id"),
                        "created_at": created,
                    }
                )
            return sorted(rows, key=lambda row: int(row["version"]), reverse=True)
        except (httpx.HTTPError, KeyError, ValueError) as err:
            log_warn(logger, "could not list model versions", reason=str(err)[:150])
            return []


class DriftWatcher:
    def __init__(self) -> None:
        self.window = settings.drift_window
        self.recent: dict[str, deque] = {}
        self.baseline: dict[str, Any] = {}
        self.image_size: int | None = None
        path = Path(settings.baseline_path)
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                self.baseline = stored.get("features", {})
                self.image_size = stored.get("image_size")
            except (OSError, ValueError) as err:
                log_warn(logger, "drift baseline could not be read", reason=str(err)[:150])
        for feature in self.baseline:
            self.recent[feature] = deque(maxlen=self.window)

    def measure(self, raw: bytes, fallback: dict[str, float]) -> dict[str, float]:
        """Measure an upload exactly as the preprocess stage measured the training images.

        The baseline was built from images resized to `image_size` and re-encoded as JPEG,
        and features such as blur depend strongly on resolution. Measuring the original
        upload instead would report drift for perfectly normal photos.
        """
        if not self.image_size:
            return fallback
        try:
            return image_stats(decode(resize_square(raw, int(self.image_size))))
        except ImageError:
            return fallback

    def record(self, stats: dict[str, float]) -> dict[str, float]:
        """Add one upload's statistics and return the current PSI for each feature."""
        scores: dict[str, float] = {}
        for feature, reference in self.baseline.items():
            if feature not in stats:
                continue
            self.recent[feature].append(stats[feature])
            if len(self.recent[feature]) >= max(10, self.window // 5):
                scores[feature] = psi(reference, list(self.recent[feature]))
        return scores


class FeedbackStore:
    def __init__(self) -> None:
        self.folder = Path(settings.feedback_dir)
        self.cipher = None
        if settings.fernet_key and settings.fernet_key != "replace-me":
            try:
                from cryptography.fernet import Fernet

                self.cipher = Fernet(settings.fernet_key.encode())
            except Exception as err:  # noqa: BLE001
                log_warn(
                    logger,
                    "FERNET_KEY is not usable, images will not be stored",
                    reason=str(err)[:150],
                )

    def save(self, request_id: str, label: str, raw: bytes | None, note: str | None) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        stored_image = None
        if raw and self.cipher:
            target = self.folder / label / f"{request_id}.bin"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.cipher.encrypt(raw))
            stored_image = str(target.relative_to(self.folder))
        record = {
            "request_id": request_id,
            "label": label,
            "image": stored_image,
            "encrypted": bool(self.cipher),
            "comment": note,
            "at": datetime.now(tz=UTC).isoformat(),
        }
        with (self.folder / "feedback.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def read_runs(limit: int = 100, stage: str | None = None) -> list[dict[str, Any]]:
    path = Path(settings.reports_dir) / "pipeline_runs.jsonl"
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if stage and record.get("stage") != stage:
            continue
        runs.append(record)
    return list(reversed(runs))[:limit]


def read_report(name: str) -> dict[str, Any] | None:
    path = Path(settings.reports_dir) / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
