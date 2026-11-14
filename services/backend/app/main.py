from __future__ import annotations

import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.config import settings
from app.schemas import (
    ClassInfo,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfo,
    ModelVersion,
    PipelineSummary,
    PredictResponse,
    ReadyResponse,
    StageRun,
)
from app.services import (
    CLASS_INFO,
    DISCLAIMER,
    UNCLEAR_HINT,
    DriftWatcher,
    FeedbackStore,
    ModelClient,
    RegistryReader,
    ValidationFailed,
    check_upload,
    describe,
    read_report,
    read_runs,
)
from leafguard.common import get_logger, log_info, log_warn

logger = get_logger("api")
RECENT_UPLOADS: OrderedDict[str, bytes] = OrderedDict()
RECENT_LIMIT = 20

model_client: ModelClient
registry: RegistryReader
drift: DriftWatcher
feedback_store: FeedbackStore


def publish_model_info(details: dict[str, Any]) -> None:
    """Expose which version is serving. Old labels are cleared so only one series is 1."""
    metrics.model_info.clear()
    metrics.model_info.labels(version=details["version"], alias=details["alias"]).set(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the long-lived helpers once, when the service starts."""
    global model_client, registry, drift, feedback_store
    model_client = ModelClient()
    registry = RegistryReader()
    drift = DriftWatcher()
    feedback_store = FeedbackStore()
    details = registry.champion()
    publish_model_info(details)
    log_info(
        logger,
        "backend ready",
        model_server=settings.model_server_url,
        model_version=details["version"],
    )
    yield


app = FastAPI(
    title="LeafGuard API",
    version="1.0.0",
    description="Tomato leaf disease diagnosis from an uploaded photo.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API = "/api/v1"


def fail(request_id: str, code: str, message: str, status_code: int) -> JSONResponse:
    """Build the single error shape every endpoint returns."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )


@app.middleware("http")
async def observe(request: Request, call_next):
    """Give every request an id, time it, and count it, whatever the outcome."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    endpoint = request.url.path
    with metrics.request_seconds.labels(endpoint=endpoint).time():
        response = await call_next(request)
    metrics.requests_total.labels(
        endpoint=endpoint, method=request.method, status=str(response.status_code)
    ).inc()
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(ValidationFailed)
async def handle_validation(request: Request, err: ValidationFailed) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    if err.status_code in (502, 503):
        metrics.model_errors_total.labels(kind=err.code).inc()
    else:
        metrics.rejected_total.labels(reason=err.code).inc()
    log_warn(logger, "request rejected", request_id=request_id, code=err.code)
    return fail(request_id, err.code, err.message, err.status_code)


@app.exception_handler(RequestValidationError)
async def handle_bad_request(request: Request, err: RequestValidationError) -> JSONResponse:
    """A malformed request (missing file, wrong JSON) gets the same error shape as the rest."""
    request_id = getattr(request.state, "request_id", "unknown")
    fields = ", ".join(".".join(str(part) for part in e["loc"][1:]) for e in err.errors())
    metrics.rejected_total.labels(reason="INVALID_REQUEST").inc()
    log_warn(logger, "malformed request", request_id=request_id, fields=fields)
    return fail(
        request_id,
        "INVALID_REQUEST",
        f"The request is missing or has invalid fields: {fields or 'body'}.",
        422,
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, err: Exception) -> JSONResponse:
    """Nothing unexpected should ever reach the user as a stack trace."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled error", extra={"fields": {"request_id": request_id}})
    return fail(
        request_id, "INTERNAL_ERROR", "Something went wrong on our side. Please try again.", 500
    )


# ------------------------------------------------------------------ health


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> dict[str, str]:
    """Liveness: the process is running. Used by the container healthcheck."""
    return {"status": "ok"}


@app.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["health"],
)
def ready(request: Request):
    """Readiness: the model server is reachable, so predictions can be served."""
    details = registry.champion()
    if not model_client.alive():
        return fail(
            request.state.request_id,
            "MODEL_UNAVAILABLE",
            "The diagnosis service is not reachable yet.",
            503,
        )
    return {"ready": True, "model_server": "up", "model_version": details["version"]}


@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics() -> PlainTextResponse:
    """Everything Prometheus scrapes from this service."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ prediction


@app.post(
    f"{API}/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["diagnosis"],
)
async def predict(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Diagnose one tomato leaf photo.

    Returns status 'uncertain' instead of a confident-looking label when the top score is
    below the threshold or the photo does not look like a leaf, so an unusual picture never
    comes back as a definite diagnosis.
    """
    request_id = request.state.request_id
    raw = await file.read()
    checked = check_upload(raw, file.content_type)

    # Measured on the image resized exactly as the training baseline was (see DriftWatcher).
    for feature, value in drift.record(drift.measure(checked.raw, checked.stats)).items():
        metrics.drift_score.labels(feature=feature).set(value)

    with metrics.model_seconds.time():
        scores, inference_ms = model_client.predict(checked.raw)

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_name, best_score = ranked[0]
    uncertain = best_score < settings.confidence_threshold or not checked.looks_like_leaf
    status = "uncertain" if uncertain else "ok"
    words = describe(best_name)

    metrics.predictions_total.labels(predicted_class=best_name, status=status).inc()
    metrics.confidence.observe(best_score)

    RECENT_UPLOADS[request_id] = checked.raw
    while len(RECENT_UPLOADS) > RECENT_LIMIT:
        RECENT_UPLOADS.popitem(last=False)

    details = registry.champion()
    publish_model_info(details)
    log_info(
        logger,
        "prediction served",
        request_id=request_id,
        predicted=best_name,
        confidence=round(best_score, 4),
        status=status,
        inference_ms=inference_ms,
    )

    return {
        "request_id": request_id,
        "status": status,
        "predicted_class": best_name,
        "display_name": words["display_name"],
        "confidence": round(best_score, 4),
        "top_k": [
            {
                "name": name,
                "display_name": describe(name)["display_name"],
                "confidence": round(score, 4),
            }
            for name, score in ranked[:3]
        ],
        "advice": words["advice"] if not uncertain else "",
        "hint": UNCLEAR_HINT if uncertain else None,
        "model_version": details["version"],
        "inference_ms": inference_ms,
        "disclaimer": DISCLAIMER,
    }


@app.post(
    f"{API}/feedback",
    response_model=FeedbackResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["diagnosis"],
)
def submit_feedback(request: Request, body: FeedbackRequest):
    """Record whether a prediction was right, so the image can be used for retraining."""
    if not body.correct and not body.correct_class:
        return fail(
            request.state.request_id, "MISSING_CLASS", "Tell us which disease it actually was.", 422
        )
    if body.correct_class and body.correct_class not in CLASS_INFO:
        return fail(
            request.state.request_id,
            "UNKNOWN_CLASS",
            "That is not one of the classes this model knows.",
            422,
        )

    metrics.feedback_total.labels(agreed=str(body.correct).lower()).inc()
    if not body.correct:
        feedback_store.save(
            body.request_id, body.correct_class, RECENT_UPLOADS.get(body.request_id), body.comment
        )
    log_info(logger, "feedback stored", request_id=body.request_id, correct=body.correct)
    return {"stored": True, "message": "Thank you. This helps improve the next model."}


@app.get(f"{API}/classes", response_model=list[ClassInfo], tags=["reference"])
def classes() -> list[dict[str, str]]:
    """Every class the model knows, with its plain-language description and advice."""
    return [{"name": name, **info} for name, info in CLASS_INFO.items()]


@app.get(f"{API}/model/info", response_model=ModelInfo, tags=["reference"])
def model_details() -> dict[str, Any]:
    """Which model version is serving right now, and how it scored when promoted."""
    details = registry.champion(refresh=True)
    publish_model_info(details)
    return {
        **details,
        "classes": list(CLASS_INFO),
        "confidence_threshold": settings.confidence_threshold,
    }


@app.get(f"{API}/models", response_model=list[ModelVersion], tags=["reference"])
def model_versions() -> list[dict[str, Any]]:
    """Every registered model version with its aliases and gate result, newest first.

    Read-only on purpose: promotion goes through the pipeline's gate and rollback through
    `make rollback`, so nothing in the browser can change which model is live.
    """
    return registry.versions()


# ------------------------------------------------------------------ pipeline console


@app.get(f"{API}/pipeline/runs", response_model=list[StageRun], tags=["pipeline"])
def pipeline_runs(limit: int = 50, stage: str | None = None) -> list[dict[str, Any]]:
    """Recent pipeline stage runs, newest first, including the ones that failed."""
    return read_runs(limit=limit, stage=stage)


@app.get(f"{API}/pipeline/summary", response_model=PipelineSummary, tags=["pipeline"])
def pipeline_summary() -> dict[str, Any]:
    """The newest run of each stage plus the latest data, throughput and gate reports."""
    newest: dict[str, dict[str, Any]] = {}
    for run in read_runs(limit=500):
        newest.setdefault(run["stage"], run)
    order = ["lint", "unit_tests", "validate", "split", "preprocess", "train", "evaluate", "gate"]
    stages = [newest[name] for name in order if name in newest]
    stages += [run for name, run in newest.items() if name not in order]
    return {
        "stages": stages,
        "gate": read_report("gate.json"),
        "data_quality": read_report("data_quality.json"),
        "throughput": read_report("throughput.json"),
        "metrics": read_report("metrics.json"),
    }


@app.get(f"{API}/pipeline/dag", response_class=PlainTextResponse, tags=["pipeline"])
def pipeline_dag() -> str:
    """The pipeline graph as Mermaid text, rendered by the Pipeline Console page.

    Written out rather than shelled out to `dvc dag`, because the backend container has
    neither DVC nor the repository mounted.
    """
    return (
        "flowchart LR\n"
        "  lint[lint] --> tests[unit_tests]\n"
        "  tests --> validate[validate: Spark]\n"
        "  raw[(data/raw)] --> validate\n"
        "  validate --> split[split]\n"
        "  split --> pre[preprocess: Spark]\n"
        "  pre --> train[train: PyTorch + MLflow]\n"
        "  train --> eval[evaluate]\n"
        "  eval --> gate{gate}\n"
        "  gate -->|promoted| champ[champion alias]\n"
        "  gate -->|rejected| keep[champion unchanged]\n"
    )
