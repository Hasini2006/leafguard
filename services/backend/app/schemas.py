from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable code, e.g. INVALID_IMAGE")
    message: str = Field(description="Sentence a non-technical user can act on")
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ClassScore(BaseModel):
    name: str
    display_name: str
    confidence: float


class PredictResponse(BaseModel):
    request_id: str
    status: str = Field(description="'ok' when confident, 'uncertain' when not")
    predicted_class: str
    display_name: str
    confidence: float
    top_k: list[ClassScore]
    advice: str
    hint: str | None = Field(default=None, description="Shown when status is 'uncertain'")
    model_version: str
    inference_ms: float
    disclaimer: str


class FeedbackRequest(BaseModel):
    request_id: str
    correct: bool = Field(description="True if the prediction was right")
    correct_class: str | None = Field(default=None, description="Required when correct is False")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    stored: bool
    message: str


class ClassInfo(BaseModel):
    name: str
    display_name: str
    description: str
    advice: str


class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str
    accuracy: str | None = None
    git_commit: str | None = None
    classes: list[str]
    confidence_threshold: float


class ModelVersion(BaseModel):
    version: str
    aliases: list[str] = Field(description="e.g. ['champion']; empty when none points here")
    accuracy: str | None = Field(default=None, description="Golden-set accuracy at the gate")
    gate: str | None = Field(default=None, description="'passed' or 'failed'")
    git_commit: str | None = None
    run_id: str | None = Field(default=None, description="MLflow run that produced it")
    created_at: str | None = Field(default=None, description="ISO-8601, UTC")


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_server: str
    model_version: str


class StageRun(BaseModel):
    run_id: str
    stage: str
    status: str
    started_at: str
    duration_s: float
    items: int
    items_per_s: float
    error: str | None = None
    git_commit: str | None = None


class PipelineSummary(BaseModel):
    stages: list[StageRun] = Field(description="Most recent run of each stage")
    gate: dict | None = Field(default=None, description="Latest promotion decision")
    data_quality: dict | None = None
    throughput: dict | None = None
    metrics: dict | None = None
