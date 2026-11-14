# LeafGuard: Low-Level Design

This document specifies every interface between the blocks in
[architecture.md](architecture.md): the backend REST API that the UI uses, the model-server
API that the backend uses, the MLflow calls, the batch-stage contracts and the metrics.
The request and response models live in `services/backend/app/schemas.py`. FastAPI
serves the same contract live at **http://localhost:8000/docs** (Swagger) and
`/openapi.json`, so this document and the running code cannot drift apart.

## 1. Backend REST API (FastAPI, port 8000)

Base path for application endpoints: **`/api/v1`**. The browser reaches it through nginx
at `http://<host>:8080/api/v1/...`.

Every response carries an **`X-Request-ID`** header. Every error, from every endpoint,
has the same shape:

```json
{ "error": { "code": "INVALID_IMAGE",
             "message": "That file could not be read as an image. It may be damaged.",
             "request_id": "9d4ea4106854" } }
```

### 1.1 Endpoint summary

| Method | Path | Purpose | Request | Success | Errors |
|---|---|---|---|---|---|
| GET | `/health` | Liveness probe (used by Docker) | none | 200 `HealthResponse` | none |
| GET | `/ready` | Readiness: is the model server reachable? | none | 200 `ReadyResponse` | 503 |
| GET | `/metrics` | Prometheus exposition | none | 200 `text/plain` | none |
| POST | `/api/v1/predict` | Diagnose one leaf photo | `multipart/form-data`, field `file` | 200 `PredictResponse` | 400, 413, 415, 422, 502, 503 |
| POST | `/api/v1/feedback` | Say whether a prediction was right | JSON `FeedbackRequest` | 200 `FeedbackResponse` | 422 |
| GET | `/api/v1/classes` | Classes with plain-language text | none | 200 `ClassInfo[]` | none |
| GET | `/api/v1/model/info` | Which model version is serving | none | 200 `ModelInfo` | none |
| GET | `/api/v1/models` | Every registered version, aliases and gate result | none | 200 `ModelVersion[]` | none |
| GET | `/api/v1/pipeline/runs` | Stage run history, newest first | query `limit:int=50`, `stage:str?` | 200 `StageRun[]` | 422 |
| GET | `/api/v1/pipeline/summary` | Newest run per stage plus reports | none | 200 `PipelineSummary` | none |
| GET | `/api/v1/pipeline/dag` | Pipeline graph as Mermaid text | none | 200 `text/plain` | none |

### 1.2 `POST /api/v1/predict`

**Input.** `multipart/form-data` with one part, `file`: a JPEG, PNG or WEBP image of at
most `MAX_UPLOAD_MB` (5) MB, at least `MIN_SIDE_PX` (64) px on its shorter side.

**Processing.**
1. Validate the upload: present → size → content type → fully decodable → minimum size.
2. Compute image statistics and update the drift window (PSI per feature).
3. Call the model server's `/invocations` (§2), timed as `leafguard_model_latency_seconds`.
4. Rank the scores. `status = "uncertain"` if the top score is below
   `CONFIDENCE_THRESHOLD` (0.60) or the green ratio is below `MIN_GREEN_RATIO` (0.05).
   Otherwise `status = "ok"`.

**Output: `PredictResponse`**

| Field | Type | Description |
|---|---|---|
| `request_id` | string | 12-hex id; pass it to `/feedback` |
| `status` | `"ok"` \| `"uncertain"` | Confident answer, or "please try a better photo" |
| `predicted_class` | string | One of the six class names |
| `display_name` | string | Human-readable class name |
| `confidence` | float 0–1 | Softmax score of the top class (4 decimals) |
| `top_k` | `ClassScore[3]` | `{name, display_name, confidence}`, best first |
| `advice` | string | What to do; empty when uncertain |
| `hint` | string \| null | Photo tips; set only when uncertain |
| `model_version` | string | Registry version behind `@champion` |
| `inference_ms` | float | Round-trip time to the model server |
| `disclaimer` | string | Fixed educational-use notice |

Example:

```bash
curl -F "file=@leaf.jpg;type=image/jpeg" http://localhost:8080/api/v1/predict
```
```json
{"request_id":"9d4ea4106854","status":"ok","predicted_class":"septoria",
 "display_name":"Septoria leaf spot","confidence":0.9722,
 "top_k":[{"name":"septoria","display_name":"Septoria leaf spot","confidence":0.9722},
          {"name":"leaf_mold","display_name":"Leaf mould","confidence":0.0214},
          {"name":"healthy","display_name":"Healthy","confidence":0.0036}],
 "advice":"Remove affected leaves, mulch to stop soil splash, and rotate crops.",
 "hint":null,"model_version":"1","inference_ms":86.49,"disclaimer":"This is an automated ..."}
```

**Error codes**

| HTTP | `code` | When |
|---|---|---|
| 400 | `NO_FILE` | The file part is empty |
| 400 | `INVALID_IMAGE` | Bytes cannot be decoded as an image |
| 400 | `IMAGE_TOO_SMALL` | Shorter side below 64 px |
| 413 | `FILE_TOO_LARGE` | Larger than 5 MB |
| 415 | `UNSUPPORTED_TYPE` | Content type is not `image/*` |
| 422 | `INVALID_REQUEST` | The `file` field is missing (malformed request) |
| 502 | `MODEL_ERROR` | Model server answered non-200 or returned nothing |
| 503 | `MODEL_UNAVAILABLE` | Model server unreachable or timed out (20 s) |
| 500 | `INTERNAL_ERROR` | Anything unexpected; details only in the logs |

### 1.3 `POST /api/v1/feedback`

**Input: `FeedbackRequest`** (JSON)

| Field | Type | Required | Description |
|---|---|---|---|
| `request_id` | string | yes | From the prediction being judged |
| `correct` | bool | yes | Was the prediction right? |
| `correct_class` | string | when `correct` is false | One of the six class names |
| `comment` | string | no | Free text |

**Output: `FeedbackResponse`** `{ "stored": true, "message": "Thank you. ..." }`

**Errors.** 422 `MISSING_CLASS` (wrong, but no class given), 422 `UNKNOWN_CLASS`,
422 `INVALID_REQUEST` (malformed JSON).

**Side effects.** Increments `leafguard_feedback_total{agreed}`. For corrections, appends
a line to `data/feedback/feedback.jsonl` and, if the image is still in the 20-item
upload cache and `FERNET_KEY` is set, writes `data/feedback/<label>/<request_id>.bin`
(encrypted).

### 1.4 Reference and pipeline endpoints

**`GET /api/v1/classes`** returns `ClassInfo[]`: `{name, display_name, description, advice}`
for all six classes.

**`GET /api/v1/model/info`** returns `ModelInfo`:

| Field | Type | Example |
|---|---|---|
| `name` | string | `leafguard` |
| `version` | string | `3` (`unknown` if MLflow is unreachable) |
| `alias` | string | `champion` |
| `accuracy` | string \| null | `0.72152` (registry tag) |
| `git_commit` | string \| null | `1a34ce46` |
| `classes` | string[] | the six class names |
| `confidence_threshold` | float | `0.6` |

**`GET /api/v1/models`** returns `ModelVersion[]`, newest first (empty list when MLflow is
unreachable). Read-only by design: promotion only happens through the pipeline's gate and
rollback through `make rollback`, so the browser can never change the live model.

| Field | Type | Example |
|---|---|---|
| `version` | string | `4` |
| `aliases` | string[] | `["champion"]` (empty if none) |
| `accuracy` | string \| null | `0.72152` (registry tag set by the gate) |
| `gate` | string \| null | `passed` / `failed` |
| `git_commit` | string \| null | `25f8215a` |
| `run_id` | string \| null | MLflow run that produced the version |
| `created_at` | string \| null | ISO-8601 UTC registration time |

**`GET /api/v1/pipeline/runs?limit=50&stage=train`** returns `StageRun[]`, newest first:

| Field | Type | Description |
|---|---|---|
| `run_id` | string | 12-hex id of this stage execution |
| `stage` | string | `lint` … `gate` |
| `status` | `success` \| `failed` | |
| `started_at` | ISO-8601 string | UTC |
| `duration_s` | float | wall time |
| `items` | int | images or tests handled |
| `items_per_s` | float | throughput |
| `error` | string \| null | `ExceptionType: message` when failed |
| `git_commit` | string \| null | short hash |

**`GET /api/v1/pipeline/summary`** returns `PipelineSummary`
`{stages: StageRun[], gate, data_quality, throughput, metrics}`. The last four are the JSON
reports the pipeline writes (`reports/gate.json`, `data_quality.json`, `throughput.json`,
`metrics.json`), or `null` before the first run.

**`GET /api/v1/pipeline/dag`** returns Mermaid `flowchart LR` text, rendered by the console.

**`GET /ready`** returns `ReadyResponse` `{ready: true, model_server: "up", model_version: "3"}`,
or 503 `MODEL_UNAVAILABLE`.

## 2. Model server API (MLflow pyfunc, port 5001)

Started by `services/model_server/start.sh` as
`mlflow models serve -m models:/leafguard@champion --env-manager local --workers 1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/ping` | Health |
| GET | `/version` | MLflow version |
| POST | `/invocations` | Score images |

**`POST /invocations`** input (`Content-Type: application/json`), using MLflow's
`dataframe_split` format:

```json
{ "dataframe_split": { "columns": ["image_b64"], "data": [["<base64 JPEG/PNG bytes>"]] } }
```

Output: one row per input image, one column per class, softmax scores:

```json
{ "predictions": [ { "bacterial_spot": 0.0012, "early_blight": 0.0008, "healthy": 0.0036,
                     "late_blight": 0.0008, "leaf_mold": 0.0214, "septoria": 0.9722 } ] }
```

Model signature (logged with the model): input `image_b64: string`, output one `double`
column per class.

**Pyfunc internals** (`src/leafguard/training/wrapper.py`, `LeafModel`):
`load_context` reads `classes.json` (class order and image size) and loads the
TorchScript weights. `predict` base64-decodes the image, converts to RGB, centre-crops
and resizes to 128×128 (bilinear), scales to 0–1, normalises with the ImageNet mean and
std, converts HWC to CHW, runs the network and applies softmax.

## 3. MLflow tracking and registry interface

| Who | Call | Data |
|---|---|---|
| `train` | `mlflow.autolog(log_models=False)`, then `start_run(log_system_metrics=True)` in experiment `leafguard` | **automatic**: `system/cpu_utilization_percentage`, `system/system_memory_usage_megabytes`, disk and network metrics sampled during the run. **dataset input** (context `training`): `leafguard-splits`, the split table with its digest. **params**: backbone, pretrained, freeze_backbone, epochs, batch_size, learning_rate, image_size, seed, class_count, classes, train_images, val_images, num_threads. **tags**: git_commit, git_dirty, data_version, stage, host. **metrics per epoch**: train_loss, train_accuracy, val_loss, val_accuracy, epoch_seconds, train_images_per_second. **final metrics**: best_val_accuracy, model_size_mb, parameter_count. **artifacts**: `metadata/classes.json`, `params.yaml`, `baseline.json`, `data_quality.json`, `split_counts.json`, `run.json` |
| `evaluate` | resumes the same run | **metrics**: test_accuracy, test_macro_f1, latency_p50_ms, latency_p95_ms, `f1_<class>` ×6. **artifacts**: `evaluation/confusion_matrix.png`, `metrics.json` |
| `gate` | `pyfunc.log_model(name="model", python_model=<wrapper.py>, artifacts={weights, classes}, registered_model_name="leafguard")` | new version. **version tags**: accuracy, macro_f1, git_commit, data_version, gate=passed\|failed. **aliases**: `challenger` always moves to the new version; `champion` moves only if the gate passes |
| `registry rollback --to N` | `set_registered_model_alias(champion, N)` | then `docker compose restart model-server` |
| backend | `GET /api/2.0/mlflow/registered-models/alias?name=leafguard&alias=champion` | version and tags for `/model/info` (cached for 30 s) |
| backend | `GET /api/2.0/mlflow/registered-models/get` + `GET /api/2.0/mlflow/model-versions/search` | version list and aliases for `/api/v1/models` (Pipeline console) |

**Gate rules** (`params.yaml` → `gate`): `accuracy ≥ min_accuracy (0.70)`;
`champion_accuracy − accuracy ≤ max_drop_vs_champion (0.02)`;
`latency_p95_ms ≤ max_latency_ms (400)`.

## 4. Batch stage contracts

Every stage is a module with `run(params) -> StageResult(items, extra)`, decorated with
`@track_stage(name)`, and runnable as `python -m <module>`. Expected failures raise a
`LeafGuardError` subclass (`ConfigError`, `DataError`, `ImageError`, `StageError`),
which becomes exit code 1 and a `failed` record carrying the error text.

| Stage | Module | Reads | Writes | Fails when |
|---|---|---|---|---|
| validate | `leafguard.data.validate` | `data/raw/<train\|val>/<class>/*` | `manifest.parquet` (one row per image: path, label, source split, md5, phash, width, height, format, blur, colour stats, flags, `reject_reason`, `valid`), `data_quality.json` | raw folder missing; any split/class has fewer than `min_images_per_class` valid images |
| split | `leafguard.data.split` | manifest | `splits.parquet` (`path, file_name, label, split, source_split, file_md5, phash`), `split_summary.json` | `val_fraction` outside [0,1); a requested class has no valid images |
| preprocess | `leafguard.data.preprocess` | splits | `data/processed/<split>/<class>/*.jpg` (128×128 RGB), `index.csv`, `models/baseline.json` (quantile bins per feature), `throughput.json` | any image fails to process |
| train | `leafguard.training.train` | `data/processed/{train,val}` | `models/model.pt`, `classes.json`, `run.json` | train and val class sets differ |
| evaluate | `leafguard.training.evaluate` | `model.pt`, `data/processed/test` | `metrics.json`, `confusion_matrix.png` | `run.json` missing; class mismatch |
| gate | `leafguard.training.registry promote` | `run.json`, `metrics.json` | `gate.json` | earlier stages missing |

Run record (`reports/pipeline_runs.jsonl`, one JSON object per line):
`{run_id, stage, status, started_at, duration_s, items, items_per_s, error, git_commit, extra}`.

## 5. Metrics interface (Prometheus)

| Metric | Type | Labels | Source |
|---|---|---|---|
| `leafguard_requests_total` | counter | endpoint, method, status | backend |
| `leafguard_request_duration_seconds` | histogram | endpoint | backend |
| `leafguard_model_latency_seconds` | histogram | none | backend |
| `leafguard_predictions_total` | counter | predicted_class, status | backend |
| `leafguard_prediction_confidence` | histogram | none | backend |
| `leafguard_rejected_total` | counter | reason | backend |
| `leafguard_model_errors_total` | counter | kind | backend |
| `leafguard_feedback_total` | counter | agreed | backend |
| `leafguard_drift_score` | gauge | feature | backend |
| `leafguard_model_info` | gauge (=1) | version, alias | backend |
| `leafguard_stage_{duration_seconds,items,throughput,last_status,last_run_timestamp,last_success_timestamp}` | gauge | stage | pipeline via Pushgateway |
| `leafguard_champion_accuracy` | gauge | version | gate via Pushgateway |
| MLflow, node, container metrics | various | none | mlflow `/metrics`, node-exporter, cAdvisor |

Alert rules are in `monitoring/prometheus/alerts.yml`: `LeafGuardTargetDown`,
`LeafGuardHighErrorRate`, `LeafGuardSlowPredictions`, `LeafGuardInputDrift`,
`LeafGuardManyUncertainAnswers`, `LeafGuardPipelineStageFailed`.

## 6. Configuration interface

| Variable (backend) | Default | Meaning |
|---|---|---|
| `MODEL_SERVER_URL` | `http://model-server:5001` | Where inference happens |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Registry for the version lookup |
| `MODEL_NAME` | `leafguard` | Registered model name |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, the answer is uncertain |
| `MIN_GREEN_RATIO` | `0.05` | Below this, the photo is "probably not a leaf" |
| `MIN_SIDE_PX` / `MAX_UPLOAD_MB` | `64` / `5` | Upload limits |
| `REQUEST_TIMEOUT_S` | `20` | Model-server timeout |
| `DRIFT_WINDOW` | `50` | Uploads in the PSI window |
| `REGISTRY_TTL_S` | `30` | Cache time for the serving version |
| `FERNET_KEY` | (empty) | Key that encrypts feedback images |
| `REPORTS_DIR`, `FEEDBACK_DIR`, `BASELINE_PATH` | `/app/...` | Mounted volumes |

| Variable (frontend) | Default | Meaning |
|---|---|---|
| `BACKEND_HOST` / `BACKEND_PORT` | `backend` / `8000` | nginx proxy target |
| `API_BASE` | `/api/v1` | Written to `config.js` as `window.LEAFGUARD_API` |
| `MLFLOW_PORT`, `GRAFANA_PORT`, `PROMETHEUS_PORT`, `BACKEND_PUBLIC_PORT` | 5000, 3000, 9090, 8000 | Tool links on the Pipeline console |

Pipeline settings are all in `params.yaml` (data, spark, train, gate, serving).

## 7. Module map

```
src/leafguard/common.py      errors, params, JSON logging, git/DVC info, @track_stage, Pushgateway
src/leafguard/images.py      decode, hashes, blur, colour stats, resize   (pure, no Spark/Torch)
src/leafguard/drift.py       build_reference(), psi()                     (pure)
src/leafguard/spark_session.py  make_spark() from params.spark
src/leafguard/data/          validate.py, split.py, preprocess.py
src/leafguard/training/      train.py, evaluate.py, registry.py (gate, rollback, list), wrapper.py
services/backend/app/        main.py (routes, middleware, error handlers), services.py
                             (ModelClient, RegistryReader, DriftWatcher, FeedbackStore,
                             check_upload), schemas.py, metrics.py, config.py
services/frontend/public/    index.html, pipeline.html, help.html, app.js, styles.css
```
