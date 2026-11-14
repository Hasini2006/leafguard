# LeafGuard
##                K.Sree Hasini | DA24B041
**An end-to-end AIOps platform for image-based tomato leaf disease diagnosis.**

A farmer uploads a photo of a tomato leaf; LeafGuard answers with one coloured label:
green **Healthy**, red **Disease found** (with the disease name and advice), or yellow
**Not sure** when the photo is unclear. The model (MobileNetV3-Small, 6 classes) is small
on purpose. The project is about everything around it: versioned data, a CI + ML pipeline,
tracked experiments, a gated model registry, REST serving, monitoring with alerts, tests
and rollback, all running locally with Docker Compose.

---

## Basic Overview

| To check | Look at | Command or address |
|---|---|---|
| How to run it from scratch | [SETUP.md](SETUP.md), steps 0–11 | |
| Mandatory documents | [docs/README.md](docs/README.md): architecture, HLD, LLD, test plan + report, user manual, and a rubric-to-implementation table | |
| Architecture | [docs/architecture.md](docs/architecture.md): text and Mermaid diagrams, one table per block | |
| The user interface | Farmer screen and help page; a separate operator console | http://localhost:8080, http://localhost:8080/pipeline.html |
| CI pipeline (DAG) | `dvc.yaml`: `lint` and `unit_tests` are the first stages | `dvc dag \| cat` |
| Data engineering (Spark) | `src/leafguard/data/validate.py`, `preprocess.py` | `cat reports/data_quality.json reports/throughput.json` |
| Reproducibility | `dvc.lock` + the git commit and MLflow run id on every run | `dvc repro` (skips everything when nothing changed) |
| Experiment tracking | MLflow: parameters, per-epoch metrics, system metrics, dataset lineage, artifacts | http://localhost:5000 |
| Model registry and gate | `src/leafguard/training/registry.py`; `champion` / `challenger` aliases | `make models`, `cat reports/gate.json` |
| Model serving (APIfication) | `mlflow models serve` on `models:/leafguard@champion` | http://localhost:5001/ping |
| Identical dev/test environments | `MLproject` + `python_env.yaml` → pinned `requirements.txt` | `mlflow run . --env-manager local` |
| REST API (FastAPI) | `services/backend/`; OpenAPI docs | http://localhost:8000/docs |
| Monitoring and alerting | Prometheus rules (drift, errors, latency, targets, stage failures); Grafana dashboard | http://localhost:9090/alerts, http://localhost:3000/d/leafguard |
| Tests | [docs/test_plan.md](docs/test_plan.md) §6: test report | `make test`, `make test-spark`, `make test-e2e` |
| Rollback | Moves the `champion` alias and restarts the model server | `make rollback TO=<version>` |

## Results of the last run

| | |
|---|---|
| Automated tests | **66 / 66 passed**: 55 unit, 1 Spark integration, 10 end-to-end; lint clean |
| Dataset | 1,440 images, 6 classes; 1,436 valid after the Spark data-quality pass |
| Spark preprocessing | about 390 images/s on `local[4]` |
| Model versions | 4 training runs, 4 registered versions: v2 refused by the gate, v4 champion |
| Champion (v4) | test accuracy **0.776**, macro F1 0.755, p95 latency 1.8 ms, 6.5 MB |
| Monitoring | 6/6 Prometheus targets up; drift alert fires on degraded input and clears on normal input |

Full details, including the defects found and fixed, are in the
[test report](docs/test_plan.md).

---

## Architecture

```
  Browser --HTTP--> frontend (nginx :8080) --REST /api/v1--> backend (FastAPI :8000)
                                                               |            |
                                                REST /invocations      alias lookup
                                                               v            v
                                      model-server (:5001) <--loads @champion-- MLflow (:5000)
                                      mlflow models serve                   tracking + registry
                                                                                  ^
  dvc repro:  lint -> unit_tests -> validate (Spark) -> split -> preprocess (Spark)
              -> train (PyTorch) -> evaluate -> gate (register, promote) ---------+

  Monitoring: backend, MLflow, Pushgateway (pipeline stages), node-exporter, cAdvisor
              -> Prometheus :9090 (alert rules) -> Grafana :3000 (dashboard)
```

The full diagram and an explanation of every block are in
[docs/architecture.md](docs/architecture.md). Three design decisions:

- **The backend never loads the model.** It validates uploads, turns scores into a
  user-facing answer, measures drift and stores feedback. Inference happens in the
  model-server container, reached over REST at `MODEL_SERVER_URL`. Each tier can be
  restarted or replaced on its own.
- **Preprocessing travels inside the model.** The MLflow pyfunc wrapper resizes and
  normalises, so every model version carries the exact transform it was trained with.
- **Aliases, not version numbers.** The model server loads `@champion`. Promotion and
  rollback change the alias; no code or configuration changes.

---

## Technology

| Concern | Tool | Where |
|---|---|---|
| Code versioning | Git (small text files only; no binaries in git) | `.gitignore` |
| Data and model versioning | DVC with a local remote | `data/raw.dvc`, `dvc.lock`, `.dvc/config` |
| CI | DVC pipeline (`lint`, `unit_tests` stages) + pre-commit | `dvc.yaml`, `scripts/run_*.py` |
| Data engineering | Apache Spark (PySpark, `local[4]`) | `src/leafguard/data/` |
| Training | PyTorch, torchvision (MobileNetV3-Small, transfer learning) | `src/leafguard/training/train.py` |
| Experiment tracking | MLflow tracking server (autolog + system metrics + dataset inputs) | `services/mlflow/` |
| Model registry | MLflow aliases `champion` / `challenger`, gate in the pipeline | `src/leafguard/training/registry.py` |
| Model serving | `mlflow models serve` | `services/model_server/` |
| Reproducible environments | MLproject | `MLproject`, `python_env.yaml`, `requirements.txt` |
| API | FastAPI | `services/backend/` |
| Frontend | HTML, CSS, JavaScript on nginx | `services/frontend/` |
| Instrumentation | prometheus-client, Pushgateway | `services/backend/app/metrics.py` |
| Monitoring and alerting | Prometheus, Grafana, node-exporter, cAdvisor | `monitoring/` |
| Packaging | Docker, Docker Compose | `docker-compose.yml`, `services/*/Dockerfile` |
| Testing | pytest, pytest-cov, pytest-html | `tests/` |
| Security | Fernet-encrypted feedback images; secrets only in the git-ignored `.env` | `services/backend/app/services.py` |

---

## Repository layout

```
README.md                   this file
SETUP.md                    step-by-step setup and reproduction guide
requirements.txt            pinned Python dependencies
params.yaml                 every tunable value (data, Spark, training, gate, serving)
dvc.yaml / dvc.lock         the CI + ML pipeline and the exact inputs of the last run
MLproject / python_env.yaml MLflow Project entry points on the pinned environment
docker-compose.yml          the service stack (bundled Grafana: profile bundled-grafana)
Makefile                    short commands for everything below
data/raw.dvc                the dataset version (the images live in DVC)
docs/                       architecture, HLD, LLD, test plan + report, user manual
src/leafguard/
  common.py                 config, logging, errors, git info, stage tracking
  images.py                 decode, blur, colour, resize
  drift.py                  reference distributions and PSI
  data/validate.py          Spark data-quality pass
  data/split.py             stratified, leakage-free split
  data/preprocess.py        Spark resize plus the drift baseline
  training/train.py         transfer learning, logged to MLflow
  training/evaluate.py      golden-set metrics and the confusion matrix
  training/wrapper.py       the pyfunc model the model server runs
  training/registry.py      register, gate, promote, rollback
services/backend/           FastAPI gateway
services/frontend/          nginx: farmer pages and the operator console
services/model_server/      mlflow models serve
services/mlflow/            tracking server and registry
monitoring/                 Prometheus config and alert rules, Grafana provisioning and dashboard
scripts/                    CI steps, report scrubbing, dataset subset, demo traffic
tests/                      unit, integration (Spark), e2e (against the running stack)
reports/                    metrics and test reports of the last run
```

---

## Quick start

The complete guide, from a fresh machine to a running, monitored stack, is
**[SETUP.md](SETUP.md)**. In short:

```bash
python3 -m venv ~/aiops && source ~/aiops/bin/activate
pip install -r requirements.txt
dvc pull                                        # the dataset (or rebuild it: SETUP.md step 3)
cp .env.example .env                            # then set FERNET_KEY (SETUP.md step 4)
docker compose up -d --build mlflow pushgateway prometheus
export MLFLOW_TRACKING_URI=http://localhost:5000
dvc repro                                       # lint -> tests -> Spark -> train -> gate
make up                                         # frontend, backend, model server, exporters
docker compose --profile bundled-grafana up -d grafana   # Grafana with the dashboard
make test-e2e
```

| Open | Address |
|---|---|
| LeafGuard (farmer screen) | http://localhost:8080 |
| Pipeline console (operators) | http://localhost:8080/pipeline.html |
| API documentation | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana (`admin` / `leafguard`) | http://localhost:3000/d/leafguard |
| Prometheus alerts / targets | http://localhost:9090/alerts, http://localhost:9090/targets |

---

## Commands

```bash
make help                    # every command
make ci                      # dvc repro: the whole pipeline
make lint / make fix         # style checks / apply fixes
make test                    # unit tests, coverage, HTML report
make test-spark              # Spark integration test
make test-e2e                # end-to-end tests against the running stack
make metrics                 # data quality, throughput and model metrics
make up / make down / make ps
make traffic                 # normal requests, so the dashboards fill
make drift                   # darkened, blurred photos: the drift alert fires
make experiment EPOCHS=6 LR=0.001 MIN_ACC=0.70   # a new training run through the gate
make models                  # registered versions, aliases, gate results
make rollback TO=1           # serve an earlier version
```

## The lifecycle in five steps

1. **Reproducibility.** Run `dvc repro` twice: the second run skips every stage.
   `dvc.lock` pins the data, code and parameters; each MLflow run records the git commit.
2. **Experiments.** `make experiment EPOCHS=... LR=...` re-runs only train → evaluate →
   gate. Each run is a new MLflow run and a new registered version.
3. **A bad candidate is refused.** `make experiment EPOCHS=1 MIN_ACC=0.95`: the gate refuses
   it, the champion stays, and the reasons appear in `reports/gate.json` and the console.
4. **Rollback.** `make rollback TO=1` moves the champion alias back and restarts the model
   server; the version in the UI footer and in Grafana follows.
5. **Drift.** `make drift` sends degraded photos: the PSI panel climbs and
   `LeafGuardInputDrift` fires. `make traffic` brings it back to normal.
