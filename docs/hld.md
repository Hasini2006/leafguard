# LeafGuard: High-Level Design

## HLD diagram

The system in four layers: data and pipeline (batch), model lifecycle, online serving, and
monitoring. Arrows are data flow; every online arrow is a REST call. Block-level detail is
in [architecture.md](architecture.md).

```mermaid
flowchart LR
  subgraph DATA["Data and CI pipeline (dvc repro)"]
    raw[("data/raw<br/>DVC + local remote")] --> lint[lint + unit tests<br/>CI gate]
    lint --> val[validate<br/>Spark]
    val --> split[split<br/>stratified, leak-free]
    split --> pre[preprocess<br/>Spark + drift baseline]
    pre --> train[train<br/>PyTorch]
    train --> eval[evaluate<br/>golden test set]
    eval --> gate{gate}
  end

  subgraph LIFE["Model lifecycle (MLflow)"]
    track[(tracking<br/>params, metrics,<br/>artifacts, lineage)]
    reg[(registry<br/>@champion / @challenger)]
  end

  subgraph SERVE["Online serving (Docker Compose)"]
    ui[frontend<br/>nginx: farmer UI +<br/>pipeline console] -->|REST /api| api[backend<br/>FastAPI gateway]
    api -->|REST /invocations| ms[model server<br/>mlflow models serve]
  end

  subgraph MON["Monitoring"]
    prom[(Prometheus<br/>+ alert rules)] --> graf[Grafana<br/>dashboard]
    pg[Pushgateway]
  end

  train --> track
  eval --> track
  gate -->|register, promote| reg
  reg -->|loads @champion| ms
  api -->|version list| reg
  user((Farmer)) --> ui
  op((Operator)) --> ui
  op --> graf
  api -->|/metrics| prom
  DATA -->|stage metrics| pg --> prom
```

## 1. Problem and goals

**Problem.** Tomato growers lose crops to leaf diseases that are easy to treat when caught
early but hard for a non-expert to tell apart. LeafGuard lets anyone upload a photo of a
tomato leaf and get a plain-language suggestion in about a second. It names one of five
common diseases or "healthy", gives a confidence score and what to do next, or answers
"uncertain" when the photo is not good enough to judge.

**What the project is really about.** The classifier is deliberately small (a frozen
MobileNetV3-Small head, 3 epochs, 128 px images). The project is assessed on the AIOps
lifecycle around the model, not on accuracy, so the effort goes into versioned data, an
automated and reproducible pipeline, tracked experiments, a gated and reversible
deployment, monitoring with alerts, and tests.

### Success metrics

| Kind | Metric | Target | Measured on the latest run |
|---|---|---|---|
| Business | Time from upload to answer | < 1 s for a typical photo | 86 ms model call; ~0.1 s end to end via nginx |
| Business | Unclear photos never get a confident answer | `uncertain` when the top score is below 0.60 or the photo is not leaf-like | enforced in code, covered by tests |
| ML | Golden-set accuracy of the serving model | ≥ 0.70 (`gate.min_accuracy`) | 0.722 |
| ML | Single-image CPU latency, p95 | ≤ 400 ms (`gate.max_latency_ms`) | 2.1 ms (model only) |
| Ops | Full pipeline wall time | < 5 min on 4 cores | 54 s |
| Ops | Spark data throughput | reported each run | validate 276 img/s, preprocess 522 img/s |
| Ops | Error rate | < 5 % of requests (alert) | 0 % |

## 2. Data

* **Source.** Tomato leaf images from the PlantVillage-derived "tomatoleaf" dataset
  (Kaggle `kaustubhb999/tomatoleaf`, mirrored on Zenodo record 8311631).
  `scripts/make_subset.py` records exactly how the working subset was built: 6 of the 10
  classes, 200 training and 40 validation images per class, seed 42. That is 1,440 JPEG
  images at 256×256.
* **Known biases.** These are laboratory photos of single leaves on plain backgrounds. Some
  of the images are augmented copies (file names containing `_original_`). The model
  will do worse on field photos. The Help page and the README say so.
* **EDA and quality checks (the `validate` stage).** Every file is decoded with Spark, then
  size, format, blur (variance of the Laplacian), colour statistics and a perceptual hash
  are recorded. Files are flagged as corrupt, bad format, too small, too blurry, a label
  conflict or a duplicate. The latest run found 4 exact duplicates out of 1,440 images,
  0 corrupt files, and a class imbalance ratio of 1.005.
* **Leakage control (the `split` stage).** The dataset's own `val` folder becomes the
  untouched *golden test set*. Images with the same perceptual hash always land in the
  same split, and test images that duplicate a training image are dropped. The split is
  seeded per class, so adding a class does not reshuffle the others.

## 3. Key design decisions and why

| # | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| D1 | **Three tiers that talk only over REST**: nginx frontend → FastAPI backend → MLflow model server | Load the model inside FastAPI | The rubric requires loose coupling. It also means the model can be swapped or rolled back by restarting one container, with no rebuild of the API or the UI. All addresses are environment variables. |
| D2 | **Preprocessing lives inside the model** (pyfunc wrapper resizes and normalises) | Do it in the backend | Training and serving then cannot drift apart: every model version carries its own transform. The backend stays free of torch, so its image is small. |
| D3 | **Model from code** (`python_model=<path to wrapper.py>`) | Pickle a `PythonModel` instance | The training environment is Python 3.14. A pickled class would tie serving to that interpreter and to the `leafguard` package. Logging the source avoids both. |
| D4 | **Registry aliases** `champion` / `challenger` | Hard-code version numbers; use stages (deprecated) | Promotion and rollback are an alias move plus a restart, with no code or config change. `make rollback TO=n` is the whole procedure. |
| D5 | **A promotion gate inside the pipeline** (accuracy floor, no regression of more than 0.02 against the champion, p95 latency budget) | Promote whatever trained last | A bad candidate can never go live. The gate writes its reasons to `reports/gate.json`, and the console shows them. |
| D6 | **DVC as both data versioning and CI**: `lint` and `unit_tests` are the first DAG stages | A separate CI server | Cloud services and hosted CI are not allowed by the guidelines. A DVC DAG gives dependency-aware, cached, reproducible runs, and a failing test stops training. |
| D7 | **Apache Spark** for validate and preprocess (`local[4]`) | Plain Python loops; Ray | Required by the rubric (Ray, Spark or K8s). `binaryFile` plus `mapInPandas` parallelises decoding across 4 cores, and the same pure-Python helpers in `images.py` run in the Spark workers and in the backend. |
| D8 | **Drift = PSI of four image statistics** (brightness, saturation, green ratio, blur) against a baseline built in `preprocess` | Embedding drift | Cheap, explainable, and computable in the slim backend. The baseline is versioned with the data by DVC. |
| D9 | **Uncertain state** when the top score is below 0.60 or the green-pixel ratio is below 0.05 | Always return the argmax | This makes the UX foolproof: a photo of a cat or a blurry leaf gets "try a better photo" instead of a confident wrong disease. |
| D10 | **Monitoring with Prometheus, Pushgateway and Grafana**, including alert rules | Logs only | The backend exports RED metrics, prediction mix, confidence, drift and feedback. Batch stages push their metrics, cAdvisor and node-exporter cover infrastructure, and the alert rules turn these into actionable signals. |
| D11 | **Reuse the user's existing Grafana** (attached to the stack network; Prometheus data source and dashboard added in its UI) | Bundle a Grafana container | One Grafana per machine, and no Grafana credentials stored in the project. The dashboard is versioned in git with a data-source variable, so it imports into any Grafana. A bundled Grafana remains available behind the `bundled-grafana` compose profile. |
| D12 | **Structured JSON logging** in every component, plus a request id in each response header | print statements | Logs can be filtered by `request_id` or `stage`, and they rotate at 5 MB. |
| D13 | **Feedback images encrypted with Fernet** | Store plain files | User photos are user data. Without a key, only the label is kept. |

## 4. Programming paradigm

The code mixes the two paradigms on purpose:

* **Functional core.** Image helpers (`images.py`), drift maths (`drift.py`), the split
  logic, the gate rules (`registry.decide`) and report builders are pure functions with
  no I/O. That is why they can run inside Spark workers and are easy to unit-test.
* **Object-oriented shell.** Long-lived collaborators with state are classes:
  `ModelClient`, `RegistryReader`, `DriftWatcher` and `FeedbackStore` in the backend,
  `LeafModel` (an MLflow `PythonModel`), and the `Settings` dataclass. They are created
  once in the FastAPI lifespan and replaced by stubs in tests.
* **Cross-cutting concerns as decorators.** `@track_stage` adds timing, logging, the run
  log and Pushgateway metrics to every pipeline stage without touching the stage's code.

## 5. Reproducibility

Every model version can be traced to the exact code, data and parameters that produced
it:

* the **MLflow run** carries tags `git_commit`, `git_dirty`, `data_version` (the DVC md5
  of `data/raw`), all parameters, per-epoch metrics, and artifacts (`params.yaml`,
  `classes.json`, the drift baseline, the data-quality report, the confusion matrix);
* the **registered version** carries tags `git_commit`, `data_version`, `accuracy`,
  `macro_f1` and `gate`;
* **`dvc.lock`** pins the hash of every input and output of every stage.

Running `dvc repro` again with nothing changed skips every stage. Running it after a
parameter change re-runs only the stages that depend on that parameter. Retraining with
the same seed reproduced the same test accuracy (0.72152) in this run.

## 6. Security

* Uploads are size-limited (5 MB in the UI and the API; nginx caps any request body at
  10 MB) and fully decoded before
  use. Non-images are refused with HTTP 415.
* Stored feedback images are encrypted with Fernet. The key lives in `.env`, which is
  git-ignored.
* Containers run as non-root where the image allows it (the backend runs as user
  `leafguard`). MLflow 3's host-header allow-list is configured explicitly.
* The UI escapes all API text before inserting it into the page.
* Out of scope for a single-machine prototype: authentication, TLS. See the README's
  known limitations.

## 7. Scalability

* The backend and model server are stateless, apart from the in-memory drift window and
  the recent-upload cache, so either can be scaled out behind a load balancer
  (`docker compose up --scale`, or a Swarm or Kubernetes deployment of the same images).
* Spark settings (`master`, `partitions`, `driver_memory`) live in `params.yaml`, so a
  bigger dataset or a cluster master is a configuration change.
* Container memory limits add up to about 3.9 GB, which fits next to Spark and training on
  an 8 GB machine.

## 8. Explainability

* Each answer shows the top three classes with their scores, plus a plain-language
  description and advice for the predicted class.
* The evaluate stage logs per-class F1 and a confusion matrix to MLflow, so you can see
  which diseases the model confuses (early blight is the weakest, with F1 0.48).
* Drift features are human-readable (brightness, saturation, greenness, sharpness), so a
  drift alert says *what* changed about the inputs.

## 9. Retraining loop

1. User corrections are stored (encrypted) in `data/feedback`. Drift and the uncertain
   share are watched in Grafana and raise alerts.
2. When an alert fires or new labelled data arrives, add the data to `data/raw`, then run
   `dvc add data/raw` and `dvc repro`.
3. The gate decides automatically. If the candidate is promoted, `docker compose restart
   model-server` serves it. If it is not, the champion stays.

Retraining is started by a person on purpose. An automatic trigger is listed as future
work in the README.
