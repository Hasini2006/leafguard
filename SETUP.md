# LeafGuard: Setup and Reproduction Guide

My machine is  Ubuntu (aarch64), 4 CPUs, 8 GB RAM, no GPU. A full first pipeline run takes a
few minutes.

---

## Step 0: Prerequisites (one time)

```bash
python3 --version            # 3.11 or newer (3.14 used)
java -version                # 17 (Apache Spark needs it)
docker --version             # Docker Engine
docker compose version       # Compose plugin v2+
git --version
```

If one is missing:

```bash
sudo apt update
sudo apt install -y python3-venv openjdk-17-jdk git
# Docker Engine + Compose plugin: https://docs.docker.com/engine/install/ubuntu/
sudo usermod -aG docker $USER   # then log out and back in
```

Ports used: 8080, 8000, 5001, 5000, 9090, 9091 and 3000 (Grafana). They must be free.

## Step 1: Get the code

```bash
cd ~
git clone https://github.com/Hasini2006/leafguard.git leafguard
cd ~/leafguard
```

## Step 2: Python environment

```bash
python3 -m venv ~/aiops
source ~/aiops/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # pinned: Spark, PyTorch (CPU), MLflow, DVC, test tools
python -c "import mlflow; print(mlflow.__version__)"   # 3.17.2, same as the MLflow containers
```

## Step 3: Get the dataset (versioned with DVC)

`data/raw` holds 1,440 tomato-leaf images in 6 classes. Git stores only the pointer
`data/raw.dvc`, which pins the exact content by hash. The DVC remote is a local folder
next to the project, `~/dvc-storage/leafguard` (see `.dvc/config`).

**Option A: you were given a copy of the DVC storage folder.** Put it at
`~/dvc-storage/leafguard`, then:

```bash
dvc pull                      # restores data/raw from the remote
```

**Option B: rebuild it from the public dataset.** The subset script is deterministic
(sorted file lists, fixed seed, byte-for-byte copies), so it reproduces the exact pinned
version.

```bash
# Download "tomatoleaf" from Kaggle (kaustubhb999/tomatoleaf; also Zenodo record 8311631).
# With the Kaggle CLI and an API token in ~/.kaggle/kaggle.json:
pip install kaggle
kaggle datasets download -d kaustubhb999/tomatoleaf -p data/download --unzip
find data/download -maxdepth 3 -type d -name train      # the folder that holds train/ and val/
python scripts/make_subset.py --src <that-folder>       # writes data/raw (6 classes, capped)
```

Either way, check the data matches the pinned version and save it to the DVC remote:

```bash
dvc status data/raw.dvc       # "Data and pipelines are up to date." = identical to the pin
mkdir -p ~/dvc-storage/leafguard
dvc push                      # stores the dataset in the local DVC remote
```

## Step 4: Configuration (`.env`)

```bash
cp .env.example .env
sed -i "s|^FERNET_KEY=.*|FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')|" .env
grep -c '^FERNET_KEY=replace-me' .env   # prints 0: the key was set
```

`.env` keeps run-time configuration out of the code: published ports, the confidence
threshold, the upload limit, and the secret `FERNET_KEY` that encrypts feedback images.
Every value except the key has a default in `docker-compose.yml`. `.env` is git-ignored.

## Step 5: Start the tracking and metrics services

The pipeline logs to MLflow and pushes stage metrics to the Pushgateway, so these start
first. `down -v` makes sure no old LeafGuard containers or volumes are left.

```bash
docker compose down -v --remove-orphans
docker compose up -d --build mlflow pushgateway prometheus
docker compose ps             # wait until mlflow is "healthy" (about 30 s)
curl -s http://localhost:5000/health    # OK
```

## Step 6: Run the CI + ML pipeline (DVC)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # needed in every new terminal
dvc dag | cat                 # the DAG: lint -> unit_tests -> validate -> split
                              #   -> preprocess -> train -> evaluate -> gate
dvc repro                     # runs all 8 stages; lint and tests first, as the CI gate
```

Check the results:

```bash
cat reports/metrics.json      # test accuracy, macro F1, per-class F1, latency
cat reports/throughput.json   # Spark preprocessing speed (images/s)
make models                   # version 1, alias champion, gate=passed
cat reports/gate.json         # the promotion decision and its reasons
dvc repro                     # second run: every stage is skipped (reproducibility)
dvc push                      # the model and processed data go to the DVC remote too
```

The same pipeline is also an **MLflow Project** (`MLproject`), whose environment
(`python_env.yaml`) installs the pinned `requirements.txt`, so development and testing use
identical packages:

```bash
mlflow run . --env-manager local    # entry point "main" = dvc repro, in this venv
# mlflow run .                      # or let MLflow build a fresh environment first
```

## Step 7: Start the application stack and Grafana

```bash
make up                       # builds and starts frontend, backend, model-server,
                              # node-exporter and cAdvisor
make ps                       # all services "running" / "healthy"
curl -s http://localhost:8000/ready     # after about a minute: {"ready":true,...,"model_version":"1"}
```

**Grafana, option A (recommended): the bundled one.** The Prometheus data source and the
*LeafGuard overview* dashboard are provisioned automatically.

```bash
docker compose --profile bundled-grafana up -d grafana
```

Open http://localhost:3000/d/leafguard and log in as `admin` / `leafguard` (change these
with `GRAFANA_USER` / `GRAFANA_PASSWORD` in `.env` before starting it).

**Grafana, option B: a Grafana you already run in Docker.** `make up` attaches a container
named `grafana` to the stack network (use `make grafana GRAFANA_CONTAINER=<name>` for
another name). Then, in Grafana:

1. *Connections → Data sources → Add data source → Prometheus*. URL
   `http://prometheus:9090`, scrape interval `5s`, then **Save & test**.
2. *Dashboards → New → Import → Upload dashboard JSON file*:
   `monitoring/grafana/dashboards/leafguard.json`.
3. In the dashboard's **Prometheus** drop-down, pick the data source from step 1.

## Step 8: Run the test suites

```bash
make lint                     # ruff + black
make test                     # 55 unit tests with coverage -> reports/tests.html
make test-spark               # Spark integration test -> reports/integration_junit.xml
make test-e2e                 # 10 end-to-end tests against the running stack -> reports/e2e_junit.xml
```

The expected results and acceptance criteria are in [docs/test_plan.md](docs/test_plan.md).

## Step 9: Open the application and the MLOps tools

| What | Address |
|---|---|
| LeafGuard: farmer screen | http://localhost:8080 |
| Pipeline console (operators) | http://localhost:8080/pipeline.html |
| Help page | http://localhost:8080/help.html |
| API documentation (Swagger) | http://localhost:8000/docs |
| MLflow experiments and registry | http://localhost:5000 |
| Grafana dashboard | http://localhost:3000/d/leafguard |
| Prometheus targets / alerts | http://localhost:9090/targets, http://localhost:9090/alerts |

On the farmer screen: **Upload photo** (any image in `data/raw/val/<class>/`), **Check this
leaf**, look at the coloured label (green / red / yellow), then **Yes / No**. Resize the
browser to phone width to see the mobile layout.

## Step 10: The model lifecycle

### 10a: Traffic and drift monitoring

```bash
make traffic                  # 60 normal requests: the Grafana panels fill up
make drift                    # 60 dark, blurred photos: the drift (PSI) panel rises and the
                              # LeafGuardInputDrift alert fires (http://localhost:9090/alerts)
```

### 10b: Experiments, gated promotion

`make experiment` writes `train.epochs`, `train.lr` and `gate.min_accuracy` into
`params.yaml` and runs `dvc repro`. Only **train → evaluate → gate** re-run. Each run is one
MLflow run and one new registered model version, which the gate promotes or refuses.
Each takes 1–2 minutes.

```bash
# Version 2: a deliberately weak model and a 95 % bar -> refused (accuracy floor)
make experiment EPOCHS=1 LR=0.001 MIN_ACC=0.95

# Version 3: a higher learning rate -> refused if it is more than 2 points worse than the
# champion (no-regression rule), promoted otherwise
make experiment EPOCHS=3 LR=0.003 MIN_ACC=0.70

# Version 4: more epochs -> promoted if it is at least as good as the champion
make experiment EPOCHS=10 LR=0.001 MIN_ACC=0.70

make models                   # versions 1-4: accuracy, gate result, which one is champion
cat reports/gate.json         # the reasons for the last decision
docker compose restart model-server     # serve the current champion
dvc status                    # "Data and pipelines are up to date."
```

Exact accuracies can differ slightly between machines, so a borderline version may land on
the other side of the gate; the rules applied are always written in `reports/gate.json`.

In MLflow: *Experiments → leafguard*, tick the runs and press **Compare** (parameters and
metrics side by side). *Models → leafguard* lists every version with its aliases and tags.

### 10c: Rollback

Rollback moves the `champion` alias to an earlier version and restarts the model server.

```bash
make models                                       # note the champion's number, e.g. 4
make rollback TO=1                                # champion -> version 1
curl -s http://localhost:8080/api/v1/model/info   # "version":"1"
make rollback TO=4                                # back to the champion you noted
```

The Pipeline console shows all of it: the latest gate decision with its reasons, every
model version with its alias and gate result, and the filterable run history.

## Step 11: Stop

```bash
make down                     # stops the containers; MLflow and Prometheus data are kept
deactivate
```

To remove everything, including MLflow runs and model versions:
`docker compose --profile bundled-grafana down -v`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dvc repro` stops at `unit_tests` | Run `make test` to see the failing test; the pipeline stops before touching data on purpose |
| `train` cannot reach MLflow | `docker compose ps mlflow` must be healthy, and `MLFLOW_TRACKING_URI` exported in this terminal |
| `/ready` is 503 or the model server restarts | No champion yet: finish step 6, then `docker compose restart model-server` |
| `dvc pull` / `dvc push` cannot find the remote | Create `~/dvc-storage/leafguard` (step 3); the remote path in `.dvc/config` is relative to the project folder |
| `dvc status data/raw.dvc` says *modified* after option B | A different download version was used; `dvc add data/raw` records it, and `dvc repro` re-runs from `validate` |
| Grafana panels say *No data* | Option A: wait 10 s and refresh. Option B: run `make grafana`, check the URL `http://prometheus:9090`, and pick the data source in the **Prometheus** drop-down |
| Port 3000 already in use | You already run a Grafana: use option B, or set `GRAFANA_PORT` in `.env` for the bundled one |
| Spark error about `JAVA_HOME` | `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture)` |
| Out of memory during training | Close other programs, or lower `train.batch_size` in `params.yaml` |
