export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help setup lint fix test test-spark test-e2e ci dag metrics up down logs ps \
        grafana traffic drift experiment models rollback clean

help:            ## list the available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

setup:           ## install the pinned requirements (incl. the package, editable) and the git hooks
	pip install -r requirements.txt
	pre-commit install

lint:            ## ruff and black checks (the first CI stage)
	python scripts/run_lint.py

fix:             ## apply the formatting and lint fixes automatically
	ruff check --fix src services tests scripts
	black src services tests scripts

test:            ## unit tests with coverage (the second CI stage)
	python scripts/run_tests.py

test-spark:      ## the Spark integration test (slower)
	LEAFGUARD_NO_PUSH=1 pytest tests/integration -q -m spark --junitxml=reports/integration_junit.xml
	@python scripts/scrub_reports.py reports/integration_junit.xml

test-e2e:        ## end-to-end tests against the running stack (after make up)
	pytest tests/e2e -q -m e2e --junitxml=reports/e2e_junit.xml
	@python scripts/scrub_reports.py reports/e2e_junit.xml

ci:              ## the whole pipeline: lint, tests, data, training, gate
	dvc repro

dag:             ## show the pipeline graph and save a copy for the docs
	dvc dag
	dvc dag --md > docs/pipeline_dag.md

metrics:         ## data quality, split counts, throughput and model metrics
	dvc metrics show

GRAFANA_CONTAINER ?= grafana

up:              ## build and start every container, then let your Grafana reach Prometheus
	docker compose up -d --build
	@$(MAKE) --no-print-directory grafana

down:            ## stop every container
	docker compose down

logs:            ## follow the backend and model server logs
	docker compose logs -f backend model-server

ps:              ## container status
	docker compose ps

grafana:         ## attach your Grafana container to the stack network (Prometheus = http://prometheus:9090)
	@docker network connect leafguard_default $(GRAFANA_CONTAINER) 2>/dev/null \
	  && echo "grafana: attached '$(GRAFANA_CONTAINER)' to leafguard_default" \
	  || echo "grafana: '$(GRAFANA_CONTAINER)' already attached (or not running)"

traffic:         ## send normal requests so the dashboards fill up
	python scripts/demo_traffic.py --count 60

drift:           ## send darkened, blurred images to make the drift panel rise
	python scripts/demo_traffic.py --count 60 --drift

EPOCHS ?= 3
LR ?= 0.001
MIN_ACC ?= 0.70

experiment:      ## set train/gate params and re-run the pipeline, e.g. make experiment EPOCHS=6 LR=0.001 MIN_ACC=0.70
	sed -i -E 's/^(  epochs: ).*/\1$(EPOCHS)/; s/^(  lr: ).*/\1$(LR)/; s/^(  min_accuracy: )[0-9.]+/\1$(MIN_ACC)/' params.yaml
	@grep -nE '^  (epochs|lr|min_accuracy): ' params.yaml
	dvc repro

models:          ## list registered model versions and their aliases
	python -m leafguard.training.registry list

rollback:        ## move champion to an earlier version, e.g. make rollback TO=1
	@echo "$(TO)" | grep -qE '^[0-9]+$$' || { echo "TO must be a version number from 'make models', e.g. make rollback TO=1"; exit 1; }
	python -m leafguard.training.registry rollback --to $(TO)
	dvc commit -f -q gate    # rollback rewrites reports/gate.json; record it so dvc repro stays a no-op
	docker compose restart model-server

clean:           ## remove caches (data, models and reports are kept)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
