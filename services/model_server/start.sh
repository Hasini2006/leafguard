#!/bin/sh
# Wait for the tracking server, then serve whichever version holds the champion alias.
# Until a model has been trained and promoted there is nothing to serve, so the loop keeps
# retrying instead of crash-looping the container.
set -e

MODEL_NAME="${MODEL_NAME:-leafguard}"
MODEL_URI="models:/${MODEL_NAME}@champion"

echo "waiting for MLflow at ${MLFLOW_TRACKING_URI}"
until python -c "import urllib.request,sys;urllib.request.urlopen('${MLFLOW_TRACKING_URI}/health')" 2>/dev/null; do
  sleep 3
done

echo "serving ${MODEL_URI}"
while true; do
  mlflow models serve \
    --model-uri "${MODEL_URI}" \
    --host 0.0.0.0 --port 5001 \
    --env-manager local --workers 1 && break
  echo "no promoted model yet (train and run the gate first); retrying in 20s"
  sleep 20
done
