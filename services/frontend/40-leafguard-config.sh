#!/bin/sh
# Runs inside the nginx image at startup (from /docker-entrypoint.d).
# Writes config.js from environment variables, so the API base path and the ports of the
# MLOps tools linked from the Pipeline console are deployment settings, not code.
set -e
cat > /usr/share/nginx/html/config.js <<CONFIG
window.LEAFGUARD_API = "${API_BASE:-/api/v1}";
window.LEAFGUARD_TOOLS = {
  mlflow: "${MLFLOW_PORT:-5000}",
  grafana: "${GRAFANA_PORT:-3000}",
  prometheus: "${PROMETHEUS_PORT:-9090}",
  api_docs: "${BACKEND_PUBLIC_PORT:-8000}"
};
CONFIG
echo "leafguard: wrote config.js (api=${API_BASE:-/api/v1})"
