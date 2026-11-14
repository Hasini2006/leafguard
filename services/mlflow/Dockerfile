# MLflow tracking server and model registry.
# SQLite holds runs and the registry; artifacts are proxied through this server so the
# model-server container can download a model without sharing a filesystem.
# The MLflow version matches the training environment exactly: a 3.x client cannot log
# models to a 2.x server.
FROM python:3.14-slim

# prometheus-flask-exporter (and gunicorn) back the --expose-prometheus /metrics endpoint.
RUN pip install --no-cache-dir mlflow==3.17.2 prometheus-flask-exporter gunicorn

# MLflow 3 starts background job consumers for its GenAI features by default (about 1 GB).
# LeafGuard only uses tracking and the registry, so they are switched off.
ENV MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

WORKDIR /mlflow
RUN mkdir -p /mlflow/artifacts /mlflow/prometheus

EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=3s --retries=10 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5000/health')"

# --allowed-hosts: MLflow 3 rejects Host headers it does not know (DNS-rebinding guard).
# The other containers call it as "mlflow:5000" and the VM as "localhost:5000".
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000", \
     "--expose-prometheus", "/mlflow/prometheus"]
