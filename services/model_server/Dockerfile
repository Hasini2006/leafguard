# Serves the registered model as a REST API with `mlflow models serve`.
# This is the only container that holds PyTorch. It loads whichever version carries the
# champion alias, so promoting or rolling back is an alias change plus a restart.
#
# Python and every library version match the training environment (environment parity),
# so the model behaves the same here as it did when the gate evaluated it.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=120 \
    OMP_NUM_THREADS=1

# CPU-only PyTorch: the GPU build would add several gigabytes for no benefit here.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install --no-cache-dir \
      mlflow==3.17.2 numpy==2.5.4 pandas==3.1.0 pillow==13.0.0

WORKDIR /serve
COPY services/model_server/start.sh /serve/start.sh
RUN chmod +x /serve/start.sh

EXPOSE 5001
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=10 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5001/ping')"

CMD ["/serve/start.sh"]
