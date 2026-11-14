FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY services/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/leafguard ./src/leafguard
COPY services/backend/app ./app

RUN useradd --create-home leafguard && mkdir -p /app/reports /app/feedback /app/models \
    && chown -R leafguard /app
USER leafguard

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
