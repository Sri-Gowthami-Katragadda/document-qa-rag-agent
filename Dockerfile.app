# ---- Streamlit UI (app.py only - pure API client, no ML deps) ----
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY app.py .

RUN useradd --create-home appuser
USER appuser

ENV API_BASE_URL=http://api:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]