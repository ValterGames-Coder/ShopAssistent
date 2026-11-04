FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 app
COPY shop_assistant ./shop_assistant
COPY app.py knowledge.json SYSTEM_PROMPT.md ./
RUN mkdir -p /app/reports && chown -R app:app /app
USER app

FROM base AS tests
COPY tests ./tests
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM base AS runtime
CMD ["python", "app.py", "bot"]
