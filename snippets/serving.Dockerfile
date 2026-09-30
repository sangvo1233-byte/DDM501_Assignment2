# Multi-stage build.
#
# Stage 1 compiles wheels — which needs a compiler for anything without a
# manylinux wheel. Stage 2 installs them into a clean image that has no
# compiler at all. The result is smaller and has a smaller attack surface:
# a build toolchain in a production image is a liability, not a convenience.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# =============================================================================
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE: no .pyc files in a read-only-ish container.
# PYTHONUNBUFFERED: logs appear in `docker logs` as they happen rather than
#   when the buffer fills — the difference between debugging an incident and
#   waiting for one.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Run as a non-root user. A container that is root inside is one kernel
# escape away from being root outside, and nothing here needs the privilege.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The health check probes /health, not /. A check that only proves the process
# is listening will pass happily while every prediction returns 503.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
