# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Build arg to optionally install voice dependencies
ARG INSTALL_VOICE=false

WORKDIR /app

# System deps for asyncpg and other C extensions
# espeak-ng is required by piper-tts for phoneme processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    if [ "$INSTALL_VOICE" = "true" ]; then \
        apt-get install -y --no-install-recommends espeak-ng; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install . && \
    if [ "$INSTALL_VOICE" = "true" ]; then \
        pip install --no-cache-dir --prefix=/install ".[voice]"; \
    fi

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Build arg carried to runtime for espeak-ng
ARG INSTALL_VOICE=false

WORKDIR /app

# Runtime deps only
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    if [ "$INSTALL_VOICE" = "true" ]; then \
        apt-get install -y --no-install-recommends espeak-ng; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Non-root user
RUN useradd --create-home --no-log-init appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
