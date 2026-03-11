# Multi-stage build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
# Install CPU-only torch first to avoid downloading the 2.5 GB GPU build.
# pip --index-url only applies to one command, so torch must be separate.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/      app/
COPY src/      src/
COPY models/   models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
