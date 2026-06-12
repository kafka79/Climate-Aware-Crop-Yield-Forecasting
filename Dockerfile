# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install compilation headers and basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch and other requirements to prevent massive CUDA wheels from bloating the image
RUN pip install --no-cache-dir --user -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies including GDAL, GEOS, PROJ (required for rasterio/geopandas) and curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Secure container footprint: run as a non-privileged system user
RUN groupadd -r appgroup && useradd -m -r -g appgroup -s /bin/bash appuser

WORKDIR /home/appuser/app

# Copy Python packages from builder stage, changing owner to user
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appgroup . .

# Environment path configuration for user-level packages
ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

EXPOSE 8501 8000

# Efficient, non-blocking health check using curl instead of initializing Python interpreter
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

