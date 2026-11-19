# ===================================================
# Production Dockerfile for X-Ray Parcel Verify
# Base: Python 3.10 slim (Debian Bookworm)
# ===================================================

FROM python:3.10-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/yolov7" \
    YOLOV7_PATH="/app/yolov7" \
    PORT=8000 \
    HOST="0.0.0.0" \
    DEVICE="cpu" \
    TORCH_THREADS=2

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for OpenCV, Git (for YOLOv7 clone), and Curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone official YOLOv7 repository for architecture and inference utilities
RUN git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /app/yolov7

# Upgrade pip and install CPU-only PyTorch & application dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/static/my_results && \
    chown -R appuser:appuser /app

# Copy application source code and model weights
COPY client_code.py /app/
COPY 24_july_detection.pt /app/
COPY best.pt /app/
COPY static /app/static

# Set permissions for non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI server port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "client_code:app", "--host", "0.0.0.0", "--port", "8000"]
