FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV
ARG USE_HYBRID_MODE

WORKDIR /app

# Update conda and install system dependencies
RUN conda update conda -y && \
    apt-get update && \
    apt-get install -y \
        git wget curl g++ freeglut3-dev build-essential ninja-build \
        libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev \
        ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# Install Python dependencies
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Grounding DINO will be installed at runtime if USE_HYBRID_MODE=true
# This avoids build-time CUDA compilation issues

# Install test requirements if needed
COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then \
        pip install --no-cache-dir -r requirements-test.txt; \
    fi

# Copy application code
COPY . ./

# Create necessary directories
RUN mkdir -p models cache_dir data

# Download YOLO models automatically
RUN echo "Downloading YOLO models..." && \
    yolo predict model=yolov8m.pt source=tests/car.jpg && \
    yolo predict model=yolov8n.pt source=tests/car.jpg && \
    yolo predict model=yolov8n-cls.pt source=tests/car.jpg && \
    yolo predict model=yolov8n-seg.pt source=tests/car.jpg && \
    yolo predict model=yolov8n-pose.pt source=tests/car.jpg && \
    yolo predict model=yolov8n-obb.pt source=tests/car.jpg && \
    mv *.pt models/ 2>/dev/null || true

# Download Grounding DINO model if hybrid mode is enabled
RUN if [ "$USE_HYBRID_MODE" = "true" ]; then \
        echo "Downloading Grounding DINO model..." && \
        wget -O models/groundingdino_swint_ogc.pth \
            https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth; \
    fi

# Health check endpoint
RUN echo '#!/bin/bash\necho "OK"' > /app/health.sh && chmod +x /app/health.sh

ENV PYTHONPATH=/app

# Expose port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["/app/start.sh"]
