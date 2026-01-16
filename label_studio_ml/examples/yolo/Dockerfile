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
    CUDA_HOME=/usr/local/cuda \
    GROUNDINGDINO_REPO_PATH=/GroundingDINO
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# Install Python dependencies
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-cache-dir -r requirements-base.txt

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install --no-cache-dir --force-reinstall ultralytics==8.2.0 && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python -c "from ultralytics import YOLO; print('✅ YOLO import successful')"

# Install OWL-ViT for zero-shot detection (reliable alternative to Grounding DINO)
RUN if [ "$USE_HYBRID_MODE" = "true" ]; then \
        echo "Installing OWL-ViT for zero-shot detection..." && \
        pip3 install --no-cache-dir transformers torch torchvision timm pillow; \
    fi

WORKDIR /app

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

# Grounding DINO weights are already downloaded during installation above

# Health check endpoint
RUN echo '#!/bin/bash\necho "OK"' > /app/health.sh && chmod +x /app/health.sh

ENV PYTHONPATH=/app

# Expose port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["/app/start.sh"]
