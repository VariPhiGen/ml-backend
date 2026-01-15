#!/bin/bash

# ===================================================
# YOLO ML Backend Automated Setup Script
# ===================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_success "Docker and Docker Compose are installed"
}

# Create .env file if it doesn't exist
create_env_file() {
    if [ ! -f ".env" ]; then
        print_info "Creating .env configuration file..."

        # Get user input for required fields
        echo ""
        echo "=================================="
        echo "Label Studio Configuration"
        echo "=================================="
        echo "Enter your Label Studio details:"
        echo "(Don't use localhost - use http://host.docker.internal:8080 for local installations)"
        echo ""

        read -p "Label Studio URL [http://host.docker.internal:8080]: " LS_HOST
        LS_HOST=${LS_HOST:-http://host.docker.internal:8080}

        read -p "Label Studio API Key: " LS_API_KEY
        if [ -z "$LS_API_KEY" ]; then
            print_error "API Key is required"
            exit 1
        fi

        echo ""
        echo "=================================="
        echo "Hybrid Mode (Optional)"
        echo "=================================="
        read -p "Enable YOLO + Grounding DINO hybrid mode? (y/N): " ENABLE_HYBRID
        case $ENABLE_HYBRID in
            [Yy]* ) USE_HYBRID=true;;
            * ) USE_HYBRID=false;;
        esac

        # Create .env file
        cat > .env << EOF
# ===================================================
# YOLO ML Backend Configuration
# ===================================================

# Label Studio Connection (REQUIRED)
LABEL_STUDIO_HOST=$LS_HOST
LABEL_STUDIO_API_KEY=$LS_API_KEY

# Hybrid Mode Configuration
USE_HYBRID_MODE=$USE_HYBRID

# Grounding DINO (only needed if USE_HYBRID_MODE=true)
GROUNDING_DINO_CONFIG=GroundingDINO_SwinT_OGC.py
GROUNDING_DINO_WEIGHTS=groundingdino_swint_ogc.pth
GROUNDING_DINO_BOX_THRESHOLD=0.3
GROUNDING_DINO_TEXT_THRESHOLD=0.25
GROUNDINGDINO_REPO_PATH=./GroundingDINO

# Performance Settings
LOG_LEVEL=INFO
WORKERS=2
THREADS=4
PORT=9090

# YOLO Settings
ALLOW_CUSTOM_MODEL_PATH=true
DEBUG_PLOT=false
MODEL_SCORE_THRESHOLD=0.5

# Build Settings
TEST_ENV=false
EOF

        print_success ".env file created successfully"
    else
        print_warning ".env file already exists. Skipping creation."
        print_info "Edit .env file to change configuration if needed."
    fi
}

# Download required models
download_models() {
    print_info "Ensuring model directories exist..."
    mkdir -p models cache_dir data

    # Download Grounding DINO model if hybrid mode is enabled
    if [ "$USE_HYBRID" = "true" ]; then
        print_warning "Hybrid mode enabled - Grounding DINO will be installed during Docker build"
        print_info "Grounding DINO model will be downloaded during build..."

        if [ ! -f "models/groundingdino_swint_ogc.pth" ]; then
            print_info "Downloading Grounding DINO weights..."
            wget -O models/groundingdino_swint_ogc.pth \
                https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
            print_success "Grounding DINO weights downloaded"
        else
            print_info "Grounding DINO weights already exist"
        fi
    fi
}

# Build and run the container
deploy() {
    print_info "Building Docker image..."
    docker compose build

    print_info "Starting YOLO ML Backend..."
    docker compose up -d

    print_info "Waiting for service to be ready..."
    sleep 10

    # Check if service is running
    if docker compose ps | grep -q "Up"; then
        print_success "YOLO ML Backend is running!"
        echo ""
        echo "=================================="
        echo "Service Information"
        echo "=================================="
        echo "URL: http://localhost:${PORT:-9090}"
        echo "Health Check: http://localhost:${PORT:-9090}/health"
        echo ""
        echo "Next steps:"
        echo "1. Open Label Studio and create/connect a project"
        echo "2. Add the ML Backend URL above in project settings"
        echo "3. Configure your labeling interface with RectangleLabels"
        echo "4. Start annotating!"
        echo ""
        print_success "Setup completed successfully!"
    else
        print_error "Failed to start service. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Show usage
show_usage() {
    echo "YOLO ML Backend Automated Setup"
    echo ""
    echo "Usage:"
    echo "  $0              # Interactive setup"
    echo "  $0 --help       # Show this help"
    echo "  $0 --quick      # Quick setup with existing .env"
    echo ""
    echo "The setup will:"
    echo "1. Check Docker installation"
    echo "2. Create/update .env configuration"
    echo "3. Download required models"
    echo "4. Build and deploy the service"
}

# Main script
main() {
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --quick)
            if [ ! -f ".env" ]; then
                print_error ".env file not found. Run without --quick for interactive setup."
                exit 1
            fi
            # Source .env file to get USE_HYBRID_MODE
            if [ -f ".env" ]; then
                export $(grep -v '^#' .env | xargs)
            fi
            ;;
        *)
            ;;
    esac

    echo "=================================="
    echo "YOLO ML Backend Setup"
    echo "=================================="

    check_docker
    create_env_file

    # Source .env file to get configuration
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    download_models
    deploy
}

# Run main function
main "$@"