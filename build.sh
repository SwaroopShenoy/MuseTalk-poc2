#!/bin/bash

# MuseTalk Docker Build and Run Script
set -e

echo "🚀 Building MuseTalk Docker Image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 > /dev/null 2>&1; then
    echo "❌ NVIDIA Docker runtime not available. Please install nvidia-docker2."
    exit 1
fi

# Create directories
mkdir -p input output configs

# Build the Docker image
echo "📦 Building Docker image (this may take 10-15 minutes)..."
docker build -t musetalk:latest .

echo "✅ Build complete!"

# Function to run inference
run_inference() {
    local video_path="$1"
    local audio_path="$2"
    
    if [[ -z "$video_path" || -z "$audio_path" ]]; then
        echo "Usage: $0 run <video_path> <audio_path>"
        echo "Example: $0 run input/video.mp4 input/audio.wav"
        exit 1
    fi
    
    if [[ ! -f "$video_path" ]]; then
        echo "❌ Video file not found: $video_path"
        exit 1
    fi
    
    if [[ ! -f "$audio_path" ]]; then
        echo "❌ Audio file not found: $audio_path"
        exit 1
    fi
    
    echo "🎬 Running MuseTalk inference..."
    echo "📹 Video: $video_path"
    echo "🎵 Audio: $audio_path"
    
    # Copy files to input directory
    cp "$video_path" input/
    cp "$audio_path" input/
    
    # Get filenames
    video_file=$(basename "$video_path")
    audio_file=$(basename "$audio_path")
    
    # Run inference
    docker run --rm --gpus all \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        musetalk:latest \
        python -m scripts.inference \
        --inference_config configs/inference/test.yaml \
        --result_dir /app/output \
        --unet_model_path models/musetalkV15/unet.pth \
        --unet_config models/musetalkV15/musetalk.json \
        --version v15 \
        --ffmpeg_path /usr/bin/ffmpeg \
        --video_path "/app/input/$video_file" \
        --audio_path "/app/input/$audio_file"
    
    echo "✅ Inference complete! Check the output directory."
}

# Function to run Gradio interface
run_gradio() {
    echo "🌐 Starting Gradio web interface..."
    docker run --rm --gpus all \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -p 7860:7860 \
        musetalk:latest \
        python app.py --use_float16 --ffmpeg_path /usr/bin/ffmpeg
}

# Function to enter container shell
run_shell() {
    echo "🐚 Entering MuseTalk container shell..."
    docker run --rm -it --gpus all \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        musetalk:latest \
        bash
}

# Main command handler
case "$1" in
    "build")
        echo "✅ Image already built. Use 'rebuild' to build again."
        ;;
    "rebuild")
        echo "🔄 Rebuilding Docker image..."
        docker build --no-cache -t musetalk:latest .
        ;;
    "run")
        run_inference "$2" "$3"
        ;;
    "gradio")
        run_gradio
        ;;
    "shell")
        run_shell
        ;;
    "clean")
        echo "🧹 Cleaning up Docker images..."
        docker rmi musetalk:latest 2>/dev/null || true
        docker system prune -f
        ;;
    *)
        echo "MuseTalk Docker Manager"
        echo ""
        echo "Commands:"
        echo "  build                    - Build Docker image"
        echo "  rebuild                  - Rebuild Docker image from scratch"
        echo "  run <video> <audio>      - Run lip-sync inference"
        echo "  gradio                   - Start web interface on port 7860"
        echo "  shell                    - Enter container shell"
        echo "  clean                    - Remove Docker images and cleanup"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 run input/video.mp4 input/audio.wav"
        echo "  $0 gradio"
        ;;
esac