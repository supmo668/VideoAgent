#!/bin/bash

# Default values
PORT=3000

# Get service type from argument
SERVICE_TYPE=${1:-"combined"}

case $SERVICE_TYPE in
    "llava")
        echo "Starting LLaVA Video Service on port $PORT..."
        bentoml serve 'service:LLaVAVideoService' --port $PORT
        ;;
    "twelvelabs")
        echo "Starting TwelveLabs API Service on port $PORT..."
        bentoml serve 'service:TwelveLabsAPIService' --port $PORT
        ;;
    "full")
        echo "Starting Full LabAR Service on port $PORT..."
        # Set CUDA environment variables for memory optimization
        export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Limit maximum CUDA memory splits
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Enable expandable segments for memory optimization

        # Set environment variables for better GPU memory management
        export CUDA_LAUNCH_BLOCKING=1  # More predictable GPU memory allocation
        export TORCH_USE_CUDA_DSA=1  # Enable CUDA graph memory optimization

        # Start BentoML service with worker optimization
        bentoml serve service:LabARVideoReportingService
        ;;
    *)
        echo "Usage: $0 [service_type]"
        echo "Available service types:"
        echo "  llava      - LLaVA Video Service"
        echo "  twelvelabs - TwelveLabs API Service"
        echo "  full       - Full LabAR Service"
        ;;
esac