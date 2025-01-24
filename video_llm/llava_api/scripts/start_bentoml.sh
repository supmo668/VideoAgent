#!/bin/bash

# Default values
PORT=3000

# Get service type from argument
SERVICE_TYPE=${1:-"combined"}

case $SERVICE_TYPE in
    "llava")
        echo "Starting LLaVA Video Service on port $PORT..."
        bentoml serve 'LLaVAVideoService:latest' --port $PORT
        ;;
    "twelvelabs")
        echo "Starting TwelveLabs API Service on port $PORT..."
        bentoml serve 'TwelveLabsAPIService:latest' --port $PORT
        ;;
    "full")
        echo "Starting Full LabAR Service on port $PORT..."
        bentoml serve 'LabARVideoReportingService:latest' --port $PORT
        ;;
    *)
        echo "Usage: $0 [service_type]"
        echo "Available service types:"
        echo "  llava      - LLaVA Video Service"
        echo "  twelvelabs - TwelveLabs API Service"
        echo "  full       - Full LabAR Service"
        ;;
esac