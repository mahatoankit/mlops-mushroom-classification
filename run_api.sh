#!/bin/bash

# Start the mushroom classification API server
# This script starts the FastAPI server for mushroom classification

# Set the base directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR" || exit 1

# Create log directory if it doesn't exist
mkdir -p logs

# Default port
PORT=${1:-8000}

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/api_${TIMESTAMP}.log"

echo "Starting API server at $(date)" | tee -a "$LOG_FILE"
echo "Port: $PORT" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Check if the model exists
if [ ! -f "models/xgboost.joblib" ]; then
    echo "ERROR: Model file not found. Please run the ETL pipeline first." | tee -a "$LOG_FILE"
    exit 1
fi

# Install uvicorn if not already installed
pip install -q uvicorn fastapi 2>&1 | tee -a "$LOG_FILE"

# Start the API server
echo "Starting API server..." | tee -a "$LOG_FILE"
uvicorn src.model_serving.api:app --host 0.0.0.0 --port "$PORT" --reload 2>&1 | tee -a "$LOG_FILE"
