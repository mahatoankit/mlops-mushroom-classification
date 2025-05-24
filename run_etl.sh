#!/bin/bash

# Run the mushroom classification ETL pipeline
# This script executes the ETL pipeline and logs the output

# Set the base directory
BASE_DIR=$(dirname "$(readlink -f "$0")")
cd "$BASE_DIR" || exit 1

# Create log directory if it doesn't exist
mkdir -p logs

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_run_${TIMESTAMP}.log"

echo "Starting ETL pipeline at $(date)" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Create necessary directories if they don't exist
mkdir -p data/processed models/metrics

# Run the ETL pipeline
echo "Running ETL pipeline..." | tee -a "$LOG_FILE"
python src/pipeline.py --config config/config.yaml 2>&1 | tee -a "$LOG_FILE"

# Check if the pipeline succeeded
if [ $? -eq 0 ]; then
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
    echo "ETL pipeline completed successfully at $(date)" | tee -a "$LOG_FILE"
    echo "Output logs available at: $LOG_FILE" | tee -a "$LOG_FILE"
    
    # Copy the latest trained model for the API
    echo "Copying latest model for API serving..." | tee -a "$LOG_FILE"
    cp models/xgboost.joblib models/latest_model.joblib 2>&1 | tee -a "$LOG_FILE"
    
    # Extract feature names for the API
    echo "Extracting feature names for API..." | tee -a "$LOG_FILE"
    python -c "import pickle; print(','.join(pickle.load(open('data/processed/processed_data.pkl', 'rb'))['feature_names']))" > models/feature_names.txt 2>&1 | tee -a "$LOG_FILE"
    
    exit 0
else
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
    echo "ERROR: ETL pipeline failed at $(date)" | tee -a "$LOG_FILE"
    echo "Check logs at: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi
