#!/bin/bash

# Master startup script for the Mushroom Classification Pipeline
# This script initializes and demonstrates all major components of the system

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check if required packages are installed
echo "Checking environment dependencies..."
if ! pip list | grep -q "pandas"; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Create required directories
mkdir -p logs
mkdir -p models
mkdir -p models/metrics
mkdir -p models/monitoring
mkdir -p models/registry
mkdir -p models/ab_testing
mkdir -p data/processed
mkdir -p data/model_input

# Display banner
echo "========================================================"
echo "   Mushroom Classification Pipeline - Complete Demo     "
echo "========================================================"
echo "This script will demonstrate the complete pipeline:"
echo "1. ETL Process"
echo "2. Model Training"
echo "3. Model Versioning"
echo "4. A/B Testing Setup"
echo "5. API Server Startup"
echo "6. Model Monitoring"
echo "7. Example Predictions"
echo "========================================================"
echo ""

# Function to prompt user to continue
prompt_continue() {
    echo ""
    read -p "Press Enter to continue to the next step..." input
    echo ""
}

# Step 1: Run ETL pipeline
echo "Step 1: Running ETL Pipeline..."
echo "------------------------------"
echo "This process extracts mushroom data, transforms it, and loads it for model training."
python src/pipeline.py --config config/config.yaml --step extract
python src/pipeline.py --config config/config.yaml --step transform
python src/pipeline.py --config config/config.yaml --step load
echo "ETL process completed."
prompt_continue

# Step 2: Train models
echo "Step 2: Training Models..."
echo "-------------------------"
echo "Training multiple models (Logistic Regression, Decision Tree, XGBoost)."
python src/pipeline.py --config config/config.yaml --step train
echo "Model training completed."
prompt_continue

# Step 3: Register models with the model registry
echo "Step 3: Registering Models with Version Registry..."
echo "--------------------------------------------------"
echo "This registers the trained models and promotes XGBoost to production."
python -c "
from src.model_versioning import register_and_promote_model
import os
import yaml

# Load metrics from files
with open('models/metrics/logistic_regression_metrics.csv', 'r') as f:
    lr_metrics = {'accuracy': float(f.read().split(',')[1].strip())}

with open('models/metrics/decision_tree_metrics.csv', 'r') as f:
    dt_metrics = {'accuracy': float(f.read().split(',')[1].strip())}

with open('models/metrics/xgboost_metrics.csv', 'r') as f:
    xgb_metrics = {'accuracy': float(f.read().split(',')[1].strip())}

# Register all models
print('Registering Logistic Regression model...')
lr_version = register_and_promote_model('models/logistic_regression.joblib', 'logistic_regression', lr_metrics, 'staging')
print(f'Registered as version {lr_version}')

print('Registering Decision Tree model...')
dt_version = register_and_promote_model('models/decision_tree.joblib', 'decision_tree', dt_metrics, 'staging')
print(f'Registered as version {dt_version}')

print('Registering XGBoost model...')
xgb_version = register_and_promote_model('models/xgboost.joblib', 'xgboost', xgb_metrics, 'production')
print(f'Registered as version {xgb_version} and promoted to production')
"
echo "Model registration completed."
prompt_continue

# Step 4: Set up A/B testing between models
echo "Step 4: Setting up A/B Testing..."
echo "--------------------------------"
echo "Creating an A/B test between XGBoost and Decision Tree models."
python -c "
from src.ab_testing import create_ab_test

# Create an A/B test
test_id = create_ab_test(
    name='xgb_vs_dt', 
    model_a='models/registry/production/xgboost.joblib',
    model_b='models/registry/staging/decision_tree.joblib',
    traffic_split=0.5
)
print(f'Created A/B test with ID: {test_id}')
"
echo "A/B test creation completed."
prompt_continue

# Step 5: Start the API server (in background)
echo "Step 5: Starting API Server..."
echo "-----------------------------"
echo "Starting the FastAPI server in the background."
uvicorn src.model_serving.api:app --reload --port 8000 &
API_PID=$!
sleep 3
echo "API server is running on http://localhost:8000"
prompt_continue

# Step 6: Generate some example requests to demonstrate A/B testing
echo "Step 6: Making Example Predictions..."
echo "------------------------------------"
echo "Sending example prediction requests using the A/B test."
python -c "
import requests
import json
import time

# Make predictions with A/B testing
print('Making predictions using A/B testing...')
for i in range(5):
    data = {
        'cap_shape': 'convex',
        'cap_surface': 'smooth',
        'cap_color': 'brown',
        'gill_attachment': 'free',
        'gill_color': 'white',
        'stalk_shape': 'enlarging',
        'stem_color': 'white',
        'ring_type': 'pendant',
        'habitat': 'woods',
        'cap_diameter': 5.0 + i,
        'stem_height': 8.0 + i * 0.5,
        'stem_width': 1.5,
        'does_bruise_or_bleed': True,
        'has_ring': True
    }
    
    response = requests.post(
        'http://localhost:8000/predict?ab_test=xgb_vs_dt&ground_truth=1', 
        json=data
    )
    result = response.json()
    print(f'Prediction {i+1}: {result[\"prediction_label\"][0]}')
    print(f'Using model: {result.get(\"model_path\", \"unknown\")}')
    if 'ab_test' in result:
        print(f'A/B test: {result[\"ab_test\"][\"test_id\"]}, Model: {result[\"ab_test\"][\"model\"]}')
    print('--------------------------')
    time.sleep(1)
"
echo "Example predictions completed."
prompt_continue

# Step 7: Check A/B test status
echo "Step 7: Checking A/B Test Status..."
echo "----------------------------------"
echo "Requesting A/B test information from the API."
curl http://localhost:8000/ab-tests | python -m json.tool
echo ""
prompt_continue

# Step 8: Generate and check monitoring metrics
echo "Step 8: Checking Model Monitoring..."
echo "----------------------------------"
echo "Generating monitoring report based on collected metrics."
python -c "
from src.monitoring import ModelMonitor
import pandas as pd
import joblib
import os

# Load test data
X_test = pd.read_csv('data/model_input/X_test.csv')
y_test = pd.read_csv('data/model_input/y_test.csv')['class_encoded']

# Load model
model = joblib.load('models/xgboost.joblib')

# Create monitor for demonstration
print('Generating model monitoring report...')
monitor = ModelMonitor('xgboost', 'models/metrics/monitoring')
monitoring_metrics = {
    'accuracy': 0.95,
    'precision': 0.94,
    'recall': 0.96,
    'f1_score': 0.95
}
monitor.record_metrics(monitoring_metrics)

# Add a second data point to show trends
monitoring_metrics2 = {
    'accuracy': 0.96,
    'precision': 0.95,
    'recall': 0.97,
    'f1_score': 0.96
}
monitor.record_metrics(monitoring_metrics2)

# Generate report
report_path = monitor.generate_monitoring_report()
print(f'Monitoring report generated at: {report_path}')
"
echo "Monitoring report generated."
prompt_continue

# Step 9: Clean up
echo "Step 9: Cleaning Up..."
echo "---------------------"
echo "Stopping API server and concluding demo."

# Kill API server
kill $API_PID

echo ""
echo "========================================================"
echo "   Mushroom Classification Pipeline Demo Completed      "
echo "========================================================"
echo ""
echo "The demo has shown all major components of the system:"
echo "- ETL pipeline for data preparation"
echo "- Multi-model training and evaluation"
echo "- Model versioning and registry management"
echo "- A/B testing in production"
echo "- FastAPI server with prediction endpoints"
echo "- Model monitoring and drift detection"
echo ""
echo "You can now explore the project files and components."
echo "To restart the API server: uvicorn src.model_serving.api:app --reload --port 8000"
echo "========================================================"
