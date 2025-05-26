#!/bin/bash
set -e

# Get the project root directory
PROJECT_ROOT=$(pwd)
echo "🔧 Fixing permissions for directories in ${PROJECT_ROOT}"

# Set Airflow directories permissions
echo "Setting Airflow directory permissions..."
mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins
sudo chown -R 50000:50000 ./airflow
sudo chmod -R 775 ./airflow
echo "✅ Airflow directories permissions fixed"

# Set MLflow directories permissions
echo "Setting MLflow directory permissions..."
mkdir -p ./mlruns ./mlflow_artifacts
sudo chown -R 0:0 ./mlruns ./mlflow_artifacts
sudo chmod -R 777 ./mlruns ./mlflow_artifacts
echo "✅ MLflow directories permissions fixed"

# Set other project directories
echo "Setting other project directories permissions..."
mkdir -p ./data/raw ./data/processed ./data/temp ./models/metrics ./config ./src
sudo chmod -R 775 ./data ./models ./config ./src
echo "✅ Other directories permissions fixed"

# Create and set permissions for __pycache__ directory
echo "Setting permissions for __pycache__ directories..."
mkdir -p ./airflow/dags/__pycache__
sudo chown -R 50000:50000 ./airflow/dags/__pycache__
sudo chmod -R 775 ./airflow/dags/__pycache__
echo "✅ __pycache__ directory permissions fixed"

echo "🔍 Verifying permissions..."
ls -la ./airflow ./mlruns ./mlflow_artifacts

echo "🚀 All permissions fixed. Restart your containers to apply changes."
echo "Run: docker compose down && docker compose up -d"