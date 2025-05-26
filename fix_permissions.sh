#!/bin/bash
set -e

echo "🔧 Fixing Airflow directory permissions..."

# Get current user
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"

# Project root
PROJECT_ROOT="/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom"

# Fix ownership and permissions for airflow directory
echo "Fixing airflow directory permissions..."
sudo chown -R $CURRENT_USER:$CURRENT_USER "$PROJECT_ROOT/airflow/"
chmod -R 755 "$PROJECT_ROOT/airflow/"

# Make specific directories writable
echo "Making logs and other directories writable..."
chmod -R 777 "$PROJECT_ROOT/airflow/logs/"
chmod -R 777 "$PROJECT_ROOT/data/"
chmod -R 777 "$PROJECT_ROOT/models/"
chmod -R 755 "$PROJECT_ROOT/airflow/dags/"

# Fix MLflow directories
echo "Fixing MLflow directories..."
mkdir -p "$PROJECT_ROOT/mlruns"
mkdir -p "$PROJECT_ROOT/mlflow_artifacts"
chmod -R 777 "$PROJECT_ROOT/mlruns/"
chmod -R 777 "$PROJECT_ROOT/mlflow_artifacts/"

# Create necessary log directories
echo "Creating log directories..."
mkdir -p "$PROJECT_ROOT/logs"
chmod -R 777 "$PROJECT_ROOT/logs/"

echo "✅ Permissions fixed!"
echo "📝 Summary of changes:"
echo "  - airflow/: 755 (read/write for user, read for others)"
echo "  - airflow/logs/: 777 (full access for containers)"
echo "  - data/: 777 (full access for data processing)"
echo "  - models/: 777 (full access for model storage)"
echo "  - mlruns/: 777 (full access for MLflow)"
echo "  - logs/: 777 (full access for application logs)"