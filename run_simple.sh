#!/bin/bash
# Simplified script for Mushroom Classification with MLOps
# This single script replaces all the separate scripts

ACTION=${1:-"run-all"}  # Default action: run everything

PROJECT_DIR="$(pwd)"
MLFLOW_PORT=5000
AIRFLOW_PORT=8080
MLFLOW_PID_FILE="$PROJECT_DIR/.mlflow.pid"
AIRFLOW_PID_FILE="$PROJECT_DIR/.airflow.pid"
LOG_DIR="$PROJECT_DIR/logs"
EXPERIMENT_NAME="mushroom-classification-pipeline"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Function to display script usage
show_usage() {
  echo "Usage: ./run_simple.sh [ACTION]"
  echo ""
  echo "Available actions:"
  echo "  run-all       - Start MLflow, Airflow and run training (default)"
  echo "  start-mlflow  - Only start MLflow server"
  echo "  start-airflow - Only start Airflow webserver"
  echo "  train         - Only run model training"
  echo "  check-tracking- Check MLflow experiments and Airflow DAGs"
  echo "  stop          - Stop all running services"
  echo "  help          - Show this help message"
  echo ""
}

# Function to start MLflow server with experiment tracking
start_mlflow() {
  echo "🚀 Starting MLflow server with experiment tracking..."
  
  # Check if running in Docker
  if [ -f "/.dockerenv" ]; then
    echo "🐳 Running in Docker - MLflow should be managed by docker-compose"
    return 0
  fi
  
  # Check if MLflow is already running
  if [ -f "$MLFLOW_PID_FILE" ]; then
    pid=$(cat "$MLFLOW_PID_FILE")
    if ps -p $pid > /dev/null; then
      echo "✅ MLflow is already running (PID: $pid)"
      echo "   Access MLflow UI at: http://localhost:$MLFLOW_PORT"
      return 0
    else
      # Remove stale PID file
      rm "$MLFLOW_PID_FILE"
    fi
  fi
  
  # Create mlruns directory if it doesn't exist
  mkdir -p "$PROJECT_DIR/mlruns"
  
  # Start MLflow server with experiment tracking
  mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    --backend-store-uri sqlite:///$PROJECT_DIR/mlflow.db \
    --default-artifact-root $PROJECT_DIR/mlruns \
    > "$LOG_DIR/mlflow.log" 2>&1 &
  
  mlflow_pid=$!
  echo $mlflow_pid > "$MLFLOW_PID_FILE"
  sleep 5
  
  # Check if MLflow started successfully
  if ps -p $mlflow_pid > /dev/null; then
    echo "✅ MLflow server started (PID: $mlflow_pid)"
    echo "   Access MLflow UI at: http://localhost:$MLFLOW_PORT"
    
    # Create experiment if it doesn't exist
    python -c "
import mlflow
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:$MLFLOW_PORT'
try:
    experiment = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
    if experiment is None:
        experiment_id = mlflow.create_experiment('$EXPERIMENT_NAME')
        print(f'✅ Created MLflow experiment: $EXPERIMENT_NAME (ID: {experiment_id})')
    else:
        print(f'✅ MLflow experiment exists: $EXPERIMENT_NAME (ID: {experiment.experiment_id})')
except Exception as e:
    print(f'⚠️ Could not create/verify experiment: {e}')
"
  else
    echo "❌ Failed to start MLflow server. Check logs at: $LOG_DIR/mlflow.log"
    exit 1
  fi
}

# Function to start Airflow webserver
start_airflow() {
  echo "🚀 Starting Airflow webserver..."
  
  # Check if running in Docker
  if [ -f "/.dockerenv" ]; then
    echo "🐳 Running in Docker - Airflow should be managed by docker-compose"
    return 0
  fi
  
  # Check if Airflow is already running
  if [ -f "$AIRFLOW_PID_FILE" ]; then
    pid=$(cat "$AIRFLOW_PID_FILE")
    if ps -p $pid > /dev/null; then
      echo "✅ Airflow is already running (PID: $pid)"
      echo "   Access Airflow UI at: http://localhost:$AIRFLOW_PORT"
      return 0
    else
      rm "$AIRFLOW_PID_FILE"
    fi
  fi
  
  # Set Airflow home
  export AIRFLOW_HOME="$PROJECT_DIR/airflow"
  mkdir -p "$AIRFLOW_HOME/dags"
  
  # Initialize Airflow database if needed
  if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo "🔧 Initializing Airflow database..."
    airflow db init > "$LOG_DIR/airflow_init.log" 2>&1
    
    # Create admin user
    airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com \
      --password admin > "$LOG_DIR/airflow_user.log" 2>&1
  fi
  
  # Start Airflow scheduler in background
  airflow scheduler > "$LOG_DIR/airflow_scheduler.log" 2>&1 &
  scheduler_pid=$!
  
  # Start Airflow webserver
  airflow webserver --port $AIRFLOW_PORT > "$LOG_DIR/airflow_webserver.log" 2>&1 &
  webserver_pid=$!
  echo "$webserver_pid,$scheduler_pid" > "$AIRFLOW_PID_FILE"
  
  sleep 5
  
  if ps -p $webserver_pid > /dev/null; then
    echo "✅ Airflow webserver started (PID: $webserver_pid)"
    echo "   Access Airflow UI at: http://localhost:$AIRFLOW_PORT"
    echo "   Username: admin, Password: admin"
  else
    echo "❌ Failed to start Airflow. Check logs at: $LOG_DIR/airflow_*.log"
  fi
}

# Function to check MLflow experiments and Airflow DAGs
check_tracking() {
  echo "🔍 Checking MLflow experiments and Airflow DAGs..."
  
  # Check MLflow experiments
  python -c "
import mlflow
import requests
import os
import sys

# Check MLflow
try:
    mlflow.set_tracking_uri('http://localhost:$MLFLOW_PORT')
    experiments = mlflow.search_experiments()
    print(f'📊 MLflow Experiments Found: {len(experiments)}')
    for exp in experiments:
        runs = mlflow.search_runs(exp.experiment_id)
        print(f'  - {exp.name}: {len(runs)} runs')
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            print(f'    Latest run: {latest_run[\"run_id\"][:8]}... (Status: {latest_run[\"status\"]})')
except Exception as e:
    print(f'❌ MLflow check failed: {e}')

# Check Airflow DAGs
try:
    response = requests.get('http://localhost:$AIRFLOW_PORT/api/v1/dags', 
                          auth=('admin', 'admin'), timeout=5)
    if response.status_code == 200:
        dags = response.json()
        print(f'\\n🔄 Airflow DAGs Found: {dags.get(\"total_entries\", 0)}')
        for dag in dags.get('dags', []):
            print(f'  - {dag[\"dag_id\"]}: {dag[\"is_active\"]} (Paused: {dag[\"is_paused\"]})')
    else:
        print(f'\\n⚠️ Could not access Airflow API (Status: {response.status_code})')
except Exception as e:
    print(f'\\n⚠️ Airflow check failed: {e}')
"
}

# Function to train models
train_models() {
  echo "🔍 Checking for dataset..."
  
  # Check if we have a dataset to use - prioritize secondary_data.csv
  SECONDARY_DATA_FILE="$PROJECT_DIR/data/raw/secondary_data.csv"
  DATA_FILE="$PROJECT_DIR/data/raw/fraction_of_dataset.csv"
  
  if [ -f "$SECONDARY_DATA_FILE" ]; then
    echo "✅ Using secondary data file: $SECONDARY_DATA_FILE"
    ACTUAL_DATA_FILE="$SECONDARY_DATA_FILE"
  elif [ -f "$DATA_FILE" ]; then
    echo "✅ Using fraction data file: $DATA_FILE"
    ACTUAL_DATA_FILE="$DATA_FILE"
  else
    echo "⚠️ No data file found. Creating sample data file."
    mkdir -p "$PROJECT_DIR/data/raw"
    ACTUAL_DATA_FILE="$PROJECT_DIR/data/raw/sample_data.csv"
  fi
  
  echo "🧠 Training models with MLflow tracking..."
  
  python -c "
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('$PROJECT_DIR')
from src.train import train_models
from src.monitoring import load_config
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from datetime import datetime

# Set MLflow tracking
mlflow.set_tracking_uri('http://localhost:$MLFLOW_PORT')
mlflow.set_experiment('$EXPERIMENT_NAME')

# Load configuration
config = load_config('config/config.yaml')

# Data file to use
DATA_FILE = '$ACTUAL_DATA_FILE'

def create_etl_pipeline(data_path):
    '''ETL pipeline matching the notebook methodology'''
    
    with mlflow.start_run(run_name=f'etl_pipeline_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}'):
        mlflow.log_param('pipeline_stage', 'ETL')
        mlflow.log_param('data_source', data_path)
        
        if not os.path.exists(data_path):
            print('Creating sample mushroom dataset...')
            # Create sample data that matches the notebook structure
            np.random.seed(42)
            n_samples = 1000
            
            # Create sample data with proper mushroom features
            sample_data = pd.DataFrame({
                'class': np.random.choice(['e', 'p'], n_samples),
                'cap_diameter': np.random.normal(50, 20, n_samples),
                'stem_height': np.random.normal(100, 30, n_samples), 
                'stem_width': np.random.normal(10, 3, n_samples),
                'cap_shape': np.random.choice(['bell', 'conical', 'flat'], n_samples),
                'cap_color': np.random.choice(['brown', 'white', 'red'], n_samples),
                'habitat': np.random.choice(['forest', 'urban', 'meadow'], n_samples),
                'does_bruise_or_bleed': np.random.choice(['t', 'f'], n_samples),
                'has_ring': np.random.choice(['t', 'f'], n_samples),
                'gill_color': np.random.choice(['brown', 'white', 'black'], n_samples),
                'stem_color': np.random.choice(['brown', 'white', 'yellow'], n_samples)
            })
            
            # Ensure positive values for physical measurements
            sample_data['cap_diameter'] = np.abs(sample_data['cap_diameter'])
            sample_data['stem_height'] = np.abs(sample_data['stem_height'])
            sample_data['stem_width'] = np.abs(sample_data['stem_width'])
            
            # Save sample data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            sample_data.to_csv(data_path, index=False)
            print(f'Sample data saved to {data_path}')
            
            mlflow.log_param('data_type', 'generated_sample')
            mlflow.log_metric('sample_size', n_samples)
            
            return sample_data
        
        else:
            print(f'Loading data from {data_path}...')
            # Load data with proper delimiter handling
            try:
                df = pd.read_csv(data_path, delimiter=';')
            except:
                try:
                    df = pd.read_csv(data_path, delimiter=',')
                except Exception as e:
                    print(f'Error loading data: {e}')
                    return create_etl_pipeline(data_path.replace('.csv', '_sample.csv'))
            
            # Clean column names (matching notebook approach)
            df.columns = df.columns.str.replace('-', '_').str.strip()
            print(f'Data loaded with shape: {df.shape}')
            
            mlflow.log_param('data_type', 'loaded_from_file')
            mlflow.log_metric('original_rows', df.shape[0])
            mlflow.log_metric('original_columns', df.shape[1])
            
            return df

def preprocess_data(df):
    '''Data preprocessing matching the notebook methodology'''
    
    with mlflow.start_run(run_name=f'preprocessing_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}', nested=True):
        mlflow.log_param('preprocessing_stage', 'data_cleaning')
        mlflow.log_metric('input_rows', df.shape[0])
        mlflow.log_metric('input_columns', df.shape[1])
        
        # Handle missing values and encoding (simplified version of notebook approach)
        label_encoder = LabelEncoder()
        
        # Create encoded target if it doesn't exist
        if 'class_encoded' not in df.columns and 'class' in df.columns:
            df['class_encoded'] = label_encoder.fit_transform(df['class'])
        
        # Handle boolean columns
        for col in ['does_bruise_or_bleed', 'has_ring']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[f'{col}_encoded'] = label_encoder.fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = df[col]
        
        # Remove outliers for numerical columns (simplified)
        numerical_cols = ['cap_diameter', 'stem_height', 'stem_width']
        outliers_removed = 0
        for col in numerical_cols:
            if col in df.columns:
                original_size = len(df)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                outliers_removed += original_size - len(df)
        
        mlflow.log_metric('outliers_removed', outliers_removed)
        
        # One-hot encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'class']
        
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            mlflow.log_metric('categorical_columns_encoded', len(categorical_cols))
        
        # Drop original categorical columns that were encoded
        columns_to_drop = ['class', 'does_bruise_or_bleed', 'has_ring']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        df = df.reset_index(drop=True)
        print(f'Data preprocessed to shape: {df.shape}')
        
        mlflow.log_metric('output_rows', df.shape[0])
        mlflow.log_metric('output_columns', df.shape[1])
        mlflow.log_param('preprocessing_complete', True)
        
        return df

# Run ETL pipeline with MLflow tracking
try {
    with mlflow.start_run(run_name=f'mushroom_classification_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}'):
        mlflow.log_param('pipeline_type', 'mushroom_classification')
        mlflow.log_param('mlflow_tracking_uri', 'http://localhost:$MLFLOW_PORT')
        
        # Extract
        raw_data = create_etl_pipeline(DATA_FILE)
        
        # Transform
        processed_data = preprocess_data(raw_data)
        
        # Validate processed data
        if processed_data.empty or processed_data.shape[1] == 0:
            raise ValueError('Processed data is empty')
        
        print(f'Final processed data shape: {processed_data.shape}')
        print(f'Columns: {list(processed_data.columns)}')
        
        # Save processed data
        processed_data_path = DATA_FILE.replace('.csv', '_processed.csv')
        processed_data.to_csv(processed_data_path, index=False)
        mlflow.log_artifact(processed_data_path, 'processed_data')
        
        # Load (train models using file path)
        print('\\nTraining models with MLflow tracking...')
        best_model, best_accuracy = train_models(processed_data_path, config)
        
        # Log final results
        mlflow.log_metric('best_accuracy', best_accuracy)
        mlflow.log_param('best_model_type', best_model)
        mlflow.set_tag('pipeline_status', 'completed')
        
        print(f'\\n✅ Training complete!')
        print(f'✅ Best model: {best_model} with accuracy: {best_accuracy:.4f}')
        print('✅ Results tracked in MLflow')
        print(f'✅ Experiment: $EXPERIMENT_NAME')
        
except Exception as e {
    mlflow.log_param('error', str(e))
    mlflow.set_tag('pipeline_status', 'failed')
    print(f'Error in ETL pipeline: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
}
  
  if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully"
    echo "🔍 Checking experiment tracking..."
    check_tracking
  else
    echo "❌ Model training failed"
    exit 1
  fi
}

# Function to stop all services
stop_services() {
  echo "🛑 Stopping services..."
  
  # Stop MLflow server
  if [ -f "$MLFLOW_PID_FILE" ]; then
    pid=$(cat "$MLFLOW_PID_FILE")
    if ps -p $pid > /dev/null; then
      echo "  Stopping MLflow server (PID: $pid)..."
      kill $pid
    fi
    rm "$MLFLOW_PID_FILE"
  fi
  
  # Stop Airflow services
  if [ -f "$AIRFLOW_PID_FILE" ]; then
    pids=$(cat "$AIRFLOW_PID_FILE")
    IFS=',' read -ra PID_ARRAY <<< "$pids"
    for pid in "${PID_ARRAY[@]}"; do
      if ps -p $pid > /dev/null; then
        echo "  Stopping Airflow service (PID: $pid)..."
        kill $pid
      fi
    done
    rm "$AIRFLOW_PID_FILE"
  fi
  
  echo "✅ All services stopped"
}

# Add Docker helper functions
show_docker_usage() {
  echo "🐳 Docker Usage:"
  echo ""
  echo "1. Build and start all services:"
  echo "   docker-compose up --build"
  echo ""
  echo "2. Start in background:"
  echo "   docker-compose up -d --build"
  echo ""
  echo "3. View logs:"
  echo "   docker-compose logs -f mushroom-app"
  echo ""
  echo "4. Stop all services:"
  echo "   docker-compose down"
  echo ""
  echo "5. Clean up volumes:"
  echo "   docker-compose down -v"
  echo ""
  echo "Access URLs:"
  echo "  - MLflow UI: http://localhost:5000"
  echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
  echo "  - Standalone MLflow: http://localhost:5001"
}

# Main script logic based on action
case "$ACTION" in
  "run-all")
    echo "========================================================"
    echo "🚀 Running simplified Mushroom Classification MLOps pipeline"
    echo "========================================================"
    start_mlflow
    start_airflow
    train_models
    echo ""
    echo "🎉 Everything is ready!"
    echo "   MLflow UI: http://localhost:$MLFLOW_PORT"
    echo "   Airflow UI: http://localhost:$AIRFLOW_PORT (admin/admin)"
    ;;
  
  "start-mlflow")
    echo "========================================================"
    echo "🚀 Starting MLflow server only"
    echo "========================================================"
    start_mlflow
    ;;
  
  "start-airflow")
    echo "========================================================"
    echo "🚀 Starting Airflow webserver only"
    echo "========================================================"
    start_airflow
    ;;
  
  "train")
    echo "========================================================"
    echo "🚀 Running model training only"
    echo "========================================================"
    train_models
    ;;
  
  "check-tracking")
    echo "========================================================"
    echo "🚀 Checking MLflow experiments and Airflow DAGs"
    echo "========================================================"
    check_tracking
    ;;
  
  "stop")
    echo "========================================================"
    echo "🚀 Stopping all services"
    echo "========================================================"
    stop_services
    ;;
  
  "help")
    show_usage
    ;;
  
  "docker-help")
    echo "========================================================"
    echo "🐳 Docker Setup Instructions"
    echo "========================================================"
    show_docker_usage
    ;;
  
  *)
    echo "❌ Unknown action: $ACTION"
    show_usage
    exit 1
    ;;
esac

echo "========================================================"
