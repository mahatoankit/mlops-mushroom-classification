#!/bin/bash
# Simplified script for Mushroom Classification with MLOps
# This single script replaces all the separate scripts

ACTION=${1:-"run-all"}  # Default action: run everything

PROJECT_DIR="$(pwd)"
MLFLOW_PORT=5000
MLFLOW_PID_FILE="$PROJECT_DIR/.mlflow.pid"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Function to display script usage
show_usage() {
  echo "Usage: ./run_simple.sh [ACTION]"
  echo ""
  echo "Available actions:"
  echo "  run-all      - Start MLflow and run training (default)"
  echo "  start-mlflow - Only start MLflow server"
  echo "  train        - Only run model training"
  echo "  stop         - Stop all running services"
  echo "  help         - Show this help message"
  echo ""
}

# Function to start MLflow server
start_mlflow() {
  echo "🚀 Starting MLflow server..."
  
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
  
  # Start MLflow server
  mlflow server \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    --backend-store-uri sqlite:///$PROJECT_DIR/mlflow.db \
    --default-artifact-root $PROJECT_DIR/mlruns \
    > "$LOG_DIR/mlflow.log" 2>&1 &
  
  mlflow_pid=$!
  echo $mlflow_pid > "$MLFLOW_PID_FILE"
  sleep 3
  
  # Check if MLflow started successfully
  if ps -p $mlflow_pid > /dev/null; then
    echo "✅ MLflow server started (PID: $mlflow_pid)"
    echo "   Access MLflow UI at: http://localhost:$MLFLOW_PORT"
  else
    echo "❌ Failed to start MLflow server. Check logs at: $LOG_DIR/mlflow.log"
    exit 1
  fi
}

# Function to train models
train_models() {
  echo "🔍 Checking for dataset..."
  
  # Check if we have a dataset to use
  DATA_FILE="$PROJECT_DIR/data/raw/fraction_of_dataset.csv"
  SECONDARY_DATA_FILE="$PROJECT_DIR/data/raw/secondary_data.csv"
  
  if [ -f "$DATA_FILE" ]; then
    echo "✅ Using existing data file: $DATA_FILE"
    ACTUAL_DATA_FILE="$DATA_FILE"
  elif [ -f "$SECONDARY_DATA_FILE" ]; then
    echo "✅ Using secondary data file: $SECONDARY_DATA_FILE"
    ACTUAL_DATA_FILE="$SECONDARY_DATA_FILE"
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

# Load configuration
config = load_config('config/config.yaml')

# Data file to use
DATA_FILE = '$ACTUAL_DATA_FILE'

def create_etl_pipeline(data_path):
    '''ETL pipeline matching the notebook methodology'''
    
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
        return df

def preprocess_data(df):
    '''Data preprocessing matching the notebook methodology'''
    
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
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'class']
    
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop original categorical columns that were encoded
    columns_to_drop = ['class', 'does_bruise_or_bleed', 'has_ring']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    df = df.reset_index(drop=True)
    print(f'Data preprocessed to shape: {df.shape}')
    return df

# Run ETL pipeline
try:
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
    
    # Load (train models using file path)
    print('\\nTraining models with MLflow tracking...')
    best_model, best_accuracy = train_models(processed_data_path, config)
    
    print(f'\\n✅ Training complete!')
    print(f'✅ Best model: {best_model} with accuracy: {best_accuracy:.4f}')
    print('✅ Results tracked in MLflow')
    
except Exception as e:
    print(f'Error in ETL pipeline: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
  
  if [ $? -eq 0 ]; then
    echo "✅ Model training completed successfully"
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
  
  echo "✅ All services stopped"
}

# Main script logic based on action
case "$ACTION" in
  "run-all")
    echo "========================================================"
    echo "🚀 Running simplified Mushroom Classification MLOps pipeline"
    echo "========================================================"
    start_mlflow
    train_models
    echo ""
    echo "🎉 Everything is ready! Access MLflow UI: http://localhost:$MLFLOW_PORT"
    ;;
  
  "start-mlflow")
    echo "========================================================"
    echo "🚀 Starting MLflow server only"
    echo "========================================================"
    start_mlflow
    ;;
  
  "train")
    echo "========================================================"
    echo "🚀 Running model training only"
    echo "========================================================"
    train_models
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
  
  *)
    echo "❌ Unknown action: $ACTION"
    show_usage
    exit 1
    ;;
esac

echo "========================================================"
