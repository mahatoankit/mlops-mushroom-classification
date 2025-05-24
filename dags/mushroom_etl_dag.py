"""
Airflow DAG for the mushroom classification ETL pipeline.
This DAG orchestrates the entire process of data extraction, transformation, modeling, and evaluation.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

import sys

# Add the project directory to the Python path
sys.path.insert(
    0, "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom"
)

# Import pipeline components
from src.extract import extract_data
from src.transform import transform_data
from src.load import load_data, save_model
from src.train import (
    train_models,
    evaluate_model,
    plot_roc_curves,
    plot_feature_importance,
    compare_models,
)

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["your_email@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define paths and configuration
CONFIG_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/config/config.yaml"
RAW_DATA_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/raw/secondary_data.csv"
PROCESSED_DATA_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/processed"
MODELS_PATH = (
    "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/models"
)
METRICS_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/models/metrics"
TEMP_PATH = (
    "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/temp"
)

# Ensure directories exist
for path in [PROCESSED_DATA_PATH, MODELS_PATH, METRICS_PATH, TEMP_PATH]:
    os.makedirs(path, exist_ok=True)


# Task functions
def task_extract_data(**kwargs):
    """Extract data from source files"""
    try:
        df = extract_data(RAW_DATA_PATH)
        # Save DataFrame to disk instead of using XCom
        temp_file = os.path.join(TEMP_PATH, "extracted_data.parquet")
        df.to_parquet(temp_file, index=False)
        return {"file_path": temp_file}
    except Exception as e:
        import traceback

        error_msg = f"Error in extraction task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_transform_data(**kwargs):
    """Transform the raw data"""
    try:
        ti = kwargs["ti"]
        extract_result = ti.xcom_pull(task_ids="extract_data")

        # Load DataFrame from disk
        temp_file = extract_result["file_path"]
        df = pd.read_parquet(temp_file)

        # Transform data
        df_transformed = transform_data(df)

        # Save transformed data to disk
        temp_file = os.path.join(TEMP_PATH, "transformed_data.parquet")
        df_transformed.to_parquet(temp_file, index=False)
        return {"file_path": temp_file}
    except Exception as e:
        import traceback

        error_msg = f"Error in transformation task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_load_data(**kwargs):
    """Split and save the processed data"""
    try:
        ti = kwargs["ti"]
        transform_result = ti.xcom_pull(task_ids="transform_data")

        # Load DataFrame from disk
        temp_file = transform_result["file_path"]
        df_transformed = pd.read_parquet(temp_file)

        # Split and save data
        X_train, X_test, y_train, y_test = load_data(
            df_transformed, PROCESSED_DATA_PATH, test_size=0.3, random_state=42
        )

        # Save split data to disk
        X_train.to_parquet(
            os.path.join(PROCESSED_DATA_PATH, "X_train.parquet"), index=False
        )
        X_test.to_parquet(
            os.path.join(PROCESSED_DATA_PATH, "X_test.parquet"), index=False
        )
        y_train.to_frame().to_parquet(
            os.path.join(PROCESSED_DATA_PATH, "y_train.parquet"), index=False
        )
        y_test.to_frame().to_parquet(
            os.path.join(PROCESSED_DATA_PATH, "y_test.parquet"), index=False
        )

        # Return file paths
        return {
            "X_train_path": os.path.join(PROCESSED_DATA_PATH, "X_train.parquet"),
            "X_test_path": os.path.join(PROCESSED_DATA_PATH, "X_test.parquet"),
            "y_train_path": os.path.join(PROCESSED_DATA_PATH, "y_train.parquet"),
            "y_test_path": os.path.join(PROCESSED_DATA_PATH, "y_test.parquet"),
        }
    except Exception as e:
        import traceback

        error_msg = f"Error in data loading task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_train_models(**kwargs):
    """Train models on the processed data"""
    try:
        ti = kwargs["ti"]
        data_paths = ti.xcom_pull(task_ids="load_data")

        # Load data from disk
        X_train = pd.read_parquet(data_paths["X_train_path"])
        y_train = pd.read_parquet(data_paths["y_train_path"]).iloc[:, 0]

        # Train models
        models = train_models(X_train, y_train)

        # Save models to disk
        model_paths = {}
        for name, model in models.items():
            save_model(model, name, MODELS_PATH)
            model_paths[name] = os.path.join(MODELS_PATH, f"{name}.joblib")

        # Return model and data paths
        return {"model_paths": model_paths, **data_paths}  # Forward the data paths
    except Exception as e:
        import traceback

        error_msg = f"Error in model training task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_evaluate_models(**kwargs):
    """Evaluate the trained models"""
    try:
        ti = kwargs["ti"]
        result = ti.xcom_pull(task_ids="train_models")

        # Load data from disk
        X_train = pd.read_parquet(result["X_train_path"])
        X_test = pd.read_parquet(result["X_test_path"])
        y_train = pd.read_parquet(result["y_train_path"]).iloc[:, 0]
        y_test = pd.read_parquet(result["y_test_path"]).iloc[:, 0]

        # Load models from disk
        import joblib

        models = {}
        for name, path in result["model_paths"].items():
            models[name] = joblib.load(path)

        # Evaluate each model
        model_metrics = {}
        for name, model in models.items():
            metrics = evaluate_model(
                name, model, X_train, y_train, X_test, y_test, output_path=METRICS_PATH
            )
            model_metrics[name] = metrics

        # Save metrics to disk
        metrics_path = os.path.join(METRICS_PATH, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    k: (
                        {
                            mk: float(mv) if isinstance(mv, (int, float)) else str(mv)
                            for mk, mv in v.items()
                        }
                        if isinstance(v, dict)
                        else str(v)
                    )
                    for k, v in model_metrics.items()
                },
                f,
            )

        return {
            "metrics_path": metrics_path,
            "model_paths": result["model_paths"],
            **result,  # Forward all results
        }
    except Exception as e:
        import traceback

        error_msg = (
            f"Error in model evaluation task: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_msg)
        raise


def task_create_visualizations(**kwargs):
    """Create visualizations for model evaluation"""
    try:
        ti = kwargs["ti"]
        result = ti.xcom_pull(task_ids="evaluate_models")

        # Load data from disk
        X_train = pd.read_parquet(result["X_train_path"])
        X_test = pd.read_parquet(result["X_test_path"])
        y_test = pd.read_parquet(result["y_test_path"]).iloc[:, 0]

        # Load models from disk
        import joblib

        models = {}
        for name, path in result["model_paths"].items():
            models[name] = joblib.load(path)

        # Load metrics from disk
        with open(result["metrics_path"], "r") as f:
            metrics = json.load(f)

        # Create visualizations
        plot_roc_curves(models, X_test, y_test, output_path=METRICS_PATH)
        plot_feature_importance(models["xgboost"], X_train, output_path=METRICS_PATH)
        compare_models(metrics, output_path=METRICS_PATH)

        return {"status": "Visualizations created successfully!"}
    except Exception as e:
        import traceback

        error_msg = f"Error in visualization task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


# Define the DAG
with DAG(
    "mushroom_etl_pipeline",
    default_args=default_args,
    description="ETL pipeline for mushroom classification",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mushroom", "classification", "etl", "xgboost"],
) as dag:

    # Define the tasks
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=task_extract_data,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=task_transform_data,
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data,
    )

    train_task = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=task_evaluate_models,
    )

    visualize_task = PythonOperator(
        task_id="create_visualizations",
        python_callable=task_create_visualizations,
    )

    # Define the task dependencies
    (
        extract_task
        >> transform_task
        >> load_task
        >> train_task
        >> evaluate_task
        >> visualize_task
    )
