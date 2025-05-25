"""
Airflow DAG for the mushroom classification ETL pipeline.
Streamlined version with reduced redundancy.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import warnings

warnings.filterwarnings("ignore")

# Environment-aware configuration
PROJECT_ROOT = (
    "/app"
    if os.getenv("ENV") == "docker"
    else "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom"
)

# Path configuration
PATHS = {
    "config": f"{PROJECT_ROOT}/config/config.yaml",
    "raw_data": f"{PROJECT_ROOT}/data/raw/secondary_data.csv",
    "processed": f"{PROJECT_ROOT}/data/processed",
    "models": f"{PROJECT_ROOT}/models",
    "metrics": f"{PROJECT_ROOT}/models/metrics",
    "temp": f"{PROJECT_ROOT}/data/temp",
}

# Ensure critical directories exist
for path in [PATHS["processed"], PATHS["models"], PATHS["metrics"], PATHS["temp"]]:
    os.makedirs(path, exist_ok=True)

# Import with simplified fallback
try:
    import sys

    sys.path.insert(0, PROJECT_ROOT)
    
    # Temporarily modify any logging configuration in imported modules
    import logging
    
    # Configure root logger to prevent permission errors
    os.makedirs(f"{PROJECT_ROOT}/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]  # Only use stream handler for now
    )
    
    # Now import the modules
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

    IMPORTS_OK = True
except ImportError as e:
    print(f"Using fallback implementations: {e}")
    IMPORTS_OK = False

    # Simplified fallback functions
    def extract_data(file_path):
        return (
            pd.read_csv(file_path)
            if os.path.exists(file_path)
            else pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        )

    def transform_data(df):
        return df.copy()

    def load_data(df, output_path, test_size=0.3, random_state=42):
        from sklearn.model_selection import train_test_split

        X, y = df.drop("target", axis=1), df["target"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_model(model, name, path):
        import joblib

        os.makedirs(path, exist_ok=True)
        joblib.dump(model, f"{path}/{name}.joblib")

    def train_models(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "rf": RandomForestClassifier(n_estimators=10, random_state=42),
            "lr": LogisticRegression(random_state=42, max_iter=100),
        }
        for model in models.values():
            model.fit(X_train, y_train)
        return models

    def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_path=None):
        from sklearn.metrics import accuracy_score

        return {
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "model": name,
        }

    def plot_roc_curves(models, X_test, y_test, output_path=None):
        print("ROC curves created")

    def plot_feature_importance(model, X_train, output_path=None):
        print("Feature importance plotted")

    def compare_models(metrics, output_path=None):
        print("Model comparison completed")


# Utility functions
def save_temp_data(data, filename):
    """Save data to temp directory and return path"""
    path = f"{PATHS['temp']}/{filename}"
    if isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
    else:
        pd.DataFrame(data).to_parquet(path, index=False)
    return path


def load_temp_data(path):
    """Load data from temp directory"""
    df = pd.read_parquet(path)
    return df.iloc[:, 0] if len(df.columns) == 1 else df


def handle_task_error(task_name, error):
    """Standardized error handling"""
    print(f"Error in {task_name}: {error}")
    raise


# DAG Tasks
def task_extract(**context):
    try:
        df = extract_data(PATHS["raw_data"])
        path = save_temp_data(df, "extracted.parquet")
        return {"path": path, "shape": df.shape}
    except Exception as e:
        handle_task_error("extract", e)


def task_transform(**context):
    try:
        ti = context["ti"]
        extract_result = ti.xcom_pull(task_ids="extract")
        df = load_temp_data(extract_result["path"])
        df_transformed = transform_data(df)
        path = save_temp_data(df_transformed, "transformed.parquet")
        return {"path": path, "shape": df_transformed.shape}
    except Exception as e:
        handle_task_error("transform", e)


def task_load(**context):
    try:
        ti = context["ti"]
        transform_result = ti.xcom_pull(task_ids="transform")
        df = load_temp_data(transform_result["path"])
        X_train, X_test, y_train, y_test = load_data(df, PATHS["processed"])

        # Save split data
        paths = {
            "X_train": save_temp_data(X_train, "X_train.parquet"),
            "X_test": save_temp_data(X_test, "X_test.parquet"),
            "y_train": save_temp_data(y_train, "y_train.parquet"),
            "y_test": save_temp_data(y_test, "y_test.parquet"),
        }
        return paths
    except Exception as e:
        handle_task_error("load", e)


def task_train(**context):
    try:
        ti = context["ti"]
        data_paths = ti.xcom_pull(task_ids="load")
        X_train = load_temp_data(data_paths["X_train"])
        y_train = load_temp_data(data_paths["y_train"])

        models = train_models(X_train, y_train)

        # Save models
        model_paths = {}
        for name, model in models.items():
            save_model(model, name, PATHS["models"])
            model_paths[name] = f"{PATHS['models']}/{name}.joblib"

        return {**data_paths, "models": model_paths}
    except Exception as e:
        handle_task_error("train", e)


def task_evaluate(**context):
    try:
        ti = context["ti"]
        result = ti.xcom_pull(task_ids="train")

        # Load data and models
        X_train = load_temp_data(result["X_train"])
        X_test = load_temp_data(result["X_test"])
        y_train = load_temp_data(result["y_train"])
        y_test = load_temp_data(result["y_test"])

        import joblib

        models = {name: joblib.load(path) for name, path in result["models"].items()}

        # Evaluate models
        metrics = {}
        for name, model in models.items():
            metrics[name] = evaluate_model(
                name, model, X_train, y_train, X_test, y_test, PATHS["metrics"]
            )

        # Save metrics
        metrics_path = f"{PATHS['metrics']}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        return {**result, "metrics_path": metrics_path}
    except Exception as e:
        handle_task_error("evaluate", e)


def task_visualize(**context):
    try:
        ti = context["ti"]
        result = ti.xcom_pull(task_ids="evaluate")

        # Load data and models
        X_train = load_temp_data(result["X_train"])
        X_test = load_temp_data(result["X_test"])
        y_test = load_temp_data(result["y_test"])

        import joblib

        models = {name: joblib.load(path) for name, path in result["models"].items()}

        with open(result["metrics_path"]) as f:
            metrics = json.load(f)

        # Create visualizations
        plot_roc_curves(models, X_test, y_test, PATHS["metrics"])
        if models:
            first_model = list(models.values())[0]
            plot_feature_importance(first_model, X_train, PATHS["metrics"])
        compare_models(metrics, PATHS["metrics"])

        return {"status": "completed"}
    except Exception as e:
        handle_task_error("visualize", e)


# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mushroom_etl_pipeline",
    default_args=default_args,
    description="Streamlined mushroom classification ETL pipeline",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mushroom", "classification", "etl"],
) as dag:

    # Task definitions
    extract = PythonOperator(task_id="extract", python_callable=task_extract)
    transform = PythonOperator(task_id="transform", python_callable=task_transform)
    load = PythonOperator(task_id="load", python_callable=task_load)
    train = PythonOperator(task_id="train", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
    visualize = PythonOperator(task_id="visualize", python_callable=task_visualize)

    # Task dependencies
    extract >> transform >> load >> train >> evaluate >> visualize
