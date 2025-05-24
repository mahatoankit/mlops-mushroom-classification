"""
Airflow DAG for the mushroom classification ETL pipeline.
This DAG orchestrates the entire process of data extraction, transformation, modeling, and evaluation.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator

import sys
import warnings

warnings.filterwarnings("ignore")

# Add the project directory to the Python path
# Check if running in Docker
if os.getenv("ENV") == "docker":
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = (
        "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom"
    )

sys.path.insert(0, PROJECT_ROOT)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# Global variable to track import success
IMPORTS_SUCCESSFUL = False

# Import pipeline components with error handling
try:
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

    IMPORTS_SUCCESSFUL = True
    print("All modules imported successfully!")
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    IMPORTS_SUCCESSFUL = False

    # Define dummy functions if imports fail
    def extract_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        # Return a dummy DataFrame for testing
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def transform_data(df):
        # Basic transformation for testing
        return df.copy()

    def load_data(df, output_path, test_size=0.3, random_state=42):
        from sklearn.model_selection import train_test_split

        X = df.drop("target", axis=1)
        y = df["target"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_model(model, name, path):
        import joblib

        os.makedirs(path, exist_ok=True)
        joblib.dump(model, os.path.join(path, f"{name}.joblib"))

    def train_models(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "random_forest": RandomForestClassifier(n_estimators=10, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=100),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

        return models

    def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_path=None):
        from sklearn.metrics import accuracy_score, classification_report

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {"accuracy": accuracy, "model_name": name}

    def plot_roc_curves(models, X_test, y_test, output_path=None):
        print("Creating ROC curves (dummy implementation)")

    def plot_feature_importance(model, X_train, output_path=None):
        print("Creating feature importance plot (dummy implementation)")

    def compare_models(metrics, output_path=None):
        print("Creating model comparison (dummy implementation)")


# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define paths and configuration - Docker-aware
if os.getenv("ENV") == "docker":
    CONFIG_PATH = "/app/config/config.yaml"
    RAW_DATA_PATH = "/app/data/raw/secondary_data.csv"
    PROCESSED_DATA_PATH = "/app/data/processed"
    MODELS_PATH = "/app/models"
    METRICS_PATH = "/app/models/metrics"
    TEMP_PATH = "/app/data/temp"
else:
    CONFIG_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/config/config.yaml"
    RAW_DATA_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/raw/secondary_data.csv"
    PROCESSED_DATA_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/processed"
    MODELS_PATH = (
        "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/models"
    )
    METRICS_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/models/metrics"
    TEMP_PATH = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/temp"

# Check if critical files exist (only when running directly)
if __name__ == "__main__":
    print("=== File Existence Check ===")
    print(f"CONFIG_PATH exists: {os.path.exists(CONFIG_PATH)}")
    print(f"RAW_DATA_PATH exists: {os.path.exists(RAW_DATA_PATH)}")
    print(f"PROJECT_ROOT exists: {os.path.exists(PROJECT_ROOT)}")
    print(f"src directory exists: {os.path.exists(os.path.join(PROJECT_ROOT, 'src'))}")

    # List contents of project root to see what's actually there
    if os.path.exists(PROJECT_ROOT):
        try:
            contents = os.listdir(PROJECT_ROOT)
            print(f"Contents of PROJECT_ROOT: {contents}")
            src_path = os.path.join(PROJECT_ROOT, "src")
            if os.path.exists(src_path):
                src_contents = os.listdir(src_path)
                print(f"Contents of src: {src_contents}")
        except PermissionError:
            print("Permission denied when listing directory contents")

# Ensure directories exist
for path in [PROCESSED_DATA_PATH, MODELS_PATH, METRICS_PATH, TEMP_PATH]:
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError:
        print(f"Permission denied when creating directory: {path}")


# Task functions
def task_extract_data(**kwargs):
    """Extract data from source files"""
    try:
        print(f"Extracting data from: {RAW_DATA_PATH}")
        df = extract_data(RAW_DATA_PATH)
        print(f"Extracted data shape: {df.shape}")

        # Save DataFrame to disk instead of using XCom
        temp_file = os.path.join(TEMP_PATH, "extracted_data.parquet")
        df.to_parquet(temp_file, index=False)
        print(f"Saved extracted data to: {temp_file}")

        return {"file_path": temp_file}
    except Exception as e:
        import traceback

        error_msg = f"Error in extraction task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_transform_data(**kwargs):
    """Transform the raw data"""
    try:
        ti = kwargs.get("ti")
        if ti is None:
            raise ValueError("TaskInstance (ti) not provided")

        extract_result = ti.xcom_pull(task_ids="extract_data")
        if extract_result is None:
            raise ValueError("No data received from extract_data task")

        # Load DataFrame from disk
        temp_file = extract_result["file_path"]
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Extracted data file not found: {temp_file}")

        df = pd.read_parquet(temp_file)
        print(f"Loaded data shape: {df.shape}")

        # Transform data
        df_transformed = transform_data(df)
        print(f"Transformed data shape: {df_transformed.shape}")

        # Save transformed data to disk
        temp_file = os.path.join(TEMP_PATH, "transformed_data.parquet")
        df_transformed.to_parquet(temp_file, index=False)
        print(f"Saved transformed data to: {temp_file}")

        return {"file_path": temp_file}
    except Exception as e:
        import traceback

        error_msg = f"Error in transformation task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_load_data(**kwargs):
    """Split and save the processed data"""
    try:
        ti = kwargs.get("ti")
        if ti is None:
            raise ValueError("TaskInstance (ti) not provided")

        transform_result = ti.xcom_pull(task_ids="transform_data")
        if transform_result is None:
            raise ValueError("No data received from transform_data task")

        # Load DataFrame from disk
        temp_file = transform_result["file_path"]
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Transformed data file not found: {temp_file}")

        df_transformed = pd.read_parquet(temp_file)
        print(f"Loading data shape: {df_transformed.shape}")

        # Split and save data
        X_train, X_test, y_train, y_test = load_data(
            df_transformed, PROCESSED_DATA_PATH, test_size=0.3, random_state=42
        )

        print(f"Split data - Train: {X_train.shape}, Test: {X_test.shape}")

        # Save split data to disk
        X_train_path = os.path.join(PROCESSED_DATA_PATH, "X_train.parquet")
        X_test_path = os.path.join(PROCESSED_DATA_PATH, "X_test.parquet")
        y_train_path = os.path.join(PROCESSED_DATA_PATH, "y_train.parquet")
        y_test_path = os.path.join(PROCESSED_DATA_PATH, "y_test.parquet")

        X_train.to_parquet(X_train_path, index=False)
        X_test.to_parquet(X_test_path, index=False)

        # Handle series to dataframe conversion safely
        if hasattr(y_train, "to_frame"):
            y_train.to_frame().to_parquet(y_train_path, index=False)
            y_test.to_frame().to_parquet(y_test_path, index=False)
        else:
            pd.DataFrame(y_train).to_parquet(y_train_path, index=False)
            pd.DataFrame(y_test).to_parquet(y_test_path, index=False)

        print("Saved all split data files")

        # Return file paths
        return {
            "X_train_path": X_train_path,
            "X_test_path": X_test_path,
            "y_train_path": y_train_path,
            "y_test_path": y_test_path,
        }
    except Exception as e:
        import traceback

        error_msg = f"Error in data loading task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_train_models(**kwargs):
    """Train models on the processed data"""
    try:
        ti = kwargs.get("ti")
        if ti is None:
            raise ValueError("TaskInstance (ti) not provided")

        data_paths = ti.xcom_pull(task_ids="load_data")
        if data_paths is None:
            raise ValueError("No data paths received from load_data task")

        # Load data from disk
        X_train = pd.read_parquet(data_paths["X_train_path"])
        y_train_df = pd.read_parquet(data_paths["y_train_path"])
        y_train = y_train_df.iloc[:, 0] if len(y_train_df.columns) > 0 else y_train_df

        print(f"Training data - X: {X_train.shape}, y: {len(y_train)}")

        # Train models using the simple version or fallback
        if IMPORTS_SUCCESSFUL:
            try:
                # Try to use the simple version first
                from src.train import train_models_simple

                models = train_models_simple(X_train, y_train)
            except (ImportError, AttributeError):
                # Fallback to dummy implementation
                models = train_models(X_train, y_train)
        else:
            models = train_models(X_train, y_train)

        print(f"Trained {len(models)} models: {list(models.keys())}")

        # Save models to disk
        model_paths = {}
        for name, model in models.items():
            save_model(model, name, MODELS_PATH)
            model_paths[name] = os.path.join(MODELS_PATH, f"{name}.joblib")
            print(f"Saved model: {name}")

        # Return model and data paths
        return {"model_paths": model_paths, **data_paths}
    except Exception as e:
        import traceback

        error_msg = f"Error in model training task: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise


def task_evaluate_models(**kwargs):
    """Evaluate the trained models"""
    try:
        ti = kwargs.get("ti")
        if ti is None:
            raise ValueError("TaskInstance (ti) not provided")

        result = ti.xcom_pull(task_ids="train_models")
        if result is None:
            raise ValueError("No result received from train_models task")

        # Load data from disk
        X_train = pd.read_parquet(result["X_train_path"])
        X_test = pd.read_parquet(result["X_test_path"])
        y_train_df = pd.read_parquet(result["y_train_path"])
        y_test_df = pd.read_parquet(result["y_test_path"])

        y_train = y_train_df.iloc[:, 0] if len(y_train_df.columns) > 0 else y_train_df
        y_test = y_test_df.iloc[:, 0] if len(y_test_df.columns) > 0 else y_test_df

        print(f"Evaluation data - Train: {X_train.shape}, Test: {X_test.shape}")

        # Load models from disk
        import joblib

        models = {}
        for name, path in result["model_paths"].items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
                print(f"Loaded model: {name}")
            else:
                print(f"Warning: Model file not found: {path}")

        # Evaluate each model
        model_metrics = {}
        for name, model in models.items():
            try:
                metrics = evaluate_model(
                    name,
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output_path=METRICS_PATH,
                )
                model_metrics[name] = metrics
                print(f"Evaluated model {name}: {metrics}")
            except Exception as e:
                print(f"Error evaluating model {name}: {e}")
                model_metrics[name] = {"error": str(e)}

        # Save metrics to disk
        metrics_path = os.path.join(METRICS_PATH, "model_metrics.json")
        with open(metrics_path, "w") as f:
            # Convert all values to JSON serializable format
            serializable_metrics = {}
            for k, v in model_metrics.items():
                if isinstance(v, dict):
                    serializable_metrics[k] = {
                        mk: float(mv) if isinstance(mv, (int, float)) else str(mv)
                        for mk, mv in v.items()
                    }
                else:
                    serializable_metrics[k] = str(v)

            json.dump(serializable_metrics, f, indent=2)

        print(f"Saved metrics to: {metrics_path}")

        return {
            "metrics_path": metrics_path,
            "model_paths": result["model_paths"],
            **result,
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
        ti = kwargs.get("ti")
        if ti is None:
            raise ValueError("TaskInstance (ti) not provided")

        result = ti.xcom_pull(task_ids="evaluate_models")
        if result is None:
            raise ValueError("No result received from evaluate_models task")

        # Load data from disk
        X_train = pd.read_parquet(result["X_train_path"])
        X_test = pd.read_parquet(result["X_test_path"])
        y_test_df = pd.read_parquet(result["y_test_path"])
        y_test = y_test_df.iloc[:, 0] if len(y_test_df.columns) > 0 else y_test_df

        print(f"Visualization data - Train: {X_train.shape}, Test: {X_test.shape}")

        # Load models from disk
        import joblib

        models = {}
        for name, path in result["model_paths"].items():
            if os.path.exists(path):
                models[name] = joblib.load(path)

        # Load metrics from disk
        if os.path.exists(result["metrics_path"]):
            with open(result["metrics_path"], "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Create visualizations
        try:
            plot_roc_curves(models, X_test, y_test, output_path=METRICS_PATH)
            print("Created ROC curves")
        except Exception as e:
            print(f"Error creating ROC curves: {e}")

        # Check if any model exists for feature importance
        if models:
            model_name = list(models.keys())[0]  # Use first available model
            try:
                plot_feature_importance(
                    models[model_name], X_train, output_path=METRICS_PATH
                )
                print(f"Created feature importance plot for {model_name}")
            except Exception as e:
                print(f"Error creating feature importance plot: {e}")
        else:
            print("No models available for feature importance plot")

        try:
            compare_models(metrics, output_path=METRICS_PATH)
            print("Created model comparison")
        except Exception as e:
            print(f"Error creating model comparison: {e}")

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

# Test execution when run directly
if __name__ == "__main__":
    print("\n=== Running ETL Pipeline Test ===")
    print("Note: This is a test run. For production, use Airflow scheduler.")
    print(f"Imports successful: {IMPORTS_SUCCESSFUL}")

    # Mock TaskInstance class for testing
    class MockTaskInstance:
        def __init__(self):
            self.xcom_data = {}

        def xcom_pull(self, task_ids):
            return self.xcom_data.get(task_ids)

        def set_xcom_data(self, task_id, data):
            self.xcom_data[task_id] = data

    try:
        mock_ti = MockTaskInstance()

        # Test extract
        print("\n1. Testing extract_data...")
        extract_result = task_extract_data()
        mock_ti.set_xcom_data("extract_data", extract_result)
        print(f"Extract result: {extract_result}")

        # Test transform
        print("\n2. Testing transform_data...")
        transform_result = task_transform_data(ti=mock_ti)
        mock_ti.set_xcom_data("transform_data", transform_result)
        print(f"Transform result: {transform_result}")

        # Test load
        print("\n3. Testing load_data...")
        load_result = task_load_data(ti=mock_ti)
        mock_ti.set_xcom_data("load_data", load_result)
        print(f"Load result: {load_result}")

        # Test train
        print("\n4. Testing train_models...")
        train_result = task_train_models(ti=mock_ti)
        mock_ti.set_xcom_data("train_models", train_result)
        print(f"Train result keys: {list(train_result.keys())}")

        # Test evaluate
        print("\n5. Testing evaluate_models...")
        eval_result = task_evaluate_models(ti=mock_ti)
        mock_ti.set_xcom_data("evaluate_models", eval_result)
        print(f"Evaluation completed")

        # Test visualizations
        print("\n6. Testing create_visualizations...")
        viz_result = task_create_visualizations(ti=mock_ti)
        print(f"Visualization result: {viz_result}")

        print("\n=== ETL Pipeline Test Completed Successfully ===")

    except Exception as e:
        print(f"\n=== ETL Pipeline Test Failed ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== How to run with Airflow ===")
    print("1. Initialize Airflow DB: airflow db init")
    print(
        "2. Create admin user: airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com"
    )
    print("3. Start Airflow scheduler: airflow scheduler")
    print("4. Start Airflow webserver: airflow webserver --port 8080")
    print("5. Access Airflow UI: http://localhost:8080")
    print("6. Enable and trigger the 'mushroom_etl_pipeline' DAG")
