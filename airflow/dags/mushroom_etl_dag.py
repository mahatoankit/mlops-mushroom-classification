"""
Airflow DAG for the mushroom classification ETL pipeline.
Updated to use ColumnStore training from train.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import warnings
import os
import sys
import logging

warnings.filterwarnings("ignore")

# Environment configuration
PROJECT_ROOT = "/app"
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Import modules
try:
    from src.extract import extract_data
    from src.transform import transform_data
    from src.load import load_data
    from src.train import train_models_from_columnstore  # Updated import

    MODULES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported all required modules")

except ImportError as e:
    MODULES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import modules: {e}")


def task_extract(**context):
    """Extract task - calls dedicated extract module"""
    try:
        logger.info("Starting data extraction")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Extract data
        data = extract_data()
        logger.info(
            f"Extracted data shape: {data.shape if hasattr(data, 'shape') else 'Unknown'}"
        )

        return {"status": "success", "message": "Data extraction completed"}

    except Exception as e:
        logger.error(f"Extract task failed: {e}")
        raise


def task_transform(**context):
    """Transform task - calls dedicated transform module"""
    try:
        logger.info("Starting data transformation")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get data from previous task
        ti = context["ti"]
        extract_result = ti.xcom_pull(task_ids="extract")

        if not extract_result or extract_result.get("status") != "success":
            raise ValueError("Extract task did not complete successfully")

        # Transform data
        transformed_data = transform_data()
        logger.info("Data transformation completed")

        return {"status": "success", "message": "Data transformation completed"}

    except Exception as e:
        logger.error(f"Transform task failed: {e}")
        raise


def task_load(**context):
    """Load task - calls dedicated load module and returns experiment_id"""
    try:
        logger.info("Starting data loading to ColumnStore")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get data from previous task
        ti = context["ti"]
        transform_result = ti.xcom_pull(task_ids="transform")

        if not transform_result or transform_result.get("status") != "success":
            raise ValueError("Transform task did not complete successfully")

        # Load data and get experiment ID
        experiment_id = load_data()  # This should return the experiment_id

        if not experiment_id:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Data loaded to ColumnStore with experiment_id: {experiment_id}")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "message": "Data loading completed",
        }

    except Exception as e:
        logger.error(f"Load task failed: {e}")
        raise


def task_train(**context):
    """Train task - calls updated train.py with ColumnStore data loading"""
    try:
        logger.info("Starting model training with ColumnStore data")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get experiment_id from previous task
        ti = context["ti"]
        load_result = ti.xcom_pull(task_ids="load")

        if not load_result or load_result.get("status") != "success":
            raise ValueError("Load task did not complete successfully")

        experiment_id = load_result.get("experiment_id")
        if not experiment_id:
            raise ValueError("No experiment_id found from load task")

        logger.info(f"Training models for experiment_id: {experiment_id}")

        # Training configuration
        config = {
            "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }

        # Call the updated training function
        results = train_models_from_columnstore(experiment_id, config)

        logger.info(f"Training completed successfully")
        logger.info(f"Best model: {results.get('best_model')}")
        logger.info(f"Best accuracy: {results.get('best_accuracy', 0):.4f}")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "best_model": results.get("best_model"),
            "best_accuracy": results.get("best_accuracy", 0),
            "results": results,
            "message": "Model training completed with ColumnStore data",
        }

    except Exception as e:
        logger.error(f"Train task failed: {e}")
        raise


def task_evaluate(**context):
    """Evaluate task - evaluates trained models"""
    try:
        logger.info("Starting model evaluation")

        # Get results from training task
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train")

        if not train_result or train_result.get("status") != "success":
            raise ValueError("Train task did not complete successfully")

        experiment_id = train_result.get("experiment_id")
        best_model = train_result.get("best_model")
        best_accuracy = train_result.get("best_accuracy", 0)

        logger.info(f"Evaluation for experiment {experiment_id}")
        logger.info(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")

        # Here you could add more detailed evaluation logic
        evaluation_results = {
            "experiment_id": experiment_id,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "evaluation_passed": best_accuracy > 0.8,  # Example threshold
            "recommendation": "Deploy" if best_accuracy > 0.9 else "Review",
        }

        return {
            "status": "success",
            "evaluation_results": evaluation_results,
            "message": "Model evaluation completed",
        }

    except Exception as e:
        logger.error(f"Evaluate task failed: {e}")
        raise


def task_visualize(**context):
    """Visualize task - creates visualizations and reports"""
    try:
        logger.info("Starting visualization generation")

        # Get results from evaluation task
        ti = context["ti"]
        eval_result = ti.xcom_pull(task_ids="evaluate")

        if not eval_result or eval_result.get("status") != "success":
            raise ValueError("Evaluate task did not complete successfully")

        evaluation_results = eval_result.get("evaluation_results", {})

        logger.info("Creating visualization artifacts")
        logger.info(f"Model performance: {evaluation_results}")

        # Here you could add visualization generation logic
        # For now, just log the results

        return {
            "status": "success",
            "visualizations_created": True,
            "message": "Visualization generation completed",
        }

    except Exception as e:
        logger.error(f"Visualize task failed: {e}")
        raise


# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "mushroom_etl_columnstore_pipeline",
    default_args=default_args,
    description="Mushroom classification pipeline using ColumnStore data with train.py",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mushroom", "classification", "columnstore", "xgboost"],
) as dag:

    # Task definitions
    extract = PythonOperator(
        task_id="extract",
        python_callable=task_extract,
        doc_md="Extract mushroom data from source",
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_transform,
        doc_md="Transform and clean mushroom data",
    )

    load = PythonOperator(
        task_id="load",
        python_callable=task_load,
        doc_md="Load processed data into ColumnStore database",
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Train XGBoost model using ColumnStore data via train.py",
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
        doc_md="Evaluate trained model performance",
    )

    visualize = PythonOperator(
        task_id="visualize",
        python_callable=task_visualize,
        doc_md="Generate visualizations and reports",
    )

    # Task dependencies
    extract >> transform >> load >> train >> evaluate >> visualize
