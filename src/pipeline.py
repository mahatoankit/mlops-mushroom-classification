"""
Main ETL pipeline script for the mushroom classification project.
Orchestrates the entire pipeline from data extraction to model training and evaluation with ColumnStore.
"""

import os
import argparse
import logging
import json
import yaml
from datetime import datetime
import sys

# Add project root to path
sys.path.append("/app")

# Configure logging
logger = logging.getLogger(__name__)

# Import ETL components with error handling
try:
    from src.extract import extract_data
    from src.transform import transform_data
    from src.load import load_data, save_model
    from src.train import train_models_from_columnstore, train_models
    from config.database import db_manager

    logger.info("Successfully imported ETL components")
except ImportError as e:
    logger.error(f"Failed to import ETL components: {e}")
    raise

# Import MLOps components with error handling
try:
    from src.monitoring import monitor_model_performance
    from src.model_versioning import ModelRegistry, register_and_promote_model
    from src.ab_testing import create_ab_test

    logger.info("Successfully imported MLOps components")
except ImportError as e:
    logger.warning(f"Some MLOps components could not be imported: {e}")

    # Define dummy functions for missing components
    class ModelMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def record_metrics(self, *args, **kwargs):
            pass

        def generate_monitoring_report(self, *args, **kwargs):
            return "monitoring_report.html"

    class ModelRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def get_production_models(self, *args, **kwargs):
            return {}

        def get_staging_models(self, *args, **kwargs):
            return {}

    def register_and_promote_model(*args, **kwargs):
        return "v1.0.0"

    def create_ab_test(*args, **kwargs):
        return "test_123"

    def monitor_model_performance(*args, **kwargs):
        return {"monitoring": "disabled"}


def run_pipeline(config_path, step=None):
    """
    Run the full ETL pipeline or a specific step with ColumnStore integration.

    Args:
        config_path (str): Path to the configuration file.
        step (str, optional): Specific pipeline step to run.
    """
    start_time = datetime.now()
    logger.info(f"Starting ETL pipeline at {start_time}")

    try:
        # Load configuration with better error handling
        if not os.path.exists(config_path):
            logger.warning(
                f"Config file {config_path} not found, creating default config"
            )
            config = {
                "paths": {
                    "raw_data": "/app/data/raw",
                    "processed_data": "/app/data/processed",
                    "models": "/app/models",
                    "metrics": "/app/models/metrics",
                },
                "data_split": {"test_size": 0.3, "random_state": 42},
                "models": {
                    "random_forest": {"n_estimators": 100},
                    "gradient_boosting": {"n_estimators": 100},
                    "xgboost": {"n_estimators": 100, "max_depth": 6},
                },
            }
        else:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")

        # Create output directories
        for path_key, path_value in config["paths"].items():
            os.makedirs(path_value, exist_ok=True)

        os.makedirs(
            os.path.join(config["paths"]["metrics"], "monitoring"), exist_ok=True
        )

        # Initialize pipeline components
        df = None
        df_transformed = None
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_results = {}
        model_metrics = {}

        # Run specified step or all steps
        steps = [
            "extract",
            "transform",
            "load",
            "train",
            "monitoring",
            "versioning",
            "ab_testing",
        ]

        # If step is specified, run only that step and subsequent steps
        if step:
            if step not in steps:
                logger.error(f"Invalid step: {step}. Must be one of {steps}")
                return
            steps = steps[steps.index(step) :]

        # Test database connectivity first
        logger.info("Testing database connectivity...")
        if not db_manager.test_mariadb_connection():
            raise Exception("MariaDB connection failed")
        if not db_manager.test_postgres_connection():
            raise Exception("PostgreSQL connection failed")
        logger.info("Database connectivity verified")

        # Extract data
        if "extract" in steps:
            logger.info("Step 1: Extracting data")
            df = extract_data(config["paths"]["raw_data"])

        # Transform data
        if "transform" in steps and df is not None:
            logger.info("Step 2: Transforming data")
            df_transformed = transform_data(df)
        elif "transform" in steps:
            logger.error("Cannot transform data: No data extracted")
            return

        # Load data into ColumnStore
        if "load" in steps and df_transformed is not None:
            logger.info("Step 3: Loading processed data into ColumnStore")

            # Initialize ColumnStore tables
            db_manager.create_columnstore_tables()

            # Load data using enhanced ETL
            from src.data_processing.enhanced_etl import EnhancedMushroomETL

            etl = EnhancedMushroomETL()
            etl_results = etl.run_etl_pipeline(experiment_id)

            if etl_results["status"] != "success":
                raise Exception(f"ETL pipeline failed: {etl_results.get('error')}")

            logger.info(f"Data loaded successfully for experiment {experiment_id}")

        elif "load" in steps:
            logger.error("Cannot load data: No transformed data available")
            return

        # Train models using ColumnStore data
        if "train" in steps:
            logger.info("Step 4: Training models from ColumnStore")

            try:
                training_results = train_models_from_columnstore(experiment_id, config)

                if training_results:
                    logger.info(f"Model training completed successfully")
                    logger.info(
                        f"Best model: {training_results.get('best_model')} with accuracy: {training_results.get('best_accuracy', 0):.4f}"
                    )

                    # Extract model metrics for monitoring
                    model_metrics = {}
                    for model_name, results in training_results.get(
                        "model_results", {}
                    ).items():
                        model_metrics[model_name] = {
                            "accuracy": results["accuracy"],
                            "precision": results["precision"],
                            "recall": results["recall"],
                            "f1_score": results["f1_score"],
                        }

                    # Save trained models
                    for model_name, pipeline in training_results.get(
                        "trained_models", {}
                    ).items():
                        model_path = save_model(
                            pipeline,
                            model_name,
                            config["paths"]["models"],
                            metrics=model_metrics.get(model_name),
                        )
                        logger.info(f"Saved {model_name} to {model_path}")
                else:
                    raise Exception("No training results returned")

            except Exception as e:
                logger.error(f"Model training failed: {e}")
                raise

        # Setup model monitoring
        if "monitoring" in steps and model_metrics:
            logger.info("Step 5: Setting up model monitoring")

            try:
                for model_name, metrics in model_metrics.items():
                    monitoring_results = monitor_model_performance(
                        None,  # Model object not needed for basic monitoring
                        None,  # X_test
                        None,  # y_test
                        metrics_path=os.path.join(
                            config["paths"]["metrics"], "monitoring", model_name
                        ),
                    )
                    logger.info(f"Monitoring setup completed for {model_name}")
            except Exception as e:
                logger.warning(f"Monitoring setup failed: {e}")

        # Setup model versioning
        if "versioning" in steps and model_metrics:
            logger.info("Step 6: Setting up model versioning")

            try:
                registry = ModelRegistry(
                    os.path.join(config["paths"]["models"], "registry")
                )

                for model_name, metrics in model_metrics.items():
                    model_path = os.path.join(
                        config["paths"]["models"], f"{model_name}.joblib"
                    )

                    if os.path.exists(model_path):
                        promote_to = (
                            "production"
                            if model_name == training_results.get("best_model")
                            else "staging"
                        )

                        try:
                            version = register_and_promote_model(
                                model_path, model_name, metrics, promote_to
                            )
                            logger.info(
                                f"Registered {model_name} as version {version} in {promote_to}"
                            )
                        except Exception as e:
                            logger.error(f"Error registering model {model_name}: {e}")
            except Exception as e:
                logger.warning(f"Model versioning setup failed: {e}")

        # Setup A/B testing
        if "ab_testing" in steps and model_metrics:
            logger.info("Step 7: Setting up A/B testing")

            try:
                if len(model_metrics) >= 2:
                    models = list(model_metrics.keys())
                    model_a = models[0]
                    model_b = models[1]

                    ab_test_id = create_ab_test(
                        name=f"{model_a}_vs_{model_b}_{experiment_id}",
                        model_a=os.path.join(
                            config["paths"]["models"], f"{model_a}.joblib"
                        ),
                        model_b=os.path.join(
                            config["paths"]["models"], f"{model_b}.joblib"
                        ),
                        traffic_split=0.5,
                    )
                    logger.info(f"Created A/B test with ID: {ab_test_id}")
                else:
                    logger.warning("Not enough models for A/B testing")
            except Exception as e:
                logger.warning(f"A/B testing setup failed: {e}")

        # Save pipeline metadata
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        metadata = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time_seconds": elapsed_time,
            "steps_executed": steps,
            "experiment_id": experiment_id,
            "training_results": training_results,
            "model_metrics": model_metrics,
        }

        # Add data shape information if available
        if df is not None:
            metadata["raw_data_shape"] = df.shape
        if df_transformed is not None:
            metadata["transformed_data_shape"] = df_transformed.shape

        # Save metadata as JSON
        metadata_path = os.path.join(
            config["paths"]["metrics"], "pipeline_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        logger.info(
            f"ETL pipeline completed successfully in {elapsed_time:.2f} seconds"
        )
        logger.info(f"Pipeline metadata saved to {metadata_path}")

    except Exception as e:
        logger.error(f"Error in ETL pipeline: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the mushroom classification ETL pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/app/config/config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "extract",
            "transform",
            "load",
            "train",
            "monitoring",
            "versioning",
            "ab_testing",
        ],
        help="Specific pipeline step to run",
    )

    args = parser.parse_args()
    run_pipeline(args.config, args.step)
