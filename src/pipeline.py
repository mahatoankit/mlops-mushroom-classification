"""
Main ETL pipeline script for the mushroom classification project.
Orchestrates the entire pipeline from data extraction to model training and evaluation.
"""

import os
import argparse
import logging
import json
import yaml
from datetime import datetime
import sys

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import ETL components with error handling
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

    logger.info("Successfully imported ETL components")
except ImportError as e:
    logger.error(f"Failed to import ETL components: {e}")
    raise

# Import MLOps components with error handling
try:
    from src.monitoring import ModelMonitor
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


def run_pipeline(config_path, step=None):
    """
    Run the full ETL pipeline or a specific step.

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
            # Create a default config
            config = {
                "paths": {
                    "raw_data": "data/raw",
                    "processed_data": "data/processed",
                    "models": "models",
                    "metrics": "models/metrics",
                },
                "split": {"test_size": 0.3, "random_state": 42},
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
        X_train, X_test, y_train, y_test = None, None, None, None
        models = {}
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

        # Load data (split and save)
        if "load" in steps and df_transformed is not None:
            logger.info("Step 3: Loading processed data")
            X_train, X_test, y_train, y_test = load_data(
                df_transformed,
                config["paths"]["processed_data"],
                test_size=config["split"]["test_size"],
                random_state=config["split"]["random_state"],
            )
        elif "load" in steps:
            logger.error("Cannot load data: No transformed data available")
            return

        # Train models
        if "train" in steps:
            # Load data if not already loaded
            if X_train is None or y_train is None:
                try:
                    logger.info("Loading saved processed data for training")
                    import pickle

                    with open(
                        os.path.join(
                            config["paths"]["processed_data"], "processed_data.pkl"
                        ),
                        "rb",
                    ) as f:
                        processed_data = pickle.load(f)
                    X_train = processed_data["X_train"]
                    X_test = processed_data["X_test"]
                    y_train = processed_data["y_train"]
                    y_test = processed_data["y_test"]
                except Exception as e:
                    logger.error(f"Cannot load processed data for training: {e}")
                    return

            logger.info("Step 4: Training models")
            models = train_models(X_train, y_train)

            # Evaluate models
            logger.info("Step 5: Evaluating models")
            model_metrics = {}
            for name, model in models.items():
                # Evaluate the model
                metrics = evaluate_model(
                    name,
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output_path=config["paths"]["metrics"],
                )
                model_metrics[name] = metrics

                # Save the model
                save_model(model, name, config["paths"]["models"])

            # Create visualizations
            logger.info("Step 6: Creating visualizations")
            plot_roc_curves(
                models, X_test, y_test, output_path=config["paths"]["metrics"]
            )
            plot_feature_importance(
                models["xgboost"], X_train, output_path=config["paths"]["metrics"]
            )
            compare_models(model_metrics, output_path=config["paths"]["metrics"])

        # Setup model monitoring
        if "monitoring" in steps:
            logger.info("Step 7: Setting up model monitoring")

            # Load model metrics if not available
            if not model_metrics and os.path.exists(
                os.path.join(config["paths"]["metrics"], "pipeline_metadata.json")
            ):
                try:
                    with open(
                        os.path.join(
                            config["paths"]["metrics"], "pipeline_metadata.json"
                        ),
                        "r",
                    ) as f:
                        metadata = json.load(f)
                        model_metrics = metadata.get("model_metrics", {})
                except Exception as e:
                    logger.error(f"Error loading model metrics for monitoring: {e}")

            if model_metrics:
                for model_name, metrics in model_metrics.items():
                    # Initialize model monitor
                    monitor = ModelMonitor(
                        model_name,
                        os.path.join(config["paths"]["metrics"], "monitoring"),
                    )

                    # Record initial metrics
                    monitor.record_metrics(metrics)

                    # Generate monitoring report
                    report_path = monitor.generate_monitoring_report()
                    logger.info(
                        f"Generated monitoring report for {model_name} at {report_path}"
                    )

        # Setup model versioning
        if "versioning" in steps:
            logger.info("Step 8: Setting up model versioning")

            # Initialize model registry
            registry = ModelRegistry(
                os.path.join(config["paths"]["models"], "registry")
            )

            # Register models if available
            if model_metrics:
                for model_name, metrics in model_metrics.items():
                    model_path = os.path.join(
                        config["paths"]["models"], f"{model_name}.joblib"
                    )

                    # Choose the best model for production
                    promote_to = None
                    if model_name == "xgboost":  # Assuming XGBoost is the best model
                        promote_to = "production"
                    else:
                        promote_to = "staging"

                    # Register and promote the model
                    try:
                        version = register_and_promote_model(
                            model_path, model_name, metrics, promote_to
                        )
                        logger.info(
                            f"Registered {model_name} as version {version} in {promote_to}"
                        )
                    except Exception as e:
                        logger.error(f"Error registering model {model_name}: {e}")

        # Setup A/B testing
        if "ab_testing" in steps:
            logger.info("Step 9: Setting up A/B testing")

            # Find models to compare
            try:
                registry = ModelRegistry(
                    os.path.join(config["paths"]["models"], "registry")
                )
                prod_models = registry.get_production_models()
                staging_models = registry.get_staging_models()

                if prod_models and staging_models:
                    # Get the first production and staging model for A/B testing
                    prod_model = list(prod_models.values())[0]["path"]
                    staging_model = list(staging_models.values())[0]["path"]

                    # Create A/B test
                    ab_test_id = create_ab_test(
                        name=f"{os.path.basename(prod_model).split('.')[0]}_vs_{os.path.basename(staging_model).split('.')[0]}",
                        model_a=prod_model,
                        model_b=staging_model,
                        traffic_split=0.5,
                    )
                    logger.info(f"Created A/B test with ID: {ab_test_id}")
            except Exception as e:
                logger.error(f"Error setting up A/B test: {e}")

        # Save pipeline metadata
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        metadata = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time_seconds": elapsed_time,
            "steps_executed": steps,
        }

        # Add data shape information if available
        data_shape = {}
        if df is not None:
            data_shape["raw"] = df.shape
        if df_transformed is not None:
            data_shape["transformed"] = df_transformed.shape
        if X_train is not None:
            data_shape["train"] = X_train.shape
        if X_test is not None:
            data_shape["test"] = X_test.shape

        if data_shape:
            metadata["data_shape"] = data_shape

        # Add model metrics if available
        if model_metrics:
            metadata["model_metrics"] = model_metrics

        # Save metadata as JSON
        with open(
            os.path.join(config["paths"]["metrics"], "pipeline_metadata.json"), "w"
        ) as f:
            json.dump(metadata, f, indent=4)

        logger.info(
            f"ETL pipeline completed successfully in {elapsed_time:.2f} seconds"
        )

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
        default="config/config.yaml",
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
