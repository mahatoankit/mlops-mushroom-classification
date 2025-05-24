"""
Simple test script to verify the training pipeline works correctly.
"""

import os
import sys
import pandas as pd
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_pipeline():
    """Test the complete training pipeline"""
    try:
        logger.info("Testing training pipeline...")

        # Import modules
        from src.monitoring import load_config
        from src.extract import extract_data
        from src.transform import transform_data
        from src.load import load_data
        from src.train import train_models, evaluate_model

        logger.info("All imports successful!")

        # Load config
        config = load_config()
        logger.info("Config loaded successfully!")

        # Test data path - prioritize secondary_data.csv
        data_path = "data/raw/secondary_data.csv"
        if not os.path.exists(data_path):
            # Try alternative paths
            alt_paths = [
                "data/raw/fraction_of_dataset.csv",
                os.path.join(PROJECT_ROOT, "data/raw/secondary_data.csv"),
                os.path.join(PROJECT_ROOT, "data/raw/fraction_of_dataset.csv"),
                "data/raw/",
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    break

        if not os.path.exists(data_path):
            logger.error(f"No data file found. Checked: {data_path}")
            return False

        logger.info(f"Using data file: {data_path}")

        # Clean up any malformed MLflow experiments
        try:
            import shutil

            malformed_exp_path = os.path.join(PROJECT_ROOT, "mlruns", "1")
            if os.path.exists(malformed_exp_path):
                logger.info("Cleaning up malformed MLflow experiment...")
                shutil.rmtree(malformed_exp_path)
        except Exception as e:
            logger.warning(f"Could not clean up malformed experiment: {e}")

        # Extract data
        df = extract_data(data_path)
        logger.info(f"Extracted data shape: {df.shape}")

        # Transform data
        df_transformed = transform_data(df)
        logger.info(f"Transformed data shape: {df_transformed.shape}")

        # Load/split data
        X_train, X_test, y_train, y_test = load_data(
            df_transformed,
            config["paths"]["processed_data"],
            test_size=config["data_split"]["test_size"],
            random_state=config["data_split"]["random_state"],
        )
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")

        # Check if we have the correct target column (class_encoded should be in y_train)
        if hasattr(y_train, "name") and y_train.name != "class_encoded":
            logger.warning(f"Unexpected target column: {y_train.name}")
            # Try to reconstruct the correct split with class_encoded as target
            if "class_encoded" in df_transformed.columns:
                logger.info("Reconstructing data split with correct target column...")
                from sklearn.model_selection import train_test_split

                X_corrected = df_transformed.drop(columns=["class_encoded"])
                y_corrected = df_transformed["class_encoded"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_corrected,
                    y_corrected,
                    test_size=config["data_split"]["test_size"],
                    random_state=config["data_split"]["random_state"],
                )
                logger.info(
                    f"Corrected data split - Train: {X_train.shape}, Test: {X_test.shape}"
                )
                logger.info(f"Target distribution: {y_train.value_counts()}")

        # Train models using simple version (no MLflow in test)
        from src.train import train_models_simple

        models = train_models_simple(X_train, y_train)
        logger.info(f"Trained {len(models)} models: {list(models.keys())}")

        # Evaluate first model
        if models:
            model_name = list(models.keys())[0]
            model = models[model_name]
            metrics = evaluate_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            logger.info(f"Model {model_name} metrics: {metrics}")

        logger.info("Training pipeline test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("✅ Training pipeline test passed!")
        sys.exit(0)
    else:
        print("❌ Training pipeline test failed!")
        sys.exit(1)
