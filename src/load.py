"""
Load component of the ETL pipeline for Mushroom Classification.
Responsible for saving processed data and trained models.
"""

import os
import pandas as pd
import pickle
import logging
import joblib
from sklearn.model_selection import train_test_split

# Import model versioning
from src.model_versioning import register_and_promote_model

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/load.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_data(df, output_path, test_size=0.3, random_state=42):
    """
    Split the transformed data into train and test sets and save them.

    Args:
        df (pd.DataFrame): Transformed data.
        output_path (str): Path to save the output files.
        test_size (float): Proportion of data for the test set.
        random_state (int): Seed for the random number generator.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        logger.info("Preparing data for modeling")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Split data into features and target
        X = df.drop(columns=["class_encoded"])
        y = df["class_encoded"]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(
            f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets"
        )

        # Save the processed data
        processed_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
        }

        # Save as pickle file
        with open(os.path.join(output_path, "processed_data.pkl"), "wb") as f:
            pickle.dump(processed_data, f)

        # Also save as CSV for easier access
        X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
        pd.DataFrame(y_train, columns=["class_encoded"]).to_csv(
            os.path.join(output_path, "y_train.csv"), index=False
        )
        pd.DataFrame(y_test, columns=["class_encoded"]).to_csv(
            os.path.join(output_path, "y_test.csv"), index=False
        )

        logger.info(f"Saved processed data to {output_path}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def save_model(model, model_name, output_path, metrics=None, promote=None):
    """
    Save a trained model and register it with the model registry.

    Args:
        model: Trained model object.
        model_name (str): Name of the model.
        output_path (str): Path to save the model.
        metrics (dict, optional): Model performance metrics.
        promote (str, optional): Where to promote the model ('staging' or 'production').

    Returns:
        str: Path to the saved model.
    """
    try:
        logger.info(f"Saving {model_name} model")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save the model
        model_path = os.path.join(output_path, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        # Register and potentially promote the model
        if metrics:
            try:
                version = register_and_promote_model(
                    model_path, model_name, metrics, promote
                )
                logger.info(f"Model {model_name} registered as version {version}")

                if promote:
                    logger.info(f"Model {model_name} promoted to {promote}")
            except Exception as e:
                logger.error(f"Error registering model: {e}")
                # Continue even if registration fails

        logger.info(f"Saved {model_name} model to {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        raise
