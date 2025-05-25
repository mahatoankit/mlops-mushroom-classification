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
    """Prepare data for modeling exactly like notebook"""
    logger.info("Preparing data for modeling")

    # Ensure class_encoded is the target
    if "class_encoded" not in df.columns:
        raise ValueError("class_encoded column not found in data")

    # Separate features and target (like notebook)
    X = df.drop(columns=["class_encoded"])
    y = df["class_encoded"]

    # Convert integer columns to float64 to handle potential missing values
    integer_cols = X.select_dtypes(include=["int64", "int32"]).columns
    X[integer_cols] = X[integer_cols].astype("float64")

    # Split data (like notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets"
    )

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    logger.info(f"Saved processed data to {output_path}")

    return X_train, X_test, y_train, y_test


def save_model(
    model,
    model_name,
    output_path,
    metrics=None,
    promote=None,
    X_sample=None,
    y_sample=None,
):
    """
    Save a trained model and register it with the model registry.

    Args:
        model: Trained model object.
        model_name (str): Name of the model.
        output_path (str): Path to save the model.
        metrics (dict, optional): Model performance metrics.
        promote (str, optional): Where to promote the model ('staging' or 'production').
        X_sample (pd.DataFrame, optional): Sample of training features for schema inference.
        y_sample (pd.Series, optional): Sample of training targets for schema inference.

    Returns:
        str: Path to the saved model.
    """
    try:
        logger.info(f"Saving {model_name} model")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Register and potentially promote the model FIRST (before joblib save)
        if metrics and X_sample is not None:
            try:
                version = register_and_promote_model(
                    model, model_name, metrics, promote, X_sample, y_sample # type: ignore
                )
                logger.info(f"Model {model_name} registered as version {version}")

                if promote:
                    logger.info(f"Model {model_name} promoted to {promote}")
            except Exception as e:
                logger.error(f"Error registering model: {e}")
                # Continue even if registration fails

        # Save the model as joblib (backup/compatibility)
        model_path = os.path.join(output_path, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        logger.info(f"Saved {model_name} model to {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        raise
