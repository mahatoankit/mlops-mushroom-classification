"""
Load component of the ETL pipeline for Mushroom Classification.
Responsible for saving processed data and trained models.
"""

import os
import pandas as pd
# import pickle # Not used in the provided code, can be removed if truly unused
import logging
import joblib # Used for model saving
from sklearn.model_selection import train_test_split

# Import model versioning (keep this if src.model_versioning exists and is used)
from src.model_versioning import register_and_promote_model

# Get a logger for this module.
# Configuration of this logger (handlers, level) should ideally happen
# in the main application (e.g., Airflow for tasks, or your test script).
logger = logging.getLogger(__name__) # GOOD: Use __name__ for module-specific logger

# --- REMOVE OR MODIFY THE FOLLOWING ---
# # Create logs directory if it doesn't exist
# os.makedirs("logs", exist_ok=True) # REMOVE THIS
#
# # Configure logging
# logging.basicConfig( # REMOVE THIS global basicConfig
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("logs/load.log"), logging.StreamHandler()],
# )
# --- END REMOVE OR MODIFY ---


# Example of how you might set up logging if this script is run standalone for testing
# This will only run if this script is the main entry point.
if __name__ == "__main__":
    # This configuration will only apply if you run `python src/load.py` directly
    # It won't affect Airflow's logging when this module is imported by a DAG.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(), # Log to console
            # Optionally, add a file handler for standalone testing:
            # logging.FileHandler("src_load_standalone_test.log")
        ]
    )
    logger.info("src/load.py executed as main script (for testing).")
    # Add test code here if needed, e.g.:
    # test_df = pd.DataFrame({'feature1': [1,2,3], 'feature2': [4,5,6], 'class_encoded': [0,1,0]})
    # X_train, X_test, y_train, y_test = load_data(test_df, "./temp_processed_data")
    # print("Standalone test of load_data completed.")


def load_data(df, output_path, test_size=0.3, random_state=42):
    """Prepare data for modeling exactly like notebook"""
    logger.info("Preparing data for modeling")

    if "class_encoded" not in df.columns:
        logger.error("class_encoded column not found in data") # Use logger
        raise ValueError("class_encoded column not found in data")

    X = df.drop(columns=["class_encoded"])
    y = df["class_encoded"]

    integer_cols = X.select_dtypes(include=["int64", "int32"]).columns
    if not integer_cols.empty: # Check if there are any integer columns
        X[integer_cols] = X[integer_cols].astype("float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets"
    )

    os.makedirs(output_path, exist_ok=True) # Ensure output path exists
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
    """
    try:
        logger.info(f"Saving {model_name} model")
        os.makedirs(output_path, exist_ok=True)

        if metrics and X_sample is not None and y_sample is not None: # Ensure y_sample also present
            try:
                version = register_and_promote_model(
                    model, model_name, metrics, promote, X_sample, y_sample
                )
                logger.info(f"Model {model_name} registered as version {version}")
                if promote:
                    logger.info(f"Model {model_name} promoted to {promote}")
            except Exception as e:
                logger.error(f"Error registering model {model_name} (versioning API): {e}")
                # Decide if this should be a fatal error or just a warning

        model_path = os.path.join(output_path, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        raise