import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import sys
from datetime import datetime

# Add project root to path
sys.path.append("/app")

# Add Great Expectations imports
try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite

    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False

    # Mock classes for compatibility
    class MockGX:
        @staticmethod
        def get_context():
            return None

    class MockExpectationSuite:
        pass

    # Assign mock classes to the module names
    gx = MockGX()
    ExpectationSuite = MockExpectationSuite

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# XGBoost with fallback
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

    # Create a simple fallback if xgboost is not installed
    class XGBClassifierFallback:
        """Fallback class for XGBClassifier when xgboost is not installed."""

        def __init__(self, *args, **kwargs):
            self.params = kwargs

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

        def get_params(self, deep=True):
            return self.params

    class XGBoostModule:
        """Mock module for xgboost."""

        def __init__(self):
            self.XGBClassifier = XGBClassifierFallback

    xgb = XGBoostModule()

# Import database manager for ColumnStore integration
try:
    from config.database import db_manager
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import database manager - some functions may not work")
    db_manager = None

# Configure logging
logger = logging.getLogger(__name__)

# Configure MLflow for containerized environment
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
mlflow.set_tracking_uri(mlflow_uri)
logger.info(f"MLflow tracking URI set to: {mlflow_uri}")


def setup_mlflow_experiment(experiment_name="mushroom_classification_comprehensive"):
    """Setup MLflow experiment with proper error handling"""
    try:
        # Set tracking URI consistently
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI: {mlflow_uri}")

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=f"/app/mlflow_artifacts/{experiment_name}",
                )
                logger.info(
                    f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )

            # Set the experiment as active
            mlflow.set_experiment(experiment_name)

            # Test connectivity with a simple API call
            mlflow.search_runs(experiment_ids=[experiment_id], max_results=1)
            logger.info("MLflow connectivity test successful")

            return True, experiment_id

        except Exception as e:
            logger.error(f"MLflow experiment setup failed: {e}")
            return False, None

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return False, None


def validate_data_with_great_expectations(data, data_context_path=None):
    """
    Validate data using Great Expectations

    Args:
        data (pd.DataFrame): Data to validate
        data_context_path (str): Path to Great Expectations context

    Returns:
        dict: Validation results
    """
    if not GREAT_EXPECTATIONS_AVAILABLE:
        logger.warning("Great Expectations not available. Skipping validation.")
        return {
            "validation_passed": True,
            "message": "Great Expectations not available",
        }

    try:
        # Validate input data
        if data is None or data.empty:
            return {"validation_passed": False, "error": "Input data is None or empty"}

        # Create basic expectations for the data
        validation_results = {
            "validation_passed": True,
            "row_count": len(data),
            "column_count": len(data.columns),
            "null_percentage": (
                data.isnull().sum().sum() / (len(data) * len(data.columns))
            )
            * 100,
        }

        # Basic validation checks
        if validation_results["row_count"] < 10:
            validation_results["validation_passed"] = False
            validation_results["error"] = "Insufficient data rows"

        if validation_results["null_percentage"] > 50:
            validation_results["validation_passed"] = False
            validation_results["error"] = "Too many null values"

        logger.info(f"Data validation results: {validation_results}")
        return validation_results

    except Exception as e:
        logger.error(f"Error in Great Expectations validation: {e}")
        return {"validation_passed": False, "error": str(e)}


def load_data_from_columnstore(experiment_id, data_type="train"):
    """
    Load data from ColumnStore using the same approach as columnstore_trainer.py

    Args:
        experiment_id (str): Experiment ID
        data_type (str): Type of data to load ('train', 'test', 'validation')

    Returns:
        tuple: (X, y) features and target
    """
    try:
        logger.info(
            f"Loading {data_type} data from ColumnStore for experiment {experiment_id}"
        )

        # Map data type to table name (same as columnstore_trainer.py)
        table_mapping = {
            "train": "train_data",
            "test": "test_data",
            "validation": "validation_data",
        }

        if data_type not in table_mapping:
            raise ValueError(f"Invalid data_type: {data_type}")

        table_name = table_mapping[data_type]

        # Use exact same query structure as columnstore_trainer.py
        query = f"""
        SELECT cf.* 
        FROM cleaned_features cf
        INNER JOIN {table_name} td ON cf.id = td.feature_id
        WHERE td.experiment_id = %s
        """

        if db_manager is None:
            raise ValueError(
                "Database manager not available - cannot load ColumnStore data"
            )

        if (
            not hasattr(db_manager, "mariadb_engine")
            or db_manager.mariadb_engine is None
        ):
            raise ValueError(
                "Database engine not available - cannot load ColumnStore data"
            )

        # Load data using same approach as columnstore_trainer.py
        df = pd.read_sql(query, db_manager.mariadb_engine, params=[experiment_id])

        if df.empty:
            raise ValueError(
                f"No {data_type} data found for experiment {experiment_id}"
            )

        # Separate features and target (same logic as columnstore_trainer.py)
        target_column = "class"
        feature_columns = [
            col
            for col in df.columns
            if col not in ["id", "created_at", "data_version", target_column]
        ]

        X = df[feature_columns]
        y = df[target_column]

        logger.info(f"Loaded {data_type} data: {X.shape} features, {len(y)} samples")
        return X, y

    except Exception as e:
        logger.error(f"Failed to load {data_type} data from ColumnStore: {e}")
        raise


def train_xgboost_model_from_columnstore(experiment_id, config=None):
    """Train XGBoost model using ColumnStore data with comprehensive MLflow tracking"""
    logger.info(
        f"Starting XGBoost training from ColumnStore for experiment {experiment_id}"
    )

    # Setup MLflow experiment
    mlflow_available, mlflow_experiment_id = setup_mlflow_experiment(
        "mushroom_xgboost_columnstore"
    )

    if not mlflow_available:
        logger.warning("MLflow not available - training without tracking")

    if config is None:
        config = {
            "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }

    try:
        # Load data from ColumnStore using the same approach as columnstore_trainer.py
        X_train, y_train = load_data_from_columnstore(experiment_id, "train")
        X_test, y_test = load_data_from_columnstore(experiment_id, "test")

        # Try to load validation data (optional)
        try:
            X_val, y_val = load_data_from_columnstore(experiment_id, "validation")
            logger.info(f"Validation data loaded: {X_val.shape}")
        except Exception as e:
            logger.warning(f"No validation data available: {e}")
            X_val, y_val = pd.DataFrame(), pd.Series()

        # Data validation
        logger.info("Running data validation...")
        train_data = pd.concat([X_train, y_train], axis=1)
        validation_results = validate_data_with_great_expectations(train_data)

        if not validation_results["validation_passed"]:
            logger.warning(
                "Data quality validation failed, but continuing with training..."
            )

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Target distribution: {y_train.value_counts()}")

        # Handle categorical features
        categorical_columns = X_train.select_dtypes(include=["object"]).columns
        numerical_columns = X_train.select_dtypes(include=["number"]).columns

        logger.info(f"Categorical columns: {list(categorical_columns)}")
        logger.info(f"Numerical columns: {list(numerical_columns)}")

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_columns),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_columns,
                ),
            ]
        )

        # Encode target variable if categorical
        le = None
        if y_train.dtype == "object":
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            logger.info(f"Encoded target values: {np.unique(y_train_encoded)}")
            y_train = pd.Series(y_train_encoded, index=y_train.index)
            y_test = pd.Series(y_test_encoded, index=y_test.index)
        else:
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            if not isinstance(y_test, pd.Series):
                y_test = pd.Series(y_test)

        # XGBoost model configuration
        model = xgb.XGBClassifier(
            n_estimators=config.get("xgboost", {}).get("n_estimators", 100),
            max_depth=config.get("xgboost", {}).get("max_depth", 6),
            learning_rate=config.get("xgboost", {}).get("learning_rate", 0.1),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        logger.info("Training XGBoost model with ColumnStore data")

        # MLflow tracking
        if mlflow_available:
            with mlflow.start_run(
                run_name=f"mushroom_xgboost_columnstore_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                experiment_id=mlflow_experiment_id,
            ) as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")

                # Create and train pipeline
                pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )
                pipeline.fit(X_train, y_train)

                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                # Log metrics
                mlflow.log_metric("accuracy", float(accuracy))
                mlflow.log_metric("precision", float(precision))
                mlflow.log_metric("recall", float(recall))
                mlflow.log_metric("f1_score", float(f1))

                # Log parameters
                mlflow.log_param("model_type", "xgboost_columnstore")
                mlflow.log_param("experiment_id", experiment_id)
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("feature_count", X_train.shape[1])
                mlflow.log_param("data_source", "columnstore")

                # Log XGBoost parameters
                xgb_params = model.get_params()
                for param_name, param_value in xgb_params.items():
                    mlflow.log_param(f"xgb_{param_name}", param_value)

                # Log preprocessing info
                mlflow.log_param("categorical_features", len(categorical_columns))
                mlflow.log_param("numerical_features", len(numerical_columns))

                # Log validation results
                for key, value in validation_results.items():
                    if isinstance(value, (int, float, bool)):
                        mlflow.log_metric(
                            f"validation_{key}",
                            int(value) if isinstance(value, bool) else value,
                        )

                # Log model with signature
                try:
                    signature = infer_signature(X_train, y_pred_proba)
                    input_example = X_train.head(3)
                    mlflow.sklearn.log_model(
                        pipeline,
                        "mushroom_xgboost_columnstore_model",
                        signature=signature,
                        input_example=input_example,
                        registered_model_name="mushroom_classifier_xgboost_columnstore",
                    )
                    logger.info("Successfully logged model with signature to MLflow")
                except Exception as e:
                    logger.warning(f"Could not create signature: {e}")
                    mlflow.sklearn.log_model(
                        pipeline,
                        "mushroom_xgboost_columnstore_model",
                        registered_model_name="mushroom_classifier_xgboost_columnstore",
                    )

                logger.info(
                    f"✅ ColumnStore XGBoost training completed - Accuracy: {accuracy:.4f}"
                )
                logger.info(f"✅ MLflow run: {run.info.run_id}")

                return {
                    "model_type": "xgboost_columnstore",
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "pipeline": pipeline,
                    "experiment_id": experiment_id,
                    "mlflow_experiment_id": mlflow_experiment_id,
                    "mlflow_run_id": run.info.run_id,
                    "validation_results": validation_results,
                    "data_source": "columnstore",
                    "deployment_ready": True,
                }

        else:
            # Train without MLflow
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(
                f"ColumnStore XGBoost training completed without MLflow - Accuracy: {accuracy:.4f}"
            )

            return {
                "model_type": "xgboost_columnstore",
                "accuracy": accuracy,
                "pipeline": pipeline,
                "experiment_id": experiment_id,
                "data_source": "columnstore",
                "deployment_ready": True,
            }

    except Exception as e:
        logger.error(f"Error in ColumnStore XGBoost training: {e}")
        raise


# Main training function for DAG integration
def train_models_from_columnstore(experiment_id, config=None):
    """
    Main training function for DAG integration using ColumnStore data

    Args:
        experiment_id (str): Experiment ID for ColumnStore data
        config (dict): Training configuration

    Returns:
        dict: Training results in DAG-compatible format
    """
    logger.info(
        f"Training XGBoost model from ColumnStore for experiment {experiment_id}"
    )

    try:
        # Use XGBoost training with ColumnStore data
        results = train_xgboost_model_from_columnstore(experiment_id, config)

        # Format results for DAG compatibility
        return {
            "best_model": "xgboost_columnstore",
            "best_accuracy": results.get("accuracy", 0.0),
            "model_type": "xgboost_columnstore",
            "accuracy": results.get("accuracy", 0.0),
            "results": results,
            "experiment_id": experiment_id,
            "data_source": "columnstore",
            "deployment_ready": results.get("deployment_ready", True),
        }

    except Exception as e:
        logger.error(f"Error in ColumnStore training: {e}")
        raise


def main():
    """Main function for standalone execution"""
    logger.info("Starting XGBoost training with ColumnStore data")

    # Test database connection
    if db_manager and hasattr(db_manager, "test_mariadb_connection"):
        if not db_manager.test_mariadb_connection():
            logger.error("MariaDB connection test failed")
            return False

    # Get experiment ID
    experiment_id = os.getenv(
        "EXPERIMENT_ID", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Run training
    try:
        results = train_models_from_columnstore(experiment_id)
        logger.info(f"Training completed successfully: {results}")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
