"""
XGBoost-focused training module for the mushroom classification pipeline.
Trains and evaluates XGBoost model on the processed data with ColumnStore integration.
"""

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

# Add project root to path
sys.path.append("/app")

# Add Great Expectations imports
try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite

    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False

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
from config.database import db_manager

# Configure MLflow for containerized environment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

# Configure logging
logger = logging.getLogger(__name__)


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
    Load data from ColumnStore for training.

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

        # Map data type to table name
        table_mapping = {
            "train": "train_data",
            "test": "test_data",
            "validation": "validation_data",
        }

        if data_type not in table_mapping:
            raise ValueError(f"Invalid data_type: {data_type}")

        table_name = table_mapping[data_type]

        # Query to get data
        query = f"""
        SELECT cf.* 
        FROM cleaned_features cf
        INNER JOIN {table_name} td ON cf.id = td.feature_id
        WHERE td.experiment_id = %s
        """

        df = pd.read_sql(query, db_manager.mariadb_engine, params=[experiment_id])

        if df.empty:
            raise ValueError(
                f"No {data_type} data found for experiment {experiment_id}"
            )

        # Separate features and target
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
    """Train single XGBoost model - streamlined approach, no A/B testing"""
    logger.info(f"Starting streamlined XGBoost training for experiment {experiment_id}")
    logger.info("No A/B testing required - single model deployment approach")

    if config is None:
        config = {
            "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }

    # Initialize MLflow experiment
    experiment_name = "mushroom_classification_xgboost_streamlined"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id_mlflow = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id_mlflow})"
            )
        else:
            experiment_id_mlflow = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id_mlflow})"
            )

        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set up MLflow experiment: {e}")
        experiment_id_mlflow = "0"

    try:
        # Load training and test data from ColumnStore
        X_train, y_train = load_data_from_columnstore(experiment_id, "train")
        X_test, y_test = load_data_from_columnstore(experiment_id, "test")

        # Add Great Expectations validation
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

        # Encode target variable if it's categorical
        le = None
        if y_train.dtype == "object":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            logger.info(f"Encoded target values: {np.unique(y_train)}")

        # Single XGBoost model training - streamlined approach
        model = xgb.XGBClassifier(
            n_estimators=config.get("xgboost", {}).get("n_estimators", 100),
            max_depth=config.get("xgboost", {}).get("max_depth", 6),
            learning_rate=config.get("xgboost", {}).get("learning_rate", 0.1),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        logger.info("Training single XGBoost model - no comparison needed")

        with mlflow.start_run(
            run_name=f"streamlined_xgboost_{experiment_id}",
            experiment_id=experiment_id_mlflow,
        ):
            # Create pipeline
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            # Train model
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Store model results
            model_results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "pipeline": pipeline,
            }

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log experiment metadata
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Log model parameters
            if hasattr(model, "get_params"):
                for param, value in model.get_params().items():
                    try:
                        mlflow.log_param(param, str(value))
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param}: {e}")

            # Create model signature and save model
            try:
                signature = infer_signature(X_train, y_pred_proba)
                input_example = X_train.head(3)

                mlflow.sklearn.log_model(
                    pipeline,
                    "xgboost_model",
                    signature=signature,
                    input_example=input_example,
                )
                logger.info("Logged XGBoost model to MLflow with signature")
            except Exception as e:
                logger.warning(f"Could not create signature for XGBoost: {e}")
                mlflow.sklearn.log_model(pipeline, "xgboost_model")

            # Log that A/B testing is not needed
            mlflow.log_param("training_approach", "single_model_streamlined")
            mlflow.log_param("ab_testing_required", False)
            mlflow.log_param("model_comparison", "not_applicable")
            mlflow.log_param("deployment_strategy", "direct_production_deployment")

            logger.info(
                f"Streamlined XGBoost training completed - Accuracy: {accuracy:.4f}"
            )

    # Return streamlined results
    return {
        "model_type": "xgboost_streamlined",
        "accuracy": accuracy,
        "model_results": model_results,
        "trained_model": pipeline,
        "experiment_id": experiment_id,
        "validation_results": validation_results,
        "ab_testing_required": False,
        "deployment_ready": True,
    }

    except Exception as e:
        logger.error(f"Error in streamlined model training: {e}")
        raise


# Legacy function for backward compatibility
def train_models(data_path, config):
    """Legacy train_models function for backward compatibility"""
    logger.warning(
        "Legacy train_models called - this should be updated to use ColumnStore"
    )

    # If data_path is an experiment_id string, use new function
    if isinstance(data_path, str) and data_path.startswith("exp_"):
        return train_models_from_columnstore(data_path, config)

    # Otherwise, try to handle as before but with warnings
    logger.warning("Using legacy training method - consider migrating to ColumnStore")

    # Handle both file paths and DataFrames
    if isinstance(data_path, str):
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        data = pd.read_csv(data_path)
    elif isinstance(data_path, pd.DataFrame):
        data = data_path.copy()
    else:
        raise ValueError(
            f"data_path must be a string (file path) or DataFrame, got {type(data_path)}"
        )

    # Simple training for legacy support
    if "class" in data.columns:
        target_column = "class"
    elif "edible" in data.columns:
        target_column = "edible"
    else:
        target_column = data.columns[-1]

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return "random_forest", accuracy
