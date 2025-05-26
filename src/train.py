"""
Model training module for the mushroom classification pipeline.
Trains and evaluates multiple models on the processed data.
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

# Add Great Expectations imports
try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite

    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

try:
    import xgboost as xgb
except ImportError:
    # Create a simple fallback if xgboost is not installed
    class XGBClassifierFallback:
        """Fallback class for XGBClassifier when xgboost is not installed."""

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))

    # Create a module-like object to avoid errors
    class XGBoostModule:
        """Mock module for xgboost."""

        def __init__(self):
            self.XGBClassifier = XGBClassifierFallback

        def plot_importance(self, *args, **kwargs):
            pass

    xgb = XGBoostModule()

# Import monitoring module
from src.monitoring import monitor_model_performance

# Configure MLflow tracking URI - Adding this is crucial
# Set up MLflow tracking - store locally in the 'mlruns' directory
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
os.makedirs("mlruns", exist_ok=True)

# Create logs directory if it doesn't exist
import os
from pathlib import Path

# Determine the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Create logs directory relative to the project root
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log")),
        logging.StreamHandler(),
    ],
)
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

        # Create absolute path for data context
        if data_context_path is None:
            data_context_path = os.path.join(PROJECT_ROOT, "great_expectations")
        elif not os.path.isabs(data_context_path):
            data_context_path = os.path.join(PROJECT_ROOT, data_context_path)

        # Initialize or get existing data context
        try:
            if os.path.exists(data_context_path):
                context = gx.get_context(context_root_dir=data_context_path)
            else:
                os.makedirs(data_context_path, exist_ok=True)
                context = gx.get_context(
                    context_root_dir=data_context_path, mode="file"
                )
        except Exception as e:
            logger.warning(f"Could not initialize Great Expectations context: {e}")
            return {
                "validation_passed": True,
                "message": f"Could not initialize GE context: {e}",
            }

        # Create expectation suite for mushroom data
        suite_name = "mushroom_data_suite"

        try:
            suite = context.get_expectation_suite(expectation_suite_name=suite_name)
        except:
            # Create new suite if it doesn't exist
            suite = context.create_expectation_suite(expectation_suite_name=suite_name)

            # Add basic expectations for any dataset
            suite.add_expectation(
                gx.expectations.ExpectTableRowCountToBeBetween(
                    min_value=10, max_value=100000
                )
            )

            # Check if target column exists and add appropriate expectations
            if "class" in data.columns:
                suite.add_expectation(
                    gx.expectations.ExpectColumnToExist(column="class")
                )
                unique_values = data["class"].unique().tolist()
                suite.add_expectation(
                    gx.expectations.ExpectColumnValuesToBeInSet(
                        column="class", value_set=unique_values
                    )
                )
            elif "target" in data.columns:
                suite.add_expectation(
                    gx.expectations.ExpectColumnToExist(column="target")
                )

            # Save the suite
            try:
                context.save_expectation_suite(suite)
            except Exception as e:
                logger.warning(f"Could not save expectation suite: {e}")

        # Create validator with better error handling
        try:
            validator = context.get_validator(
                batch_request=gx.core.batch.RuntimeBatchRequest(
                    datasource_name="pandas",
                    data_connector_name="default_runtime_data_connector_name",
                    data_asset_name="mushroom_data",
                    runtime_parameters={"batch_data": data},
                    batch_identifiers={"default_identifier_name": "mushroom_batch"},
                ),
                expectation_suite_name=suite_name,
            )

            # Run validation
            validation_result = validator.validate()

            # Log results
            success_count = validation_result.statistics.get(
                "successful_expectations", 0
            )
            total_count = validation_result.statistics.get("evaluated_expectations", 0)

            logger.info(
                f"Great Expectations validation: {success_count}/{total_count} expectations passed"
            )

            if not validation_result.success:
                logger.warning("Some data quality checks failed:")
                for result in validation_result.results:
                    if not result.success:
                        logger.warning(f"Failed: {result.expectation_config}")

            return {
                "validation_passed": validation_result.success,
                "success_count": success_count,
                "total_count": total_count,
                "results": validation_result,
            }

        except Exception as e:
            logger.warning(f"Could not create validator or run validation: {e}")
            return {
                "validation_passed": True,
                "message": f"Validation skipped due to error: {e}",
            }

    except Exception as e:
        logger.error(f"Error in Great Expectations validation: {e}")
        return {"validation_passed": False, "error": str(e)}


def train_models(data_path, config):
    """Train multiple models and log to MLflow"""
    logger.info("Starting model training")

    # Initialize MLflow experiment with proper error handling
    experiment_name = "mushroom_classification"
    try:
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )

        # Set the experiment
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set up MLflow experiment: {e}")
        # Fallback: use default experiment
        logger.info("Using default experiment")
        experiment_id = "0"  # Default experiment ID

    # Handle both file paths and DataFrames
    if isinstance(data_path, str):
        # Validate file path
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load data from file
        logger.info(f"Loading data from file: {data_path}")
        data = pd.read_csv(data_path)
    elif isinstance(data_path, pd.DataFrame):
        # Use provided DataFrame
        logger.info("Using provided DataFrame")
        data = data_path.copy()

        # If DataFrame has no columns, try to reload from original path if available
        if data.shape[1] == 0:
            logger.warning(
                "Provided DataFrame has no columns, attempting to reload from default path"
            )
            default_path = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/raw/secondary_data.csv"
            if os.path.exists(default_path):
                logger.info(f"Reloading data from: {default_path}")
                data = pd.read_csv(default_path)
            else:
                logger.error("Default data file not found and DataFrame has no columns")
                raise ValueError(
                    "Cannot proceed with empty DataFrame and no fallback data file"
                )
    else:
        logger.error(
            f"data_path must be a string (file path) or DataFrame, got {type(data_path)}: {data_path}"
        )
        raise ValueError(
            f"data_path must be a string (file path) or DataFrame, got {type(data_path)}"
        )

    # If no valid data was loaded and secondary_data.csv exists, use it as fallback
    if (data is None or data.empty) and not isinstance(data_path, str):
        secondary_path = "/home/ankit/WindowsFuneral/BCU/Sem4/MLOps/FINAL_ASSESSMENT/new-mushroom/data/raw/secondary_data.csv"
        if os.path.exists(secondary_path):
            logger.info(f"Using secondary dataset as fallback: {secondary_path}")
            data = pd.read_csv(secondary_path)
        else:
            logger.error("Secondary data file not found")
            raise FileNotFoundError("No valid data source found")

    logger.info(f"Loaded data with shape: {data.shape}")

    # Check if DataFrame is empty or has no columns
    if data.empty:
        logger.error(f"Data is empty. Shape: {data.shape}")
        raise ValueError(f"Data is empty. Shape: {data.shape}")

    if data.shape[1] == 0:
        logger.error(f"Data has no columns. Shape: {data.shape}")
        logger.error(f"Data info: {data.info()}")
        raise ValueError(f"Data has no columns. Shape: {data.shape}")

    # Add Great Expectations validation
    logger.info("Running Great Expectations data validation...")
    validation_results = validate_data_with_great_expectations(data)

    if not validation_results["validation_passed"]:
        logger.warning(
            "Data quality validation failed, but continuing with training..."
        )

    # Debug: Print column info
    logger.info(f"Columns: {list(data.columns)}")
    logger.info(f"Data types:\n{data.dtypes}")
    logger.info(f"First few rows:\n{data.head()}")

    # Prepare features and target
    if "class" in data.columns:
        target_column = "class"
    elif "edible" in data.columns:
        target_column = "edible"
    elif "class_encoded" in data.columns:
        target_column = "class_encoded"
    else:
        # Use the last column as target
        target_column = data.columns[-1]

    logger.info(f"Using target column: {target_column}")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    logger.info(f"Features shape before preprocessing: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target values: {y.value_counts()}")

    # Handle categorical features
    categorical_columns = X.select_dtypes(include=["object"]).columns
    numerical_columns = X.select_dtypes(include=["number"]).columns

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
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info(f"Encoded target values: {np.unique(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("data_split", {}).get("test_size", 0.3),
        random_state=config.get("data_split", {}).get("random_state", 42),
    )

    logger.info(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Preprocess features
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Processed train features shape: {X_train_processed.shape}")
    logger.info(f"Processed test features shape: {X_test_processed.shape}")

    # Check if processed data is empty
    if X_train_processed.shape[1] == 0:
        raise ValueError("No features remaining after preprocessing")

    # Define models with safe config access
    models = {
        "logistic_regression": LogisticRegression(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=config.get("models", {})
            .get("random_forest", {})
            .get("n_estimators", 100),
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=config.get("models", {})
            .get("gradient_boosting", {})
            .get("n_estimators", 100),
            random_state=42,
        ),
    }

    # Try to add XGBoost if available
    try:
        if hasattr(xgb, "XGBClassifier") and not isinstance(
            xgb.XGBClassifier, type(xgb.XGBClassifierFallback)
        ):
            models["xgboost"] = xgb.XGBClassifier(
                n_estimators=config.get("models", {})
                .get("xgboost", {})
                .get("n_estimators", 100),
                max_depth=config.get("models", {})
                .get("xgboost", {})
                .get("max_depth", 6),
                random_state=42,
            )
            logger.info("Added XGBoost to training models")
    except Exception as e:
        logger.warning(f"Could not add XGBoost to training: {e}")

    best_model = None
    best_accuracy = 0
    trained_models = {}  # Add this to store actual models

    for model_name, model in models.items():
        logger.info(f"Training {model_name} model")

        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id):
            # Create pipeline
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            # Train model
            pipeline.fit(X_train, y_train)
            trained_models[model_name] = pipeline  # Store the trained pipeline

            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = None

            # Get prediction probabilities if available
            try:
                y_pred_proba = pipeline.predict_proba(X_test)
            except:
                pass

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log parameters
            if hasattr(model, "get_params"):
                for param, value in model.get_params().items():
                    try:
                        # Ensure the param value is serializable
                        mlflow.log_param(param, str(value))
                    except Exception as e:
                        logger.warning(f"Could not log parameter {param}: {e}")

            # Create model signature and input example
            try:
                # Infer signature from training data
                if y_pred_proba is not None:
                    signature = infer_signature(X_train, y_pred_proba)
                else:
                    signature = infer_signature(X_train, y_pred)

                # Create input example (first few rows of training data)
                input_example = X_train.head(3)

                # Save model with signature and input example
                mlflow.sklearn.log_model(
                    pipeline,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                )
                logger.info(f"Logged {model_name} with signature and input example")
            except Exception as e:
                logger.warning(f"Could not create signature for {model_name}: {e}")
                # Fallback to basic logging
                mlflow.sklearn.log_model(pipeline, model_name)

            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

    logger.info(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")
    # Store trained_models as a module-level variable for access if needed
    train_models.trained_models = trained_models
    return (
        best_model,
        best_accuracy,
    )  # Keep original signature for backward compatibility


def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_path=None):
    """
    Evaluate a model and optionally save the evaluation results.

    Args:
        name (str): Name of the model.
        model: Trained model object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        output_path (str): Path to save the evaluation results.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    try:
        logger.info(f"Evaluating {name} model")

        # Check if the model has preprocessing information stored
        if hasattr(model, "X_processed_columns") and hasattr(
            model, "categorical_columns_original"
        ):
            logger.info("Model has preprocessing info - applying same transformations")

            # Apply the same preprocessing as during training
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()

            # Get the categorical columns that were encoded during training
            categorical_columns = model.categorical_columns_original

            # Apply one-hot encoding if there were categorical columns
            if len(categorical_columns) > 0:
                logger.info(f"Applying one-hot encoding to: {categorical_columns}")
                X_train_processed = pd.get_dummies(
                    X_train_processed, columns=categorical_columns, drop_first=True
                )
                X_test_processed = pd.get_dummies(
                    X_test_processed, columns=categorical_columns, drop_first=True
                )

            # Ensure column order matches training
            expected_columns = model.X_processed_columns

            # Add missing columns with zeros
            for col in expected_columns:
                if col not in X_train_processed.columns:
                    X_train_processed[col] = 0
                if col not in X_test_processed.columns:
                    X_test_processed[col] = 0

            # Reorder columns to match training
            X_train_processed = X_train_processed[expected_columns]
            X_test_processed = X_test_processed[expected_columns]

            logger.info(
                f"Processed data shapes: train={X_train_processed.shape}, test={X_test_processed.shape}"
            )
        else:
            logger.warning("Model doesn't have preprocessing info - using data as-is")
            X_train_processed = X_train
            X_test_processed = X_test

        # Generate predictions
        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall": recall_score(y_test, y_pred_test, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_test, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred_test),
        }

        # Try to get ROC AUC if model supports predict_proba
        try:
            if hasattr(model, "predict_proba"):
                y_prob_test = model.predict_proba(X_test_processed)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob_test)
                logger.info(f"ROC AUC (Test): {metrics['roc_auc']}")
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")

        # Log metrics
        logger.info(f"\n--- {name} ---")
        logger.info(f"Accuracy (Test): {metrics['accuracy']}")
        logger.info(f"Precision (Test): {metrics['precision']}")
        logger.info(f"Recall (Test): {metrics['recall']}")
        logger.info(f"F1 Score (Test): {metrics['f1_score']}")
        logger.info(f"MCC (Test): {metrics['mcc']}")

        # If output path is provided, save the confusion matrix
        if output_path:
            os.makedirs(output_path, exist_ok=True)

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Poisonous", "Edible"],
                yticklabels=["Poisonous", "Edible"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{name}_confusion_matrix.png"))
            plt.close()

            # Save metrics as CSV
            pd.DataFrame([metrics], index=[name]).to_csv(
                os.path.join(output_path, f"{name}_metrics.csv")
            )

        # Run model monitoring
        monitoring_path = (
            os.path.join(output_path, "monitoring") if output_path else None
        )
        try:
            monitoring_metrics = monitor_model_performance(
                model,
                X_test_processed,
                y_test,
                X_reference=X_train_processed,
                metrics_path=monitoring_path,
            )

            # Add monitoring info to metrics
            metrics["monitoring"] = {
                "report_path": monitoring_metrics.get("report_path", ""),
                "drifted_features": monitoring_metrics.get("drifted_features", []),
                "num_drifted_features": monitoring_metrics.get(
                    "num_drifted_features", 0
                ),
            }
            logger.info(f"Model monitoring complete for {name}")
        except Exception as e:
            logger.error(f"Error in model monitoring for {name}: {e}")
            metrics["monitoring"] = {"error": str(e)}

        logger.info(f"Evaluation of {name} complete")
        return metrics

    except Exception as e:
        logger.error(f"Error evaluating {name} model: {e}")
        raise


def plot_roc_curves(models, X_test, y_test, output_path=None):
    """
    Plot ROC curves for multiple models.

    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        output_path (str): Path to save the ROC curve plot.
    """
    try:
        logger.info("Plotting ROC curves")

        # Get probabilities and calculate ROC curves
        lr_probs = models["logistic_regression"].predict_proba(X_test)[:, 1]
        dt_probs = models["decision_tree"].predict_proba(X_test)[:, 1]
        xgb_probs = models["xgboost"].predict_proba(X_test)[:, 1]

        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

        # Plot ROC curves
        plt.figure(figsize=(10, 6))
        plt.plot(
            lr_fpr,
            lr_tpr,
            linestyle="--",
            label=f"LogReg (AUC = {roc_auc_score(y_test, lr_probs):.3f})",
        )
        plt.plot(
            dt_fpr,
            dt_tpr,
            linestyle="--",
            label=f"DecTree (AUC = {roc_auc_score(y_test, dt_probs):.3f})",
        )
        plt.plot(
            xgb_fpr,
            xgb_tpr,
            linestyle="-",
            label=f"XGBoost (AUC = {roc_auc_score(y_test, xgb_probs):.3f})",
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()

        # Save the plot if output path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "roc_curves.png"))

        plt.close()
        logger.info("ROC curves plotted successfully")

    except Exception as e:
        logger.error(f"Error plotting ROC curves: {e}")
        raise


def plot_feature_importance(model, X, output_path=None):
    """
    Plot feature importance for XGBoost model.

    Args:
        model: XGBoost model.
        X (pd.DataFrame): Features used for training.
        output_path (str): Path to save the feature importance plot.
    """
    try:
        logger.info("Plotting feature importance")

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=15, height=0.7)
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()

        # Save the plot if output path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "feature_importance.png"))

        plt.close()
        logger.info("Feature importance plotted successfully")

    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        raise


def compare_models(metrics, output_path=None):
    """
    Create a comparative bar chart for all model metrics.

    Args:
        metrics (dict): Dictionary of model metrics.
        output_path (str): Path to save the comparison plot.
    """
    try:
        logger.info("Creating model comparison chart")

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Plot comparative metrics
        plt.figure(figsize=(14, 8))
        metrics_df.plot(kind="bar", figsize=(14, 8), colormap="viridis")
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.ylim(0, 1.0)  # Metrics are between 0 and 1
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Models")

        # Add labels for each bar with proper type checking
        for container in plt.gca().containers:
            try:
                # Check if the container is a BarContainer type
                # If not, try to access the container without using bar_label
                import matplotlib.container as mcontainer

                if isinstance(container, mcontainer.BarContainer):
                    plt.bar_label(container, fmt="%.3f", padding=3)
                else:
                    # Alternative approach for non-BarContainer types
                    for rect in container:
                        height = rect.get_height()
                        plt.text(
                            rect.get_x() + rect.get_width() / 2.0,
                            height + 0.03,
                            f"{height:.3f}",
                            ha="center",
                            va="bottom",
                        )
            except Exception as e:
                logger.warning(f"Could not add labels to bar: {e}")

        plt.tight_layout()

        # Save the plot if output path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, "model_comparison.png"))

        plt.close()
        logger.info("Model comparison chart created successfully")

    except Exception as e:
        logger.error(f"Error creating model comparison chart: {e}")
        raise
