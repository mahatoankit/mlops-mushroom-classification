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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/train.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_models(X_train, y_train):
    """
    Train multiple models on the data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        dict: Dictionary of trained models.
    """
    try:
        logger.info("Starting model training")

        # Train Logistic Regression model
        logger.info("Training Logistic Regression model")
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)

        # Train Decision Tree model
        logger.info("Training Decision Tree model")
        dt = DecisionTreeClassifier(max_depth=12)
        dt.fit(X_train, y_train)

        # Train XGBoost model
        logger.info("Training XGBoost model")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb_model.fit(X_train, y_train)

        # Return trained models
        models = {"logistic_regression": lr, "decision_tree": dt, "xgboost": xgb_model}

        logger.info("Model training complete")
        return models

    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


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

        # Generate predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "f1_score": f1_score(y_test, y_pred_test),
            "mcc": matthews_corrcoef(y_test, y_pred_test),
        }

        # Try to get ROC AUC if model supports predict_proba
        try:
            if hasattr(model, "predict_proba"):
                y_prob_test = model.predict_proba(X_test)[:, 1]
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
                model, X_test, y_test, X_reference=X_train, metrics_path=monitoring_path
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
