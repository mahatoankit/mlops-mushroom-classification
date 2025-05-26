"""
Monitoring module for the mushroom classification pipeline.
Tracks model performance and data drift.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import logging
from pathlib import Path  # Add Path import

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
        logging.FileHandler(
            os.path.join(LOGS_DIR, "monitoring.log"), mode="a"
        ),  # Use LOGS_DIR
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def monitor_model_performance(
    model, X_test, y_test, X_reference=None, metrics_path=None
):
    """
    Monitor model performance and check for data drift.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        X_reference: Reference data to compare against for drift detection
        metrics_path: Path to save monitoring metrics and visualizations

    Returns:
        dict: Monitoring metrics
    """
    logger.info("Starting model performance monitoring")

    monitoring_results = {
        "feature_importance": [],
        "drifted_features": [],
        "num_drifted_features": 0,
        "report_path": "",
    }

    try:
        # Create output directory
        if metrics_path:
            os.makedirs(metrics_path, exist_ok=True)
            monitoring_results["report_path"] = metrics_path

        # Check for feature importance
        if hasattr(model, "feature_importances_"):
            # Get feature importance
            importances = model.feature_importances_

            # If X_test is a DataFrame, use column names
            if isinstance(X_test, pd.DataFrame):
                indices = np.argsort(importances)[::-1]
                feature_names = X_test.columns
                top_features = [
                    (feature_names[i], importances[i]) for i in indices[:20]
                ]
            else:
                # If not a DataFrame, use indices
                indices = np.argsort(importances)[::-1]
                top_features = [(f"feature_{i}", importances[i]) for i in indices[:20]]

            monitoring_results["feature_importance"] = top_features

            # Plot feature importance
            if metrics_path:
                plt.figure(figsize=(12, 8))
                if isinstance(X_test, pd.DataFrame):
                    plt.bar(
                        range(len(indices[:20])),
                        [importances[i] for i in indices[:20]],
                        align="center",
                    )
                    plt.xticks(
                        range(len(indices[:20])),
                        [feature_names[i] for i in indices[:20]],
                        rotation=90,
                    )
                else:
                    plt.bar(range(20), importances[indices[:20]], align="center")
                    plt.xticks(
                        range(20), [f"Feature {i}" for i in indices[:20]], rotation=90
                    )

                plt.xlabel("Features")
                plt.ylabel("Importance")
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(metrics_path, "feature_importance.png"))
                plt.close()

        # Detect data drift if reference data is provided
        if X_reference is not None:
            if isinstance(X_test, pd.DataFrame) and isinstance(
                X_reference, pd.DataFrame
            ):
                # Get common columns
                common_columns = list(set(X_test.columns) & set(X_reference.columns))

                # Calculate KS statistic for each feature
                drift_metrics = {}
                for col in common_columns:
                    try:
                        # Run KS test
                        ks_stat, p_value = ks_2samp(X_reference[col], X_test[col])
                        drift_metrics[col] = {"ks_stat": ks_stat, "p_value": p_value}

                        # Consider drift if p-value is small
                        if p_value < 0.05:
                            monitoring_results["drifted_features"].append(col)
                    except Exception as e:
                        logger.warning(
                            f"Could not calculate drift for column {col}: {e}"
                        )

                monitoring_results["num_drifted_features"] = len(
                    monitoring_results["drifted_features"]
                )
                monitoring_results["drift_metrics"] = drift_metrics

                # Create drift report
                if metrics_path and monitoring_results["drifted_features"]:
                    plt.figure(figsize=(12, 6))
                    plt.bar(
                        monitoring_results["drifted_features"],
                        [
                            drift_metrics[col]["ks_stat"]
                            for col in monitoring_results["drifted_features"]
                        ],
                    )
                    plt.xlabel("Features")
                    plt.ylabel("KS Statistic")
                    plt.title("Data Drift Detection")
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(os.path.join(metrics_path, "data_drift.png"))
                    plt.close()

                    # Save drift metrics as CSV
                    pd.DataFrame(drift_metrics).T.to_csv(
                        os.path.join(metrics_path, "drift_metrics.csv")
                    )

        logger.info("Model performance monitoring complete")
        return monitoring_results

    except Exception as e:
        logger.error(f"Error in model performance monitoring: {e}")
        monitoring_results["error"] = str(e)
        return monitoring_results
