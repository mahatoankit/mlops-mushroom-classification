"""
Model monitoring module for the mushroom classification pipeline.
Monitors model performance, drift, and provides alerting capabilities.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import ks_2samp
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/monitoring.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from config/config.yaml")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    config = {"monitoring": {"threshold": 0.05}}


class ModelMonitor:
    """Class for monitoring model performance."""

    def __init__(self, model_name, metrics_path):
        """Initialize model monitor.

        Args:
            model_name (str): Name of the model to monitor
            metrics_path (str): Path to store monitoring metrics
        """
        self.model_name = model_name
        self.metrics_path = metrics_path
        self.history_path = os.path.join(metrics_path, f"{model_name}_history.json")
        self.drift_threshold = config.get("monitoring", {}).get("threshold", 0.05)

        # Create metrics directory if it doesn't exist
        os.makedirs(metrics_path, exist_ok=True)

        # Load history if available
        self.metrics_history = self._load_history()

    def _load_history(self):
        """Load metrics history from file."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, "r") as f:
                    return json.load(f)
            else:
                return {"timestamps": [], "metrics": []}
        except Exception as e:
            logger.error(f"Error loading metrics history: {e}")
            return {"timestamps": [], "metrics": []}

    def _save_history(self):
        """Save metrics history to file."""
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.metrics_history, f)
            logger.info(f"Metrics history saved to {self.history_path}")
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")

    def record_metrics(self, metrics):
        """Record new metrics.

        Args:
            metrics (dict): Dictionary containing model metrics
        """
        timestamp = datetime.now().isoformat()
        self.metrics_history["timestamps"].append(timestamp)
        self.metrics_history["metrics"].append(metrics)
        self._save_history()
        logger.info(f"Recorded new metrics for model {self.model_name} at {timestamp}")

    def detect_drift(self, X_reference, X_current):
        """Detect feature drift using Kolmogorov-Smirnov test.

        Args:
            X_reference (pd.DataFrame): Reference data (training data)
            X_current (pd.DataFrame): Current data to check for drift

        Returns:
            dict: Dictionary with drift metrics for each feature
        """
        drift_metrics = {}

        for column in X_reference.columns:
            try:
                # Perform K-S test
                # Check for data types to ensure proper KS test
                ks_result = ks_2samp(
                    X_reference[column].dropna().values,
                    X_current[column].dropna().values,
                )

                # Handle different return types from ks_2samp
                # Always treat as tuple to ensure compatibility
                if isinstance(ks_result, tuple) and len(ks_result) >= 2:
                    ks_statistic = ks_result[0]
                    p_value = ks_result[1]
                else:
                    # If it's not properly unpacking, use a safer default
                    ks_statistic = 0.0
                    p_value = 1.0
                    logger.warning(
                        f"Could not unpack KS test result for column {column}, using default values"
                    )

                # Ensure they're proper floats
                ks_statistic = float(str(ks_statistic).replace(",", "."))
                p_value = float(str(p_value).replace(",", "."))

                # Record drift metrics
                drift_metrics[column] = {
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                    "has_drift": p_value < self.drift_threshold,
                }

                if p_value < self.drift_threshold:
                    logger.warning(
                        f"Drift detected in feature {column}: KS={ks_statistic:.4f}, p={p_value:.4f}"
                    )
            except Exception as e:
                logger.error(f"Error detecting drift for feature {column}: {e}")

        return drift_metrics

    def plot_metrics_trend(self):
        """Plot the trend of model metrics over time."""
        if len(self.metrics_history["timestamps"]) < 2:
            logger.warning("Not enough data points to plot metrics trend")
            return None

        try:
            df_metrics = pd.DataFrame(self.metrics_history["metrics"])
            df_metrics["timestamp"] = self.metrics_history["timestamps"]
            df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])

            metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
            available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]

            plt.figure(figsize=(12, 6))
            for metric in available_metrics:
                plt.plot(
                    df_metrics["timestamp"],
                    df_metrics[metric],
                    marker="o",
                    label=metric,
                )

            plt.title(f"Model Performance Metrics Over Time: {self.model_name}")
            plt.xlabel("Date")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)

            # Save the plot
            plot_path = os.path.join(
                self.metrics_path, f"{self.model_name}_metrics_trend.png"
            )
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Metrics trend plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            logger.error(f"Error plotting metrics trend: {e}")
            return None

    def generate_monitoring_report(self, output_path=None):
        """Generate a model monitoring report.

        Args:
            output_path (str, optional): Path to save the report. Defaults to metrics path.

        Returns:
            str: Path to the generated report
        """
        if output_path is None:
            output_path = self.metrics_path

        os.makedirs(output_path, exist_ok=True)
        report_path = os.path.join(
            output_path, f"{self.model_name}_monitoring_report.html"
        )

        try:
            # Convert metrics history to DataFrame for analysis
            df_metrics = pd.DataFrame(self.metrics_history["metrics"])
            df_metrics["timestamp"] = self.metrics_history["timestamps"]
            df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])

            # Generate HTML report
            with open(report_path, "w") as f:
                f.write(
                    f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>Model Monitoring Report: {self.model_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2c3e50; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                        th {{ background-color: #f2f2f2; }}
                        .alert {{ color: red; font-weight: bold; }}
                        .good {{ color: green; }}
                        .chart {{ margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Model Monitoring Report</h1>
                    <p><strong>Model:</strong> {self.model_name}</p>
                    <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Performance Metrics History</h2>
                    <table>
                        <tr>
                            <th>Timestamp</th>
                """
                )

                # Add metric column headers
                available_metrics = [
                    col for col in df_metrics.columns if col != "timestamp"
                ]
                for metric in available_metrics:
                    f.write(f"<th>{metric}</th>\n")
                f.write("</tr>\n")

                # Add metrics data rows
                for _, row in df_metrics.sort_values(
                    "timestamp", ascending=False
                ).iterrows():
                    f.write("<tr>\n")
                    f.write(f"<td>{row['timestamp']}</td>\n")

                    for metric in available_metrics:
                        value = row.get(metric, "N/A")
                        if isinstance(value, (int, float)):
                            f.write(f"<td>{value:.4f}</td>\n")
                        else:
                            f.write(f"<td>{value}</td>\n")

                    f.write("</tr>\n")

                f.write(
                    """
                    </table>
                    
                    <h2>Performance Analysis</h2>
                """
                )

                # Add metric trends analysis
                if len(df_metrics) >= 2:
                    for metric in ["accuracy", "precision", "recall", "f1_score"]:
                        if metric in df_metrics.columns:
                            latest = df_metrics.iloc[-1][metric]
                            previous = df_metrics.iloc[-2][metric]
                            change = latest - previous
                            change_pct = (change / previous) * 100 if previous else 0

                            css_class = "good" if change >= 0 else "alert"

                            f.write(
                                f"""
                            <p><strong>{metric.capitalize()}:</strong> {latest:.4f} 
                            <span class="{css_class}">({'+' if change >= 0 else ''}{change:.4f}, {change_pct:.2f}%)</span>
                            </p>
                            """
                            )

                # Include the metrics trend chart if available
                metrics_plot = self.plot_metrics_trend()
                if metrics_plot:
                    f.write(
                        f"""
                    <div class="chart">
                        <h2>Performance Metrics Trend</h2>
                        <img src="{os.path.basename(metrics_plot)}" alt="Metrics Trend" style="max-width: 100%;">
                    </div>
                    """
                    )

                f.write(
                    """
                </body>
                </html>
                """
                )

            logger.info(f"Monitoring report generated at {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            return None


def monitor_model_performance(
    model, X_test, y_test, X_reference=None, metrics_path=None
):
    """Monitor model performance and detect data drift.

    Args:
        model: Trained model to monitor
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        X_reference (pd.DataFrame, optional): Reference data for drift detection
        metrics_path (str, optional): Path to store monitoring metrics

    Returns:
        dict: Dictionary with monitoring metrics
    """
    if metrics_path is None:
        metrics_path = "models/monitoring"

    os.makedirs(metrics_path, exist_ok=True)

    try:
        # Get model name
        model_name = getattr(model, "__class__").__name__
        if hasattr(model, "steps") and len(model.steps) > 0:
            model_name = model.steps[-1][1].__class__.__name__

        # Create model monitor
        monitor = ModelMonitor(model_name, metrics_path)

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = None

        # Try to get prediction probabilities if available
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        # Calculate performance metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
        }

        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

        # Initialize metadata dict
        metrics_metadata = {}

        # Add drift metrics if reference data is provided
        if X_reference is not None:
            drift_metrics = monitor.detect_drift(X_reference, X_test)
            metrics_for_record = metrics.copy()  # Create a copy for recording

            # Count features with drift
            drifted_features = [
                f for f, m in drift_metrics.items() if m.get("has_drift", False)
            ]

            # Store as numeric metrics
            metrics["drifted_feature_count"] = len(drifted_features)

            # Store non-numeric data in metadata
            metrics_metadata = {
                "drift_metrics": drift_metrics,
                "drifted_features": drifted_features,
            }

        # Record metrics (only record the float values)
        monitor.record_metrics(metrics)

        # Generate report
        report_path = monitor.generate_monitoring_report()
        metrics_metadata["report_path"] = report_path

        # Add metadata to return dict but not to the recorded metrics
        metrics.update(metrics_metadata)

        logger.info(
            f"Model performance metrics recorded: accuracy={metrics['accuracy']:.4f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"Error monitoring model performance: {e}")
        return {"error": str(e)}
