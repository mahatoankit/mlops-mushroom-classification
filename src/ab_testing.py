"""
A/B Testing module for the mushroom classification pipeline.
Enables comparison of different model versions in a production setting.
"""

import os
import json
import pandas as pd
import numpy as np
import random
import logging
import datetime
import uuid
import yaml
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/ab_testing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load configuration with better error handling
try:
    config_paths = [
        "config/config.yaml",
        "../config/config.yaml",
        "./config.yaml",
        "/app/config/config.yaml",  # Docker path
    ]

    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            break

    if config is None:
        config = {"ab_testing": {"default_traffic_split": 0.5}}
        logger.warning("Configuration file not found, using defaults")

except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    config = {"ab_testing": {"default_traffic_split": 0.5}}


class ABTest:
    """Class for managing A/B tests between model versions."""

    def __init__(
        self,
        name: str,
        model_a: str,
        model_b: str,
        traffic_split: Optional[float] = None,
        success_metric: str = "accuracy",
        min_sample_size: int = 100,
    ):
        """Initialize A/B test.

        Args:
            name: Name of the A/B test
            model_a: Name or path of model A (control/baseline)
            model_b: Name or path of model B (variant/challenger)
            traffic_split: Percentage of traffic to route to model B (0.0-1.0)
            success_metric: Metric to use for determining success
            min_sample_size: Minimum number of samples before evaluating
        """
        self.name = name
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split or config.get("ab_testing", {}).get(
            "default_traffic_split", 0.5
        )
        self.success_metric = success_metric
        self.min_sample_size = min_sample_size

        self.id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.status = "active"

        # Results storage
        self.results_a = []
        self.results_b = []

        # Metrics
        self.metrics_a = {}
        self.metrics_b = {}
        self.comparison = {}

        # Save test configuration
        self._save_test_config()

        logger.info(f"Initialized A/B test '{name}' with ID {self.id}")
        logger.info(
            f"Model A: {model_a}, Model B: {model_b}, Traffic split: {self.traffic_split}"
        )

    def _save_test_config(self):
        """Save test configuration to file."""
        os.makedirs("models/ab_testing", exist_ok=True)

        config = {
            "id": self.id,
            "name": self.name,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "traffic_split": self.traffic_split,
            "success_metric": self.success_metric,
            "min_sample_size": self.min_sample_size,
            "start_time": self.start_time.isoformat(),
            "status": self.status,
        }

        with open(f"models/ab_testing/{self.id}_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def select_model(self) -> str:
        """Select which model to use for a prediction based on traffic split.

        Returns:
            str: Either "A" or "B" indicating which model to use
        """
        if random.random() < self.traffic_split:
            return "B"
        return "A"

    def record_prediction(
        self,
        model: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a prediction result.

        Args:
            model: Which model made the prediction ("A" or "B")
            prediction: The model's prediction
            ground_truth: The actual ground truth (if available)
            metadata: Additional information about the prediction
        """
        if model not in ["A", "B"]:
            raise ValueError(f"Invalid model identifier: {model}. Must be 'A' or 'B'")

        # Create result record
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metadata": metadata or {},
        }

        # Store in the appropriate results list
        if model == "A":
            self.results_a.append(result)
        else:
            self.results_b.append(result)

        # Save results periodically (every 100 predictions)
        if (len(self.results_a) + len(self.results_b)) % 100 == 0:
            self._save_results()

            # Check if we have enough data to evaluate
            if (
                len(self.results_a) >= self.min_sample_size
                and len(self.results_b) >= self.min_sample_size
            ):
                self.evaluate()

    def _save_results(self):
        """Save current results to file."""
        os.makedirs("models/ab_testing", exist_ok=True)

        # Convert to DataFrames for easier analysis
        df_a = pd.DataFrame(self.results_a)
        df_b = pd.DataFrame(self.results_b)

        # Save as parquet files
        df_a.to_parquet(f"models/ab_testing/{self.id}_results_a.parquet", index=False)
        df_b.to_parquet(f"models/ab_testing/{self.id}_results_b.parquet", index=False)

        logger.info(
            f"Saved results for A/B test {self.id}: "
            f"{len(self.results_a)} for model A, {len(self.results_b)} for model B"
        )

    def evaluate(self) -> Dict:
        """Evaluate the current results of the A/B test.

        Returns:
            Dict: Evaluation results
        """
        # Check if we have enough data
        if (
            len(self.results_a) < self.min_sample_size
            or len(self.results_b) < self.min_sample_size
        ):
            logger.warning(
                f"Not enough data for evaluation. "
                f"Model A: {len(self.results_a)}/{self.min_sample_size}, "
                f"Model B: {len(self.results_b)}/{self.min_sample_size}"
            )
            return {
                "status": "insufficient_data",
                "message": f"Need at least {self.min_sample_size} samples per model",
            }

        try:
            # Convert to DataFrames
            df_a = pd.DataFrame(self.results_a)
            df_b = pd.DataFrame(self.results_b)

            # Calculate metrics for each model (if ground truth is available)
            if "ground_truth" in df_a.columns and not df_a["ground_truth"].isna().all():
                # Calculate accuracy
                self.metrics_a["accuracy"] = (
                    df_a["prediction"] == df_a["ground_truth"]
                ).mean()
                self.metrics_b["accuracy"] = (
                    df_b["prediction"] == df_b["ground_truth"]
                ).mean()

                # Additional metrics can be added here

                # Compare metrics
                self.comparison = self._compare_metrics(self.metrics_a, self.metrics_b)

                logger.info(f"A/B test {self.id} evaluation results:")
                logger.info(
                    f"Model A ({self.model_a}) accuracy: {self.metrics_a['accuracy']:.4f}"
                )
                logger.info(
                    f"Model B ({self.model_b}) accuracy: {self.metrics_b['accuracy']:.4f}"
                )
                logger.info(
                    f"Difference: {self.comparison['accuracy']['absolute_diff']:.4f} "
                    f"({self.comparison['accuracy']['relative_diff']:.2f}%)"
                )
                logger.info(
                    f"Statistical significance: {self.comparison['accuracy']['significant']}"
                )
            else:
                logger.warning(
                    "Ground truth data not available for statistical comparison"
                )

            # Save evaluation results
            self._save_evaluation()

            return {
                "status": "success",
                "metrics_a": self.metrics_a,
                "metrics_b": self.metrics_b,
                "comparison": self.comparison,
            }

        except Exception as e:
            logger.error(f"Error evaluating A/B test: {e}")
            return {"status": "error", "message": str(e)}

    def _compare_metrics(self, metrics_a: Dict, metrics_b: Dict) -> Dict:
        """Compare metrics between models and perform statistical tests.

        Args:
            metrics_a: Metrics for model A
            metrics_b: Metrics for model B

        Returns:
            Dict: Comparison results with statistical significance
        """
        comparison = {}

        for metric in metrics_a:
            if metric in metrics_b:
                # Calculate absolute and relative differences
                absolute_diff = metrics_b[metric] - metrics_a[metric]
                relative_diff = (
                    absolute_diff / metrics_a[metric] * 100
                    if metrics_a[metric] != 0
                    else float("inf")
                )

                # Perform statistical significance test (basic z-test for proportions)
                df_a = pd.DataFrame(self.results_a)
                df_b = pd.DataFrame(self.results_b)

                n_a = len(df_a)
                n_b = len(df_b)

                if metric == "accuracy":
                    # For accuracy, we can use proportion test
                    successes_a = (df_a["prediction"] == df_a["ground_truth"]).sum()
                    successes_b = (df_b["prediction"] == df_b["ground_truth"]).sum()

                    # Calculate z-statistic and p-value
                    p_a = successes_a / n_a
                    p_b = successes_b / n_b
                    p_pool = (successes_a + successes_b) / (n_a + n_b)

                    z = (p_b - p_a) / np.sqrt(
                        p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)
                    )
                    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed test

                    significant = p_value < 0.05

                    comparison[metric] = {
                        "absolute_diff": absolute_diff,
                        "relative_diff": relative_diff,
                        "p_value": p_value,
                        "significant": significant,
                        "better_model": (
                            "B"
                            if absolute_diff > 0
                            else "A" if absolute_diff < 0 else "tie"
                        ),
                        "sample_size_a": n_a,
                        "sample_size_b": n_b,
                    }
                else:
                    # For other metrics, just record the difference without statistical test
                    comparison[metric] = {
                        "absolute_diff": absolute_diff,
                        "relative_diff": relative_diff,
                    }

        return comparison

    def _save_evaluation(self):
        """Save evaluation results to file."""
        os.makedirs("models/ab_testing", exist_ok=True)

        evaluation = {
            "id": self.id,
            "name": self.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "comparison": self.comparison,
            "sample_sizes": {"a": len(self.results_a), "b": len(self.results_b)},
        }

        with open(f"models/ab_testing/{self.id}_evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)

    def conclude(self, winner: Optional[str] = None) -> Dict:
        """Conclude the A/B test and select a winner.

        Args:
            winner: Explicitly select winner ("A" or "B"). If None, uses statistical results.

        Returns:
            Dict: Conclusion results
        """
        # Perform final evaluation
        self.evaluate()

        # Determine winner if not explicitly provided
        if winner is None and self.success_metric in self.comparison:
            metric_comparison = self.comparison[self.success_metric]
            if metric_comparison.get("significant", False):
                winner = metric_comparison.get("better_model", None)
            else:
                winner = "tie"

        # Update test status
        self.status = "completed"
        self.end_time = datetime.datetime.now()

        # Save conclusion
        conclusion = {
            "id": self.id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_days": (self.end_time - self.start_time).days,
            "winner": winner,
            "reason": (
                "statistical_significance"
                if winner != "tie"
                else "no_significant_difference"
            ),
            "metrics_a": self.metrics_a,
            "metrics_b": self.metrics_b,
            "comparison": self.comparison,
            "sample_sizes": {"a": len(self.results_a), "b": len(self.results_b)},
        }

        # Save to file
        with open(f"models/ab_testing/{self.id}_conclusion.json", "w") as f:
            json.dump(conclusion, f, indent=2)

        # Update config file with conclusion
        self._save_test_config()

        logger.info(f"Concluded A/B test {self.id} with winner: {winner}")

        return conclusion


class ABTestRegistry:
    """Registry for managing multiple A/B tests."""

    def __init__(self, registry_path="models/ab_testing"):
        """Initialize the A/B test registry.

        Args:
            registry_path: Path to store A/B test data
        """
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)

        # Load active tests
        self.active_tests = self._load_active_tests()

    def _load_active_tests(self) -> Dict[str, ABTest]:
        """Load active A/B tests from disk.

        Returns:
            Dict: Dictionary of active tests by ID
        """
        active_tests = {}

        # Look for test config files
        for filename in os.listdir(self.registry_path):
            if filename.endswith("_config.json"):
                try:
                    with open(os.path.join(self.registry_path, filename), "r") as f:
                        test_config = json.load(f)

                    # Check if test is active
                    if test_config.get("status") == "active":
                        test_id = test_config["id"]

                        # Recreate the ABTest object
                        test = ABTest(
                            name=test_config["name"],
                            model_a=test_config["model_a"],
                            model_b=test_config["model_b"],
                            traffic_split=test_config["traffic_split"],
                            success_metric=test_config["success_metric"],
                            min_sample_size=test_config["min_sample_size"],
                        )

                        # Override the ID to match the stored one
                        test.id = test_id
                        test.start_time = datetime.datetime.fromisoformat(
                            test_config["start_time"]
                        )

                        # Load existing results if available
                        results_a_path = os.path.join(
                            self.registry_path, f"{test_id}_results_a.parquet"
                        )
                        results_b_path = os.path.join(
                            self.registry_path, f"{test_id}_results_b.parquet"
                        )

                        if os.path.exists(results_a_path):
                            df_a = pd.read_parquet(results_a_path)
                            test.results_a = df_a.to_dict("records")

                        if os.path.exists(results_b_path):
                            df_b = pd.read_parquet(results_b_path)
                            test.results_b = df_b.to_dict("records")

                        active_tests[test_id] = test

                except Exception as e:
                    logger.error(f"Error loading test from {filename}: {e}")

        logger.info(f"Loaded {len(active_tests)} active A/B tests")
        return active_tests

    def create_test(
        self,
        name: str,
        model_a: str,
        model_b: str,
        traffic_split: Optional[float] = None,
        success_metric: str = "accuracy",
        min_sample_size: int = 100,
    ) -> ABTest:
        """Create a new A/B test.

        Args:
            name: Name of the A/B test
            model_a: Name or path of model A (control/baseline)
            model_b: Name or path of model B (variant/challenger)
            traffic_split: Percentage of traffic to route to model B (0.0-1.0)
            success_metric: Metric to use for determining success
            min_sample_size: Minimum number of samples before evaluating

        Returns:
            ABTest: The created test
        """
        test = ABTest(
            name=name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            success_metric=success_metric,
            min_sample_size=min_sample_size,
        )

        self.active_tests[test.id] = test
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get an A/B test by ID.

        Args:
            test_id: ID of the test to retrieve

        Returns:
            ABTest: The requested test or None if not found
        """
        return self.active_tests.get(test_id)

    def get_test_by_name(self, name: str) -> Optional[ABTest]:
        """Get an active A/B test by name.

        Args:
            name: Name of the test to retrieve

        Returns:
            ABTest: The most recent test with the given name, or None if not found
        """
        matching_tests = [
            test for test in self.active_tests.values() if test.name == name
        ]

        if not matching_tests:
            return None

        # Return the most recently created test
        return sorted(matching_tests, key=lambda t: t.start_time, reverse=True)[0]

    def list_active_tests(self) -> List[Dict]:
        """List all active A/B tests.

        Returns:
            List: List of active tests basic info
        """
        return [
            {
                "id": test.id,
                "name": test.name,
                "model_a": test.model_a,
                "model_b": test.model_b,
                "traffic_split": test.traffic_split,
                "start_time": test.start_time.isoformat(),
                "samples_a": len(test.results_a),
                "samples_b": len(test.results_b),
            }
            for test in self.active_tests.values()
        ]

    def list_all_tests(self) -> List[Dict]:
        """List all A/B tests (both active and completed).

        Returns:
            List: List of all tests basic info
        """
        tests = []

        # Look for all test config files
        for filename in os.listdir(self.registry_path):
            if filename.endswith("_config.json"):
                try:
                    with open(os.path.join(self.registry_path, filename), "r") as f:
                        test_config = json.load(f)

                    # Find conclusion if available
                    conclusion_path = os.path.join(
                        self.registry_path, f"{test_config['id']}_conclusion.json"
                    )

                    conclusion = None
                    if os.path.exists(conclusion_path):
                        with open(conclusion_path, "r") as f:
                            conclusion = json.load(f)

                    tests.append(
                        {
                            "id": test_config["id"],
                            "name": test_config["name"],
                            "model_a": test_config["model_a"],
                            "model_b": test_config["model_b"],
                            "status": test_config["status"],
                            "start_time": test_config["start_time"],
                            "winner": conclusion["winner"] if conclusion else None,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error loading test from {filename}: {e}")

        return tests

    def conclude_test(self, test_id: str, winner: Optional[str] = None) -> Dict:
        """Conclude an A/B test.

        Args:
            test_id: ID of the test to conclude
            winner: Explicitly select winner ("A" or "B"). If None, uses statistical results.

        Returns:
            Dict: Conclusion results or error message
        """
        test = self.active_tests.get(test_id)
        if not test:
            return {
                "status": "error",
                "message": f"Test {test_id} not found or not active",
            }

        # Conclude the test
        conclusion = test.conclude(winner)

        # Remove from active tests
        self.active_tests.pop(test_id, None)

        return conclusion


def create_ab_test(
    name: str, model_a: str, model_b: str, traffic_split: float = 0.5
) -> str:
    """Create a new A/B test and register it.

    Args:
        name: Name for the test
        model_a: Path or identifier for model A (control)
        model_b: Path or identifier for model B (variant)
        traffic_split: Percentage of traffic to route to model B

    Returns:
        str: ID of the created test
    """
    registry = ABTestRegistry()
    test = registry.create_test(
        name=name, model_a=model_a, model_b=model_b, traffic_split=traffic_split
    )

    return test.id


def get_model_for_request(test_name: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Determine which model to use for a prediction request.

    Args:
        test_name: Name of the A/B test to use (most recent if multiple)

    Returns:
        Tuple: (model_path, test_id) - The model to use and the test ID if part of a test
    """
    # If no test specified, use production model
    if test_name is None:
        from src.model_versioning import get_production_model_path

        try:
            return get_production_model_path("xgboost"), None
        except:
            # Default fallback
            return "models/xgboost.joblib", None

    # If test is specified, select based on A/B test
    registry = ABTestRegistry()
    test = registry.get_test_by_name(test_name)

    if test is None:
        logger.warning(f"A/B test '{test_name}' not found, using production model")
        return get_model_for_request(None)

    # Select model based on traffic split
    model_key = test.select_model()
    model_path = test.model_a if model_key == "A" else test.model_b

    return model_path, test.id


def record_prediction_result(
    test_id: str,
    model: str,
    prediction: Any,
    ground_truth: Optional[Any] = None,
    metadata: Optional[Dict] = None,
) -> bool:
    """Record a prediction result for an A/B test.

    Args:
        test_id: ID of the A/B test
        model: Which model made the prediction ("A" or "B")
        prediction: The model's prediction
        ground_truth: The actual ground truth (if available)
        metadata: Additional information about the prediction

    Returns:
        bool: True if recorded successfully, False otherwise
    """
    if not test_id:
        return False

    registry = ABTestRegistry()
    test = registry.get_test(test_id)

    if test is None:
        logger.warning(f"A/B test '{test_id}' not found, prediction not recorded")
        return False

    test.record_prediction(model, prediction, ground_truth, metadata)
    return True
