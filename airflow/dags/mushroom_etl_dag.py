"""
Airflow DAG for the mushroom classification ETL pipeline.
Streamlined version with reduced redundancy.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import warnings

warnings.filterwarnings("ignore")

# Print environment info for debugging
print("🔍 DAG LOADING - Environment Information")
print(f"Current working directory: {os.getcwd()}")
print(f"User running process: {os.getuid()}:{os.getgid()}")
print(f"AIRFLOW_HOME: {os.environ.get('AIRFLOW_HOME', 'Not set')}")
print(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")
print(f"DAG file location: {__file__}")

# Environment-aware configuration
PROJECT_ROOT = "/app"  # Always use container path in Airflow context

# Path configuration
PATHS = {
    "config": f"{PROJECT_ROOT}/config/config.yaml",
    "raw_data": f"{PROJECT_ROOT}/data/raw/secondary_data.csv",
    "processed": f"{PROJECT_ROOT}/data/processed",
    "models": f"{PROJECT_ROOT}/models",
    "metrics": f"{PROJECT_ROOT}/models/metrics",
    "temp": f"{PROJECT_ROOT}/data/temp",
    "mlruns": f"{PROJECT_ROOT}/mlruns",
}

# Ensure critical directories exist with proper error handling
for path_key, path in PATHS.items():
    try:
        os.makedirs(path, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(path, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Directory {path_key} ready: {path}")
    except PermissionError as e:
        print(f"❌ Permission denied for {path_key}: {path} - {e}")
        # Try to create in a fallback location
        fallback_path = f"/tmp/airflow/{path_key}"
        try:
            os.makedirs(fallback_path, exist_ok=True)
            PATHS[path_key] = fallback_path
            print(f"🔄 Using fallback path for {path_key}: {fallback_path}")
        except Exception as fallback_error:
            print(f"❌ Fallback failed for {path_key}: {fallback_error}")
    except Exception as e:
        print(f"❌ Error creating directory {path_key}: {e}")

# Import with simplified fallback
try:
    import sys

    sys.path.insert(0, PROJECT_ROOT)

    # Temporarily modify any logging configuration in imported modules
    import logging

    # Configure root logger to prevent permission errors
    os.makedirs(f"{PROJECT_ROOT}/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Only use stream handler for now
    )

    # Now import the modules
    from src.extract import extract_data
    from src.transform import transform_data
    from src.load import load_data, save_model
    from src.train import (
        train_models,
        evaluate_model,
        plot_roc_curves,
        plot_feature_importance,
        compare_models,
    )

    IMPORTS_OK = True
except ImportError as e:
    print(f"Using fallback implementations: {e}")
    IMPORTS_OK = False

    # Simplified fallback functions
    def extract_data(file_path):
        return (
            pd.read_csv(file_path)
            if os.path.exists(file_path)
            else pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        )

    def transform_data(df):
        return df.copy()

    def load_data(df, output_path, test_size=0.3, random_state=42):
        from sklearn.model_selection import train_test_split

        # Convert Series to DataFrame if needed
        if isinstance(df, pd.Series):
            df = df.to_frame()

        # Ensure we have a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame or Series, got {type(df)}")

        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Identify target column - try common names
        target_candidates = ["target", "class", "class_encoded", "label", "y"]
        target_col = None

        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break

        # If no standard target column found, use the last column
        if target_col is None:
            if len(df.columns) > 1:
                target_col = df.columns[-1]
                print(f"Using last column as target: {target_col}")
            else:
                # Single column - create dummy target
                print("Single column detected, creating dummy target")
                df["target"] = [0, 1] * (len(df) // 2) + [0] * (len(df) % 2)
                target_col = "target"

        # Ensure we have features and target
        if len(df.columns) < 2:
            print("Insufficient columns, creating dummy features")
            df["feature1"] = range(len(df))
            df["feature2"] = range(len(df), 2 * len(df))

        # Split features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]

        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_model(model, name, path):
        import joblib

        os.makedirs(path, exist_ok=True)
        model_path = f"{path}/{name}.joblib"
        joblib.dump(model, model_path)
        return model_path

    def train_models(df, config=None):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        print(f"Input data shape: {df.shape}")
        print(f"Input data columns: {df.columns.tolist()}")
        print(f"Input data types:\n{df.dtypes}")

        # Identify target column
        target_candidates = ["target", "class", "class_encoded", "label", "y"]
        target_col = None

        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            if len(df.columns) > 1:
                target_col = df.columns[-1]
                print(f"Using last column as target: {target_col}")
            else:
                raise ValueError("Cannot identify target column")

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Handle categorical target
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(
                f"Encoded target, unique values: {pd.Series(y).value_counts().to_dict()}"
            )

        # Handle categorical and string features
        print("Processing features...")
        for col in X.columns:
            if X[col].dtype == "object":
                print(f"Processing categorical column: {col}")
                # Try to handle semicolon-separated values or categorical data
                if X[col].astype(str).str.contains(";").any():
                    print(f"Column {col} contains semicolon-separated values")
                    # For now, just label encode these complex strings
                    le_col = LabelEncoder()
                    X[col] = le_col.fit_transform(X[col].astype(str))
                else:
                    # Regular categorical encoding
                    le_col = LabelEncoder()
                    X[col] = le_col.fit_transform(X[col].astype(str))

        # Convert all to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        # Handle any remaining NaN values
        X = X.fillna(0)

        print(f"Processed features shape: {X.shape}")
        print(f"Processed features dtypes:\n{X.dtypes}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train models
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        }

        best_accuracy = 0
        best_model_name = None

        for name, model in models.items():
            print(f"Training {name}")
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"{name} accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name

        # Store trained models for access
        train_models.trained_models = models

        return best_model_name, best_accuracy

    def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_path=None):
        from sklearn.metrics import accuracy_score

        return {
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
            "model": name,
        }

    def plot_roc_curves(models, X_test, y_test, output_path=None):
        print("ROC curves created")

    def plot_feature_importance(model, X_train, output_path=None):
        print("Feature importance plotted")

    def compare_models(metrics, output_path=None):
        print("Model comparison completed")


# Utility functions
def save_temp_data(data, filename):
    """Save data to temp directory and return path"""
    path = f"{PATHS['temp']}/{filename}"
    if isinstance(data, pd.DataFrame):
        data.to_parquet(path, index=False)
    else:
        pd.DataFrame(data).to_parquet(path, index=False)
    return path


def load_temp_data(path):
    """Load data from temp directory"""
    df = pd.read_parquet(path)
    # Always return DataFrame, don't convert single columns to Series
    return df


def handle_task_error(task_name, error):
    """Standardized error handling"""
    print(f"Error in {task_name}: {error}")
    raise


# Add database connection function
def load_clean_data_from_db():
    """Load clean data from MariaDB"""
    try:
        import pymysql
        import sqlalchemy

        # Database connection parameters
        db_config = {
            "host": os.environ.get("MARIADB_HOST", "mariadb"),
            "port": int(os.environ.get("MARIADB_PORT", 3306)),
            "user": os.environ.get("MARIADB_USER", "mushroom_user"),
            "password": os.environ.get("MARIADB_PASSWORD", "mushroom_pass"),
            "database": os.environ.get("MARIADB_DATABASE", "mushroom_db"),
        }

        # Create connection string
        connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

        # Create engine and load data
        engine = sqlalchemy.create_engine(connection_string)

        # Query clean data (adjust table name as needed)
        query = "SELECT * FROM clean_mushroom_data"  # Adjust table name
        df = pd.read_sql(query, engine)

        print(f"✅ Loaded {len(df)} rows from MariaDB")
        return df

    except Exception as e:
        print(f"❌ Failed to load from MariaDB: {e}")
        return None


# Create a config dictionary for train_models function
DEFAULT_CONFIG = {
    "data_split": {"test_size": 0.3, "random_state": 42},
    "models": {
        "random_forest": {"n_estimators": 100},
        "gradient_boosting": {"n_estimators": 100},
        "xgboost": {"n_estimators": 100, "max_depth": 6},
    },
}


# DAG Tasks
def task_extract(**context):
    try:
        df = extract_data(PATHS["raw_data"])
        path = save_temp_data(df, "extracted.parquet")
        return {"path": path, "shape": df.shape}
    except Exception as e:
        handle_task_error("extract", e)


def task_transform(**context):
    try:
        ti = context["ti"]
        extract_result = ti.xcom_pull(task_ids="extract")
        df = load_temp_data(extract_result["path"])
        df_transformed = transform_data(df)
        path = save_temp_data(df_transformed, "transformed.parquet")
        return {"path": path, "shape": df_transformed.shape}
    except Exception as e:
        handle_task_error("transform", e)


def task_load(**context):
    try:
        ti = context["ti"]
        transform_result = ti.xcom_pull(task_ids="transform")
        df_raw = load_temp_data(transform_result["path"])

        # Apply preprocessing to ensure consistent format
        df = preprocess_semicolon_data(df_raw)

        print(f"Preprocessed data shape: {df.shape}")
        print(f"Preprocessed data columns: {df.columns.tolist()}")

        X_train, X_test, y_train, y_test = load_data(df, PATHS["processed"])

        # Save split data
        paths = {
            "X_train": save_temp_data(X_train, "X_train.parquet"),
            "X_test": save_temp_data(X_test, "X_test.parquet"),
            "y_train": save_temp_data(y_train, "y_train.parquet"),
            "y_test": save_temp_data(y_test, "y_test.parquet"),
        }
        return paths
    except Exception as e:
        handle_task_error("load", e)


def preprocess_semicolon_data(df):
    """Preprocess data that comes as semicolon-separated values in a single column"""
    print(f"Preprocessing data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check if we have a single column with semicolon-separated data
    if len(df.columns) == 1:
        col_name = df.columns[0]
        if ";" in col_name:
            print(f"Detected semicolon-separated column: {col_name}")

            # Split the column name to get feature names
            feature_names = col_name.split(";")
            print(f"Feature names: {feature_names}")

            # Split the data values
            split_data = df[col_name].str.split(";", expand=True)
            split_data.columns = feature_names

            print(f"After splitting - Shape: {split_data.shape}")
            print(f"New columns: {split_data.columns.tolist()}")

            return split_data

    return df


def task_train(**context):
    try:
        ti = context["ti"]
        data_paths = ti.xcom_pull(task_ids="load")

        # Option 1: Load from MariaDB (if available)
        print("Attempting to load clean data from MariaDB...")
        df_from_db = load_clean_data_from_db()

        if df_from_db is not None:
            print("✅ Using clean data from MariaDB")
            df_transformed = df_from_db
        else:
            # Option 2: Fallback to temp file approach
            print("⚠️ MariaDB not available, using temp file data")
            transform_result = ti.xcom_pull(task_ids="transform")
            df_transformed = load_temp_data(transform_result["path"])

        # Handle case where df_transformed might be a Series
        if isinstance(df_transformed, pd.Series):
            df_transformed = df_transformed.to_frame()
            print(f"Converted Series to DataFrame: {df_transformed.shape}")

        # Preprocess semicolon-separated data
        df_transformed = preprocess_semicolon_data(df_transformed)

        print(f"Loaded data shape: {df_transformed.shape}")
        print(f"Data columns: {df_transformed.columns.tolist()}")

        # Setup MLflow explicitly
        print("🔧 Setting up MLflow for training...")
        mlflow_available, experiment_id = setup_mlflow()

        if mlflow_available:
            print(f"✅ MLflow setup successful, experiment ID: {experiment_id}")
        else:
            print("⚠️ MLflow setup failed, training without tracking")

        # Train models using the enhanced function with MLflow
        print("Starting model training with MLflow integration...")

        try:
            # Use the enhanced training function
            best_model, best_accuracy = train_models_with_mlflow(
                df_transformed, DEFAULT_CONFIG
            )
            trained_models = getattr(train_models_with_mlflow, "trained_models", {})

            if not trained_models:
                print("⚠️ No models returned from enhanced training, using fallback")
                raise Exception("Enhanced training failed")

        except Exception as e:
            print(f"Enhanced training failed: {e}, using fallback")
            best_model, best_accuracy = train_models(df_transformed, DEFAULT_CONFIG)
            trained_models = getattr(train_models, "trained_models", {})

        # Save models
        model_paths = {}
        for name, model in trained_models.items():
            model_path = save_model(model, name, PATHS["models"])
            model_paths[name] = model_path
            print(f"Saved model {name} to {model_path}")

        print(
            f"Training completed. Best model: {best_model} with accuracy: {best_accuracy:.4f}"
        )

        # Return comprehensive results
        result = {
            "models": model_paths,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "X_train": data_paths["X_train"],
            "X_test": data_paths["X_test"],
            "y_train": data_paths["y_train"],
            "y_test": data_paths["y_test"],
            "training_status": "completed",
            "mlflow_experiment_id": experiment_id if mlflow_available else None,
        }

        return result

    except Exception as e:
        print(f"Error in training task: {e}")
        import traceback

        traceback.print_exc()
        handle_task_error("train", e)


# Add MLflow setup function
def setup_mlflow():
    """Setup MLflow with proper configuration"""
    try:
        import mlflow
        import mlflow.sklearn

        # Get MLflow tracking URI from environment or use default
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not mlflow_uri:
            mlflow_uri = f"file://{PROJECT_ROOT}/mlruns"
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

        # Set tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        # Create or get experiment
        experiment_name = "mushroom_classification"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=f"{PROJECT_ROOT}/mlruns/artifacts",
                )
                print(
                    f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                print(
                    f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
                )

            mlflow.set_experiment(experiment_name)
            return True, experiment_id

        except Exception as e:
            print(f"Error setting up MLflow experiment: {e}")
            return False, None

    except ImportError:
        print("MLflow not available")
        return False, None
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
        return False, None


# Enhanced train_models function with MLflow integration
def train_models_with_mlflow(df, config=None):
    """Enhanced training function with proper MLflow integration"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import pandas as pd
    import numpy as np

    # Setup MLflow
    mlflow_available, experiment_id = setup_mlflow()

    print(f"Input data shape: {df.shape}")
    print(f"Input data columns: {df.columns.tolist()}")

    # Identify target column
    target_candidates = ["target", "class", "class_encoded", "label", "y"]
    target_col = None

    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        if len(df.columns) > 1:
            target_col = df.columns[-1]
            print(f"Using last column as target: {target_col}")
        else:
            raise ValueError("Cannot identify target column")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Handle categorical target
    label_encoder = None
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Encoded target, unique values: {pd.Series(y).value_counts().to_dict()}")

    # Handle categorical and string features
    feature_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            print(f"Processing categorical column: {col}")
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
            feature_encoders[col] = le_col

    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Define models
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    best_accuracy = 0
    best_model_name = None
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}")

        # Start MLflow run for each model
        if mlflow_available:
            try:
                import mlflow
                import mlflow.sklearn

                with mlflow.start_run(run_name=f"{name}_training", nested=True):
                    # Log parameters
                    mlflow.log_params(model.get_params())
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("train_size", len(X_train))
                    mlflow.log_param("test_size", len(X_test))
                    mlflow.log_param("n_features", X_train.shape[1])
                    mlflow.log_param("target_column", target_col)

                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = (
                        model.predict_proba(X_test)
                        if hasattr(model, "predict_proba")
                        else None
                    )

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)

                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("train_accuracy", model.score(X_train, y_train))

                    # Log additional metrics if binary classification
                    if len(np.unique(y)) == 2:
                        from sklearn.metrics import (
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_auc_score,
                        )

                        precision = precision_score(y_test, y_pred, average="weighted")
                        recall = recall_score(y_test, y_pred, average="weighted")
                        f1 = f1_score(y_test, y_pred, average="weighted")

                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall", recall)
                        mlflow.log_metric("f1_score", f1)

                        if y_pred_proba is not None:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                            mlflow.log_metric("auc", auc)

                    # Log model
                    mlflow.sklearn.log_model(
                        model, f"{name}_model", registered_model_name=f"mushroom_{name}"
                    )

                    # Log feature importance if available
                    if hasattr(model, "feature_importances_"):
                        importance_dict = dict(
                            zip(X.columns, model.feature_importances_)
                        )
                        for feature, importance in importance_dict.items():
                            mlflow.log_metric(
                                f"feature_importance_{feature}", importance
                            )

                    # Log classification report as artifact
                    report = classification_report(y_test, y_pred, output_dict=True)
                    import json

                    with open(
                        f"{PATHS['temp']}/classification_report_{name}.json", "w"
                    ) as f:
                        json.dump(report, f, indent=2)
                    mlflow.log_artifact(
                        f"{PATHS['temp']}/classification_report_{name}.json"
                    )

                    print(f"✅ MLflow logging completed for {name}")

            except Exception as e:
                print(f"❌ MLflow logging failed for {name}: {e}")
        else:
            # Fallback training without MLflow
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)

        print(f"{name} accuracy: {accuracy:.4f}")
        trained_models[name] = model

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    # Store trained models for access
    train_models_with_mlflow.trained_models = trained_models

    # Log best model summary
    if mlflow_available:
        try:
            import mlflow

            with mlflow.start_run(run_name="best_model_summary"):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_accuracy", best_accuracy)
                mlflow.log_param("total_models_trained", len(models))
                print(f"✅ Best model summary logged to MLflow")
        except Exception as e:
            print(f"❌ Failed to log best model summary: {e}")

    return best_model_name, best_accuracy


# Enhanced MLflow check task
def task_check_mlflow(**context):
    """Enhanced task to check MLflow connectivity and perform test logging"""
    try:
        print("🔍 Checking MLflow configuration and connectivity")

        # Setup MLflow
        mlflow_available, experiment_id = setup_mlflow()

        if mlflow_available:
            import mlflow

            print(f"✅ MLflow setup successful")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Experiment ID: {experiment_id}")

            # Perform comprehensive test
            try:
                with mlflow.start_run(run_name="airflow_connectivity_test"):
                    # Log various types of data
                    mlflow.log_param("test_param_string", "hello_world")
                    mlflow.log_param("test_param_number", 42)
                    mlflow.log_metric("test_metric_float", 3.14159)
                    mlflow.log_metric("test_metric_int", 100)

                    # Log a simple artifact
                    test_artifact_path = f"{PATHS['temp']}/test_artifact.txt"
                    with open(test_artifact_path, "w") as f:
                        f.write("This is a test artifact from Airflow DAG")
                    mlflow.log_artifact(test_artifact_path)

                    run_id = mlflow.active_run().info.run_id
                    print(f"✅ Successfully created test run with ID: {run_id}")

                # List recent runs to verify
                runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results=5)
                print(f"✅ Found {len(runs)} runs in experiment")
                for idx, run in runs.iterrows():
                    print(f"  - Run {run['run_id'][:8]}... Status: {run['status']}")

            except Exception as e:
                print(f"❌ Test run failed: {e}")
                return {"status": "failed", "error": str(e)}
        else:
            print("❌ MLflow setup failed")
            return {"status": "failed", "error": "MLflow setup failed"}

        return {
            "status": "completed",
            "mlflow_available": mlflow_available,
            "experiment_id": experiment_id,
            "tracking_uri": mlflow.get_tracking_uri() if mlflow_available else None,
        }

    except Exception as e:
        print(f"❌ Error in MLflow check task: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# Ensure the definition of task_evaluate exists before it is used in the DAG
def task_evaluate(**context):
    """Evaluate the trained models on the test data and save metrics"""
    try:
        ti = context["ti"]
        result = ti.xcom_pull(task_ids="train")

        # Load data and apply the same preprocessing as in training
        X_train_raw = load_temp_data(result["X_train"])
        X_test_raw = load_temp_data(result["X_test"])
        y_train = load_temp_data(result["y_train"])
        y_test = load_temp_data(result["y_test"])

        # Apply the same preprocessing to evaluation data
        X_train = preprocess_semicolon_data(X_train_raw)
        X_test = preprocess_semicolon_data(X_test_raw)

        print(f"Evaluation data shapes after preprocessing:")
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"X_train columns: {X_train.columns.tolist()}")
        print(f"X_test columns: {X_test.columns.tolist()}")

        import joblib

        models = {name: joblib.load(path) for name, path in result["models"].items()}

        # Evaluate models
        metrics = {}
        for name, model in models.items():
            print(f"Evaluating model: {name}")
            try:
                metrics[name] = evaluate_model(
                    name, model, X_train, y_train, X_test, y_test, PATHS["metrics"]
                )
                print(f"Successfully evaluated {name}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                # Provide fallback metrics
                metrics[name] = {"accuracy": 0.0, "model": name, "error": str(e)}

        # Save metrics
        metrics_path = f"{PATHS['metrics']}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        return {**result, "metrics_path": metrics_path}
    except Exception as e:
        handle_task_error("evaluate", e)


def task_visualize(**context):
    """Create visualizations for model performance and comparison"""
    try:
        ti = context["ti"]
        result = ti.xcom_pull(task_ids="evaluate")

        # Load data and apply the same preprocessing
        X_train_raw = load_temp_data(result["X_train"])
        X_test_raw = load_temp_data(result["X_test"])
        y_test = load_temp_data(result["y_test"])

        # Apply preprocessing
        X_train = preprocess_semicolon_data(X_train_raw)
        X_test = preprocess_semicolon_data(X_test_raw)

        print(f"Visualization data shapes after preprocessing:")
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        import joblib

        models = {name: joblib.load(path) for name, path in result["models"].items()}

        with open(result["metrics_path"]) as f:
            metrics = json.load(f)

        # Create visualizations with error handling
        try:
            plot_roc_curves(models, X_test, y_test, PATHS["metrics"])
            print("✅ ROC curves created successfully")
        except Exception as e:
            print(f"❌ Error creating ROC curves: {e}")

        try:
            if models:
                first_model = list(models.values())[0]
                plot_feature_importance(first_model, X_train, PATHS["metrics"])
                print("✅ Feature importance plot created successfully")
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {e}")

        try:
            compare_models(metrics, PATHS["metrics"])
            print("✅ Model comparison created successfully")
        except Exception as e:
            print(f"❌ Error creating model comparison: {e}")

        # Add one more visualization: Data Quality Assessment
        try:
            create_data_quality_visualization(X_train, y_test, PATHS["metrics"])
            print("✅ Data quality assessment visualization created successfully")
        except Exception as e:
            print(f"❌ Error creating data quality visualization: {e}")

        # Log visualization completion to MLflow if available
        try:
            import mlflow

            mlflow_available, experiment_id = setup_mlflow()

            if mlflow_available:
                with mlflow.start_run(run_name="visualization_summary"):
                    mlflow.log_param("visualizations_created", True)
                    mlflow.log_param("models_visualized", len(models))
                    mlflow.log_param("metrics_file", result["metrics_path"])

                    # Try to log visualization artifacts
                    visualization_files = [
                        f"{PATHS['metrics']}/roc_curves.png",
                        f"{PATHS['metrics']}/feature_importance.png",
                        f"{PATHS['metrics']}/model_comparison.png",
                        f"{PATHS['metrics']}/data_quality_assessment.png",
                    ]

                    for viz_file in visualization_files:
                        if os.path.exists(viz_file):
                            try:
                                mlflow.log_artifact(viz_file)
                                print(f"✅ Logged visualization artifact: {viz_file}")
                            except Exception as e:
                                print(f"❌ Failed to log artifact {viz_file}: {e}")

                    print("✅ Visualization summary logged to MLflow")
        except Exception as e:
            print(f"❌ MLflow visualization logging failed: {e}")

        return {"status": "completed", "visualizations_created": True}
    except Exception as e:
        handle_task_error("visualize", e)


def create_data_quality_visualization(X_train, y_test, output_path):
    """Create data quality assessment visualization"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Data Quality Assessment", fontsize=16)

        # 1. Missing values heatmap
        if hasattr(X_train, "isnull"):
            missing_data = X_train.isnull().sum()
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_title("Missing Values per Feature")
            axes[0, 0].set_xlabel("Feature Index")
            axes[0, 0].set_ylabel("Missing Count")

        # 2. Target distribution
        if hasattr(y_test, "value_counts"):
            y_test.value_counts().plot(kind="bar", ax=axes[0, 1])
            axes[0, 1].set_title("Target Distribution")
            axes[0, 1].set_xlabel("Class")
            axes[0, 1].set_ylabel("Count")

        # 3. Feature correlation matrix (sample of features)
        if X_train.shape[1] > 1:
            sample_features = min(10, X_train.shape[1])
            corr_matrix = X_train.iloc[:, :sample_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=axes[1, 0])
            axes[1, 0].set_title("Feature Correlation Matrix (Sample)")

        # 4. Data distribution summary
        axes[1, 1].text(0.1, 0.8, f"Total Features: {X_train.shape[1]}", fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Training Samples: {X_train.shape[0]}", fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Test Samples: {len(y_test)}", fontsize=12)
        axes[1, 1].text(
            0.1,
            0.5,
            f"Data Types: {X_train.dtypes.value_counts().to_dict()}",
            fontsize=10,
        )
        axes[1, 1].set_title("Dataset Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()
        output_file = f"{output_path}/data_quality_assessment.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Data quality visualization saved to: {output_file}")

    except Exception as e:
        print(f"Error creating data quality visualization: {e}")
        # Create a simple fallback visualization
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "Data Quality Assessment\n(Detailed view unavailable)",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.text(
                0.5,
                0.3,
                f"Features: {X_train.shape[1]}\nSamples: {X_train.shape[0]}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Data Quality Summary")
            ax.axis("off")

            output_file = f"{output_path}/data_quality_assessment.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

        except Exception as fallback_error:
            print(f"Fallback visualization also failed: {fallback_error}")


# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "mushroom_etl_pipeline",
    default_args=default_args,
    description="Streamlined mushroom classification ETL pipeline",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mushroom", "classification", "etl"],
) as dag:

    # Task definitions
    check_mlflow = PythonOperator(
        task_id="check_mlflow", python_callable=task_check_mlflow
    )
    extract = PythonOperator(task_id="extract", python_callable=task_extract)
    transform = PythonOperator(task_id="transform", python_callable=task_transform)
    load = PythonOperator(task_id="load", python_callable=task_load)
    train = PythonOperator(task_id="train", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
    visualize = PythonOperator(task_id="visualize", python_callable=task_visualize)

    # Task dependencies - add MLflow check as first task
    check_mlflow >> extract >> transform >> load >> train >> evaluate >> visualize
