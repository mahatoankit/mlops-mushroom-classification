# Mushroom Classification MLOps Project: Technical Documentation

## Project Overview

This project implements a complete MLOps pipeline for mushroom classification. It demonstrates the end-to-end machine learning lifecycle from data extraction, transformation, and loading (ETL) to model training, evaluation, deployment, and monitoring using industry-standard tools.

## System Architecture

The project employs a containerized architecture using Docker and Docker Compose to manage various services:

1. **Airflow**: Workflow orchestration
2. **MLflow**: Experiment tracking and model registry
3. **Custom Python services**: Data processing and model training

## Directory Structure

```
new-mushroom/
├── airflow/              # Airflow configuration and DAG definitions
│   ├── dags/             # Workflow definitions
│   ├── logs/             # Airflow logs
│   └── plugins/          # Airflow plugins
├── data/                 # Data storage
│   ├── raw/              # Raw, unprocessed data
│   ├── processed/        # Cleaned and processed data
│   └── temp/             # Temporary data files
├── models/               # Trained models
│   └── metrics/          # Model evaluation metrics
├── mlruns/               # MLflow experiment tracking database
├── mlflow_artifacts/     # MLflow model artifacts
├── config/               # Configuration files
├── src/                  # Source code
└── fix_all.sh            # Utility script to fix permissions and configurations
```

## Key Components

### 1. Airflow

**Purpose**: Apache Airflow handles workflow orchestration, scheduling, and monitoring of the machine learning pipeline.

**Key Features**:

- DAG (Directed Acyclic Graph) definitions for workflow management
- Task scheduling and dependency management
- Web UI for monitoring execution status
- Integration with external systems

**Configuration**:

- Custom dags_folder path: `/app/airflow/dags`
- Authentication setup based on Airflow version
- Disabled example DAGs
- Fast DAG directory scanning (30-second intervals)

### 2. MLflow

**Purpose**: MLflow provides experiment tracking, model versioning, and model registry capabilities.

**Key Features**:

- Experiment tracking for hyperparameter tuning
- Metric logging and visualization
- Model versioning and management
- Model serving capabilities

**Configuration**:

- Artifact storage in `./mlflow_artifacts`
- Database storage in `./mlruns`
- Web UI accessible on port 5000

### 3. Data Pipeline

**Purpose**: Extract, transform, and load data for model training.

**Components**:

- Data extraction from source
- Data cleaning and preprocessing
- Feature engineering
- Dataset splitting and preparation for model training

### 4. Model Training & Evaluation

**Purpose**: Train machine learning models to classify mushrooms and evaluate their performance.

**Components**:

- Model selection and hyperparameter tuning
- Cross-validation
- Performance metrics calculation
- Model comparison and selection

### 5. Docker Infrastructure

**Purpose**: Containerize all components for consistent development and deployment.

**Configuration**:

- Docker Compose for multi-container orchestration
- Volume mounts for persistent storage
- Port mapping for service access
- User permissions management (UID 50000 for Airflow)

## Workflow Execution

The typical workflow in this MLOps pipeline follows these steps:

1. **Data Ingestion**: Raw mushroom data is loaded into the `data/raw` directory
2. **Data Processing**: Airflow DAGs trigger data cleaning and transformation tasks
3. **Model Training**: Multiple models are trained with different parameters
4. **Model Evaluation**: Models are evaluated and compared using predefined metrics
5. **Model Registration**: The best model is registered in the MLflow registry
6. **Deployment**: The selected model is prepared for deployment

## Operational Information

### Access Points

- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5000

### Maintenance Tasks

The `fix_all.sh` script handles common maintenance tasks:

- Setting appropriate permissions for directories
- Creating required directory structure
- Fixing Airflow configuration issues
- Creating test DAGs to verify functionality
- Managing Docker container lifecycle

### Permission Structure

- Airflow directories: UID 50000, permissions 775
- MLflow directories: UID 0 (root), permissions 777
- Data and model directories: permissions 775

## Troubleshooting

Common issues and their solutions:

- DAGs not appearing: Check scheduler logs with `docker compose logs airflow-scheduler`
- Permission problems: Run the fix_all.sh script to reset permissions
- Connection issues: Ensure all containers are running with `docker compose ps`

## Conclusion

This MLOps project demonstrates a comprehensive approach to machine learning operations, employing best practices for automation, reproducibility, and scalability. The combination of Airflow for workflow orchestration and MLflow for experiment tracking creates a robust platform for developing and deploying machine learning models.
