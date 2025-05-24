# Mushroom Classification ETL Pipeline

This project implements a robust ETL (Extract, Transform, Load) pipeline for mushroom classification using machine learning. The pipeline includes data preprocessing, model training, evaluation, and serving predictions through an API.

## Project Structure

```
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration file
├── dags/                    # Airflow DAGs
│   └── mushroom_etl_dag.py  # DAG for running the ETL pipeline
├── data/                    # Data files
│   ├── model_input/         # Input data for models
│   ├── processed/           # Processed data
│   ├── raw/                 # Raw data files
│   │   ├── secondary_data.csv
│   └── validation/          # Validation data
├── docker/                  # Docker configuration
│   └── docker-compose.yml   # Docker Compose file
├── logs/                    # Log files
├── models/                  # Trained models
│   └── metrics/             # Model evaluation metrics
│       └── monitoring/      # Model monitoring results
├── notebooks/               # Jupyter notebooks
│   └── Mushroom_Classifier_XGBoost_FullEDA.ipynb  # Original analysis notebook
├── src/                     # Source code
│   ├── extract.py           # Data extraction module
│   ├── load.py              # Data loading module
│   ├── monitoring.py        # Model monitoring module
│   ├── pipeline.py          # Main pipeline script
│   ├── train.py             # Model training module
│   ├── transform.py         # Data transformation module
│   └── model_serving/       # Model serving components
│       ├── api.py           # FastAPI server
│       ├── database.py      # Database utilities
│       └── model_loader.py  # Model loading utilities
└── tests/                   # Unit tests
    ├── test_api.py          # API tests
    ├── test_extract.py      # Extract component tests
    ├── test_load.py         # Load component tests
    └── test_transform.py    # Transform component tests
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for database components)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Start the database services:

```bash
cd docker && docker-compose up -d
```

### Running the ETL Pipeline

To run the ETL pipeline manually:

```bash
./run_etl.sh
```

This will:

1. Extract data from the raw data source
2. Transform the data (clean, encode, impute missing values)
3. Load the processed data to the appropriate files
4. Train multiple models (Logistic Regression, Decision Tree, XGBoost)
5. Evaluate the models and generate metrics
6. Save the models for future use

### Starting the API Server

To start the FastAPI server for model serving:

```bash
./run_api.sh [port]
```

By default, the API will be available at http://localhost:8000. You can also specify a custom port.

### API Endpoints

#### Core Endpoints

- `GET /` - API information
- `POST /predict` - Make a prediction
- `GET /health` - Health check

#### A/B Testing Endpoints

- `GET /ab-tests` - List all A/B tests
- `POST /ab-tests` - Create a new A/B test
- `GET /ab-tests/{test_id}` - Get details about a specific test
- `POST /ab-tests/{test_id}/conclude` - Conclude a test and select winner

### Using the API with Airflow

The ETL pipeline can also be run as an Airflow DAG. Configure Airflow to look at the `dags/` directory and the DAG will be available in the Airflow UI.

## Database Integration

The project includes integration with MariaDB for both OLTP and OLAP operations:

- **OLTP Database (Port 3307)**: Stores transaction data like prediction requests and results
- **OLAP Database (Port 3308)**: Stores analytical data for reporting and analysis

You can access the databases using the Adminer web interface at http://localhost:8081.

## Model Evaluation

The pipeline evaluates models using several metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC Curves and AUC

Visualizations are saved in the `models/metrics/` directory.

## Model Monitoring

The pipeline includes a comprehensive model monitoring system that:

- Tracks model performance over time
- Detects data drift using Kolmogorov-Smirnov tests
- Generates performance visualization and trend charts
- Creates HTML reports for easy interpretation
- Provides alerting capabilities for performance degradation

Monitoring metrics and reports are saved in the `models/metrics/monitoring/` directory.

## A/B Testing

The pipeline includes an A/B testing system that allows you to:

- Compare the performance of different model versions in production
- Allocate traffic between two competing models
- Collect real-world performance metrics
- Perform statistical significance tests
- Automatically select the winning model based on performance

The A/B testing system is integrated with the API and provides endpoints for managing tests.

### Creating an A/B Test

To create a new A/B test:

```bash
curl -X POST "http://localhost:8000/ab-tests?name=xgboost_comparison&model_a=models/xgboost.joblib&model_b=models/registry/xgboost/v2/xgboost.joblib&traffic_split=0.5"
```

### Using an A/B Test in Predictions

To make predictions using an A/B test:

```bash
curl -X POST "http://localhost:8000/predict?ab_test=xgboost_comparison" -H "Content-Type: application/json" -d '{...}'
```

The API will automatically route the request to either model A or B based on the traffic split configuration.

## Testing

Comprehensive unit tests are included for all pipeline components:

- Data extraction tests
- Data transformation tests
- Model loading tests
- API endpoint tests

Run tests using the provided test script:

```bash
./run_tests.sh
```

## Acknowledgements

This project was developed as part of the MLOps curriculum and is based on the original mushroom classification notebook.
