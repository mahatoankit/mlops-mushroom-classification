# Simplified Mushroom Classification with MLOps

This project demonstrates a streamlined approach to Machine Learning Operations (MLOps) for mushroom classification. It prioritizes simplicity while maintaining essential MLOps principles.

## Quick Start

Run this single command to set up and start MLflow tracking with model training:

```bash
./run_simple.sh
```

Then access the MLflow UI at: [http://localhost:5000](http://localhost:5000)

## Project Structure (Simplified)

```
├── data/                    # Data files
│   └── raw/                 # Raw data files
├── models/                  # Saved trained models
├── src/                     # Source code
│   ├── train.py             # Model training with MLflow tracking
│   └── transform.py         # Data preprocessing
├── run_simple.sh            # Single script to run everything
└── README.md                # This file
```

## What This Project Includes

1. **MLflow Tracking**

   - Model parameters logging
   - Model metrics tracking
   - Model versioning

2. **Model Training**
   - Multiple model types (Logistic Regression, Decision Tree, XGBoost)
   - Hyperparameter tracking
   - Performance metrics

## The Process

When you run `run_simple.sh`, this happens:

1. MLflow server starts in the background
2. Sample data is prepared (or your existing data is used)
3. Models are trained with MLflow tracking
4. Results are available in the MLflow UI

## Using Your Own Data

To use your own dataset:

```bash
# Start MLflow server
./run_simple.sh start-mlflow

# Train with your own data
python -c "
from src.train import train_models
import pandas as pd

# Load your data
data = pd.read_csv('path/to/your/data.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Train with MLflow tracking
train_models(X, y)
"
```

## Viewing Results

Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

In the MLflow UI, you can:

- Compare model performance
- View model parameters
- Access saved models for deployment

## Advanced Usage (Optional)

For those interested in more advanced MLOps features, the project includes:

1. **Data Transformation**

   - Feature engineering
   - Missing value imputation
   - Categorical encoding

2. **Deployment**
   - Model loading utility
   - Prediction functions

## Cleaning Up

To stop all services:

```bash
./run_simple.sh stop
```

## Requirements

- Python 3.8+
- Basic packages: mlflow, pandas, scikit-learn, xgboost
