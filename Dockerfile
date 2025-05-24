FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional MLOps dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    apache-airflow==2.7.3 \
    apache-airflow-providers-postgres==5.7.1 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.24.4 \
    pyyaml==6.0.1 \
    requests==2.31.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/mlruns /app/airflow/dags /app/data/raw /app/data/processed

# Set permissions
RUN chmod +x run_simple.sh docker-entrypoint.sh

# Expose ports
EXPOSE 5000 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Use entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["run-all"]
