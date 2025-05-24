FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs data/raw data/processed data/temp models models/metrics config \
    && chmod -R 755 logs data models config

# Set environment variables
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow
ENV ENV=docker

# Create airflow user and group for security
RUN groupadd -r airflow && useradd -r -g airflow airflow \
    && chown -R airflow:airflow /app

# Switch to airflow user
USER airflow

# Expose ports
EXPOSE 8080 8888

# Default command (can be overridden by docker-compose)
CMD ["python", "dags/mushroom_etl_dag.py"]
