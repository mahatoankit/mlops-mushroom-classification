FROM apache/airflow:2.9.1

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libmariadb-dev \
    default-mysql-client \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user to install Python packages
USER airflow

# Copy requirements and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Switch back to root to set up directories
USER root

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/config /app/logs && \
    mkdir -p /app/airflow/dags /app/airflow/logs /app/airflow/plugins && \
    mkdir -p /tmp/airflow && \
    chown -R airflow:0 /app && \
    chown -R airflow:0 /tmp/airflow && \
    chmod -R 775 /app && \
    chmod -R 775 /tmp/airflow

# Copy application code with proper ownership
COPY --chown=airflow:0 . /app/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV AIRFLOW_HOME=/app/airflow

# Ensure airflow config directory exists and copy configs
RUN mkdir -p /app/airflow && \
    chown -R airflow:0 /app/airflow && \
    chmod -R 775 /app/airflow

# Copy airflow configs with proper ownership
COPY --chown=airflow:0 airflow/airflow.cfg /app/airflow/airflow.cfg
COPY --chown=airflow:0 airflow/webserver_config.py /app/airflow/webserver_config.py

# Switch back to airflow user for runtime
USER airflow

# Default command
CMD ["airflow", "webserver"]