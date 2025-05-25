FROM apache/airflow:2.9.1

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    libgomp1 \
    libsnappy-dev \
    curl \
    postgresql-client \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    pkg-config \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

USER airflow

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir pybind11 Cython
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

USER root

COPY . .

RUN mkdir -p /app/airflow/dags /app/airflow/logs /app/airflow/plugins \
             /app/data/raw /app/data/processed /app/mlruns && \
    chown -R airflow /app

# Create a symbolic link to ensure DAGs are correctly found
RUN ln -sf /app/airflow/dags/* /opt/airflow/dags/

RUN chmod +x /app/entry_point.sh

ENV AIRFLOW_HOME=/app/airflow
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV ENV=docker 

USER airflow

ENTRYPOINT ["bash", "/app/entry_point.sh"]
CMD ["webserver"]