#!/bin/bash
set -e

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Starting comprehensive fixes for Mushroom MLOps project${NC}"

# Fix permissions
echo -e "${YELLOW}Setting correct permissions for all directories...${NC}"
mkdir -p ./airflow/dags ./airflow/logs ./airflow/plugins
sudo chown -R 50000:50000 ./airflow
sudo chmod -R 775 ./airflow

mkdir -p ./mlruns ./mlflow_artifacts
sudo chown -R 0:0 ./mlruns ./mlflow_artifacts
sudo chmod -R 777 ./mlruns ./mlflow_artifacts

mkdir -p ./data/raw ./data/processed ./data/temp ./models/metrics ./config ./src
sudo chmod -R 775 ./data ./models ./config ./src

mkdir -p ./airflow/dags/__pycache__
sudo chown -R 50000:50000 ./airflow/dags/__pycache__
sudo chmod -R 775 ./airflow/dags/__pycache__

# Stop all containers
echo -e "${YELLOW}Stopping all containers...${NC}"
docker compose down

# Fix the Airflow configuration
echo -e "${YELLOW}Fixing Airflow configuration...${NC}"

# Create a simple test DAG to verify everything is working
echo -e "${YELLOW}Creating a test DAG file...${NC}"
cat > ./airflow/dags/test_dag.py << 'EOD'
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'simple_test_dag',
    default_args=default_args,
    description='A simple DAG',
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    t2 = BashOperator(
        task_id='echo_hello',
        bash_command='echo "Hello from Airflow!"',
    )

    t1 >> t2
EOD

# Set proper permissions
sudo chown 50000:50000 ./airflow/dags/test_dag.py
sudo chmod 644 ./airflow/dags/test_dag.py

# Ensure proper ownership for mushroom_etl_dag.py
if [ -f ./airflow/dags/mushroom_etl_dag.py ]; then
    echo -e "${YELLOW}Setting correct permissions for mushroom_etl_dag.py...${NC}"
    sudo chown 50000:50000 ./airflow/dags/mushroom_etl_dag.py
    sudo chmod 644 ./airflow/dags/mushroom_etl_dag.py
fi

# Start containers
echo -e "${YELLOW}Starting all containers...${NC}"
docker compose up -d

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Fix Airflow configurations inside the container
echo -e "${YELLOW}Fixing Airflow configuration inside the container...${NC}"
docker compose exec airflow-scheduler bash -c '
# Update dags_folder path
sed -i "s|^dags_folder = .*$|dags_folder = /app/airflow/dags|g" /app/airflow/airflow.cfg

# Fix auth backend based on Airflow version
AIRFLOW_VERSION=$(airflow version | cut -d" " -f2 | cut -d"." -f1-2)
if [[ $(echo "$AIRFLOW_VERSION >= 2.5" | bc -l) -eq 1 ]]; then
    sed -i "s/^auth_manager = .*$/auth_manager = airflow.api.auth.managers.default.DefaultAuthManager/g" /app/airflow/airflow.cfg
else
    sed -i "s/^auth_backends = .*$/auth_backends = airflow.api.auth.backend.basic_auth/g" /app/airflow/airflow.cfg
fi

# Additional fixes
sed -i "s/^load_examples = .*$/load_examples = False/g" /app/airflow/airflow.cfg
sed -i "s/^dag_dir_list_interval = .*$/dag_dir_list_interval = 30/g" /app/airflow/airflow.cfg

echo "Airflow configuration fixed!"
'

# Restart services to apply changes
echo -e "${YELLOW}Restarting Airflow services...${NC}"
docker compose restart airflow-webserver airflow-scheduler

# Wait for services to restart
echo -e "${YELLOW}Waiting for services to stabilize...${NC}"
sleep 15

# Check if DAGs are detected
echo -e "${YELLOW}Checking if DAGs are detected...${NC}"
docker compose exec airflow-scheduler airflow dags list

echo -e "${GREEN}✅ Fix completed! Access the Airflow UI at http://localhost:8080${NC}"
echo -e "${GREEN}✅ Access MLflow at http://localhost:5000${NC}"
echo -e "${GREEN}If DAGs still don't appear, check logs with: docker compose logs airflow-scheduler${NC}"
