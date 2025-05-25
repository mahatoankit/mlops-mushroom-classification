#!/bin/bash

echo "🔁 Waiting for database and Redis..."
sleep 10

echo "🔧 Running airflow db migrate..."
airflow db migrate

echo "👤 Creating admin user..."
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin || echo "✅ User already exists or creation failed silently."

echo "🚀 Starting Airflow webserver..."
exec airflow webserver --port 8080
