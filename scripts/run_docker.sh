#!/bin/bash

# Make script executable and run with: chmod +x scripts/run_docker.sh && ./scripts/run_docker.sh

echo "Starting Mushroom ETL Pipeline with Docker..."

# Build and start containers
echo "Building and starting containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check container status
echo "Container status:"
docker-compose ps

# Show logs
echo "Recent logs:"
docker-compose logs --tail=20

echo ""
echo "Services started successfully!"
echo "Access Airflow UI at: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f [service-name]"
echo "To run ETL directly: docker-compose exec mushroom-app python dags/mushroom_etl_dag.py"
