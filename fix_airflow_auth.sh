#!/bin/bash

echo "🔑 Fixing Airflow Authentication"
echo "================================"

# Set Airflow environment
export AIRFLOW_HOME="$(pwd)/airflow"
echo "🔧 Airflow Home: $AIRFLOW_HOME"

# Create directories if they don't exist
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/logs" "$AIRFLOW_HOME/plugins"

# Set Airflow configuration
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
export AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
export AIRFLOW__WEBSERVER__SECRET_KEY="mushroom_classification_secret"
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"

echo "🔄 Initializing Airflow database..."
airflow db init

echo "🗑️ Removing existing admin user (if any)..."
airflow users delete --username admin 2>/dev/null || echo "   No existing user found"

echo "👤 Creating new admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Airflow authentication fixed!"
    echo "   Username: admin"
    echo "   Password: admin"
    echo "   URL: http://localhost:8080"
    echo ""
    
    # Check if webserver is already running
    if pgrep -f "airflow webserver" > /dev/null; then
        echo "ℹ️ Airflow webserver already running"
    else
        echo "🌐 Starting Airflow webserver..."
        airflow webserver --port 8080 --daemon
        sleep 3
    fi
    
    # Check if scheduler is already running
    if pgrep -f "airflow scheduler" > /dev/null; then
        echo "ℹ️ Airflow scheduler already running"
    else
        echo "📅 Starting Airflow scheduler..."
        airflow scheduler --daemon
        sleep 3
    fi
    
    echo ""
    echo "🔍 Service Status:"
    if pgrep -f "airflow webserver" > /dev/null; then
        webserver_pid=$(pgrep -f "airflow webserver")
        echo "   ✅ Webserver running (PID: $webserver_pid)"
    else
        echo "   ❌ Webserver not running"
        echo "   💡 Start manually: airflow webserver --port 8080 --daemon"
    fi
    
    if pgrep -f "airflow scheduler" > /dev/null; then
        scheduler_pid=$(pgrep -f "airflow scheduler")
        echo "   ✅ Scheduler running (PID: $scheduler_pid)"
    else
        echo "   ❌ Scheduler not running" 
        echo "   💡 Start manually: airflow scheduler --daemon"
    fi
    
    echo ""
    echo "🔗 Access Airflow at: http://localhost:8080"
    echo "📋 To view DAGs: curl -s http://localhost:8080/api/v1/dags"
    echo "🛑 To stop services: pkill -f airflow"
    
else
    echo "❌ Failed to create admin user"
    exit 1
fi
