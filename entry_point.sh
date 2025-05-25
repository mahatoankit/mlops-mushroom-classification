#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Default AIRFLOW_HOME to /app/airflow if not set
: "${AIRFLOW_HOME:="/app/airflow"}"
export AIRFLOW_HOME

# Default AIRFLOW_PORT to 8080 if not set (though we exec with it anyway)
: "${AIRFLOW_PORT:="8080"}"

echo "💨 Initializing folders and permissions for AIRFLOW_HOME: ${AIRFLOW_HOME}"
# Create standard Airflow subdirectories if they don't exist
# This is important because these paths might be volume mounts from the host
# and might not exist or have correct ownership initially.
mkdir -p "${AIRFLOW_HOME}/dags" \
         "${AIRFLOW_HOME}/logs" \
         "${AIRFLOW_HOME}/plugins"
# Ensure the airflow user owns these directories.
# This is run as the airflow user (due to USER airflow in Dockerfile)
# If it were run as root, we'd use `chown -R airflow:airflow "${AIRFLOW_HOME}"`
# but since we are airflow, mkdir should create them with our ownership.
# However, if the mount point itself from the host has restrictive root ownership,
# the airflow user might not be able to `mkdir` or `chown` it.
# The chown in the Dockerfile should have handled /app, but mounts can be tricky.
# Let's try to ensure ownership of the specific subdirectories if they were mounted from host by root.
# This is a bit of a dance if USER is airflow. Ideally, these are owned by airflow from the start.

# The `chown -R airflow /app` in Dockerfile (run as root) should have set this.
# The issue is if the *mounted volume itself* (e.g. ./airflow/logs on host) is owned by host's root.
# The container's airflow user might not have rights then.
# A common solution if the user inside the container is known (e.g. airflow UID 50000)
# is to chown the host directories to that UID.
# Or, as a last resort in entrypoint if running as root:
# chown -R airflow:airflow "${AIRFLOW_HOME}/dags" "${AIRFLOW_HOME}/logs" "${AIRFLOW_HOME}/plugins"
# But we are running as 'airflow' user here.

# The most likely fix is that the *parent* directory /app/airflow needs to be writable by airflow user
# for it to create subdirs like /app/airflow/logs/scheduler.
# The chown -R airflow /app in the Dockerfile *should* have handled this.

# Let's ensure the base logs directory is writable by trying to create a test file.
if touch "${AIRFLOW_HOME}/logs/.airflow_can_write" 2>/dev/null ; then
    echo "✅ Write access to ${AIRFLOW_HOME}/logs confirmed."
    rm "${AIRFLOW_HOME}/logs/.airflow_can_write"
else
    echo "❌ ERROR: No write access to ${AIRFLOW_HOME}/logs for user $(whoami) (UID $(id -u))!"
    echo "This is likely due to host volume mount permissions."
    echo "Please ensure the host directory mounted to ${AIRFLOW_HOME}/logs is writable by UID $(id -u)."
    # As a WAITING strategy to allow manual fix from another terminal:
    # echo "Waiting for 60 seconds to allow manual permission fix on host volumes..."
    # sleep 60
    exit 1 # Exit if we can't write, as Airflow will fail.
fi


# Function to wait for PostgreSQL
wait_for_postgres() {
    echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-postgres}:${POSTGRES_PORT:-5432}..."
    local pg_host="${POSTGRES_HOST:-postgres}"
    local pg_port="${POSTGRES_PORT:-5432}"
    local pg_user="${POSTGRES_USER:-airflow}"
    local pg_db="${POSTGRES_DB:-airflow}"
    local attempt_num=0
    local max_attempts=30
    while ! pg_isready -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -q; do
        attempt_num=$((attempt_num+1))
        if [ "$attempt_num" -ge "$max_attempts" ]; then
            echo "PostgreSQL did not become ready after $max_attempts attempts. Exiting."
            exit 1
        fi
        echo "PostgreSQL not ready (attempt $attempt_num/$max_attempts), waiting 2 seconds..."
        sleep 2
    done
    echo "✅ PostgreSQL is ready."
}

# Function to wait for Redis
wait_for_redis() {
    echo "Waiting for Redis at ${REDIS_HOST:-redis}:${REDIS_PORT:-6379}..."
    local redis_host="${REDIS_HOST:-redis}"
    local redis_port="${REDIS_PORT:-6379}"
    local attempt_num=0
    local max_attempts=30
    while ! redis-cli -h "$redis_host" -p "$redis_port" ping | grep -q "PONG"; do
        attempt_num=$((attempt_num+1))
        if [ "$attempt_num" -ge "$max_attempts" ]; then
            echo "Redis did not become ready after $max_attempts attempts. Exiting."
            exit 1
        fi
        echo "Redis not ready (attempt $attempt_num/$max_attempts), waiting 2 seconds..."
        sleep 2
    done
    echo "✅ Redis is ready."
}


echo "🔁 Waiting for dependencies..."
wait_for_postgres
wait_for_redis

if [ "$1" = "webserver" ] || [ "$1" = "scheduler" ]; then
    echo "🔧 Running airflow db migrate..."
    airflow db migrate || { echo "❌ DB migration failed."; exit 1; }
    echo "✅ Database migration successful or already up-to-date."
fi

if [ "$1" = "webserver" ]; then
    echo "👤 Creating/Verifying admin user..."
    if ! airflow users list | grep -q "admin@example.com"; then
        echo "Admin user not found, creating..."
        airflow users create \
          --username admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com \
          --password admin
        echo "✅ Admin user created."
    else
        echo "✅ Admin user 'admin' (admin@example.com) already exists."
    fi
fi

echo "🚀 Starting Airflow component: $@"
exec airflow "$@"