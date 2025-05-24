#!/usr/bin/env python3

import os
import subprocess
import sys

def check_airflow_connection():
    """Check if Airflow is properly connected and can see DAGs"""
    print("🔍 Checking Airflow DAG Connection")
    print("=" * 40)
    
    # Set Airflow environment
    airflow_home = os.path.join(os.getcwd(), "airflow")
    os.environ["AIRFLOW_HOME"] = airflow_home
    
    print(f"📂 AIRFLOW_HOME: {airflow_home}")
    print(f"📂 DAGs directory: {airflow_home}/dags")
    
    # Check if DAGs directory exists and has files
    dags_dir = os.path.join(airflow_home, "dags")
    if os.path.exists(dags_dir):
        dag_files = [f for f in os.listdir(dags_dir) if f.endswith('.py')]
        print(f"✅ DAGs directory exists with {len(dag_files)} Python files:")
        for dag_file in dag_files:
            print(f"   - {dag_file}")
    else:
        print("❌ DAGs directory not found")
        return False
    
    # Copy DAG to Airflow directory if needed
    source_dag = "dags/mushroom_etl_dag.py"
    target_dag = os.path.join(dags_dir, "mushroom_etl_dag.py")
    
    if os.path.exists(source_dag):
        print(f"📋 Copying DAG from {source_dag} to {target_dag}")
        import shutil
        shutil.copy2(source_dag, target_dag)
        print("✅ DAG copied to Airflow directory")
    else:
        print(f"❌ Source DAG not found: {source_dag}")
    
    # Try to list DAGs using Airflow CLI
    try:
        print("\n🔄 Checking DAGs with Airflow CLI...")
        result = subprocess.run(
            ["airflow", "dags", "list"], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Airflow CLI is working")
            dags_output = result.stdout
            
            if "mushroom_etl_pipeline" in dags_output:
                print("✅ mushroom_etl_pipeline DAG found!")
            else:
                print("❌ mushroom_etl_pipeline DAG not found in Airflow")
                print("Available DAGs:")
                for line in dags_output.split('\n'):
                    if line.strip() and not line.startswith('dag_id'):
                        print(f"   - {line.strip()}")
        else:
            print(f"❌ Airflow CLI error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Airflow CLI command timed out")
    except Exception as e:
        print(f"❌ Error running Airflow CLI: {e}")
    
    # Check Airflow webserver accessibility
    try:
        print("\n🌐 Checking Airflow webserver...")
        import requests
        response = requests.get("http://localhost:8080", timeout=5)
        
        if response.status_code == 200:
            print("✅ Airflow webserver is accessible")
            print("🔗 URL: http://localhost:8080")
            print("👤 Username: admin")
            print("🔑 Password: admin (or check the generated password file)")
        else:
            print(f"⚠️ Webserver returned status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cannot access webserver: {e}")
        print("💡 Make sure Airflow is running: airflow standalone")
    
    # Check if processes are running
    print("\n🔧 Checking Airflow processes...")
    try:
        result = subprocess.run(["pgrep", "-f", "airflow"], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Found {len(pids)} Airflow process(es) running")
            for pid in pids:
                if pid.strip():
                    # Get process details
                    proc_result = subprocess.run(
                        ["ps", "-p", pid.strip(), "-o", "pid,cmd"], 
                        capture_output=True, text=True
                    )
                    if proc_result.returncode == 0:
                        print(f"   PID {pid.strip()}: {proc_result.stdout.split('\n')[1] if len(proc_result.stdout.split('\n')) > 1 else 'Unknown'}")
        else:
            print("❌ No Airflow processes found")
            print("💡 Start Airflow with: airflow standalone")
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

if __name__ == "__main__":
    check_airflow_connection()
