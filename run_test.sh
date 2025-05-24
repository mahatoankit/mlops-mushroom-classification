#!/bin/bash

echo "🧪 Testing the MLOps pipeline..."

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Test the training pipeline
python test_training.py

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    echo "🚀 You can now run the full pipeline with:"
    echo "   python src/pipeline.py --config config/config.yaml"
    echo "   OR"
    echo "   python dags/mushroom_etl_dag.py"
else
    echo "❌ Tests failed. Please check the errors above."
    exit 1
fi
