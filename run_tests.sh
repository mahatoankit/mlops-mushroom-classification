#!/bin/bash

# Run all unit tests for the mushroom ETL pipeline
echo "Running mushroom ETL pipeline tests..."

# Ensure the test data directory exists
mkdir -p data/test

# Run all tests with pytest
python -m pytest tests/ -v

# Check the exit status
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Please check the output above."
    exit 1
fi
