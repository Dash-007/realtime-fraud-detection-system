#!/bin/bash
# Startup script for Hugging Face Space

start -e

echo "Starting Fraud Detection System..."
echo "=================================="

# Wait for the model to be available
if [ ! -f "/app/models/production_model_ensemble.pkl" ]; then
    echo "Model file not found!"
    echo "Checking alternative locations..."
    find /app -name "*.pkl" -type f
fi

# Test python imports
echo "Testing Python environment..."
python -c "import fastapi; import streamlit; import sklearn; import xgboost; print('All imports successful')"

# Start supervisor (manage all services)
echo "Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf