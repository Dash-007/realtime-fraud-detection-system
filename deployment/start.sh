#!/bin/bash
set -e

echo "Starting Fraud Detection System..."

export HOME=/app
export STREAMLIT_SERVER_HEADLESS=true

python3 -c "import fastapi, streamlit, sklearn, xgboost; print('Imports OK')"

if [ -f "/app/models/production_model_ensemble.pkl" ]; then
    echo "Model found"
fi

# Start API in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
sleep 5

# Start Streamlit on port 7860 (foreground)
exec streamlit run dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false