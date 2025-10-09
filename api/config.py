"""
API Configuration
"""
import os
from pathlib import Path

# API Settings
API_TITLE = "Fraud Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Real-Time Credit Card Fraud Detection API

This API provides real-time fraud detection for credit card transactions using an ensemble machine learning model.

### Features:
- Single transaction prediction
- Batch prediction (up to 100 transactions)
- Optimized threshold for high precision
- Risk level assessment

### Model Performance:
- Precision: 91.9%
- Recall: 80.6%
- F1_score: 85.9%
"""

# Model settings
MODEL_PATH = Path("models/production_model_ensemble.pkl")
MODEL_VERSION = "ensemble_v1"

# Performance settings
MAX_BATCH_SIZE = 100
REQUEST_TIMEOUT = 30

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOF_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"