"""
Utility functions for the Streamlit dashboard
"""

import sys
from pathlib import Path
import requests
from typing import Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.client import FraudDetectionClient

# API Configuration
API_URL = "http://localhost:8000"

def get_api_client() -> FraudDetectionClient:
    """
    Get configured API client
    """
    return FraudDetectionClient(API_URL)

def check_api_health() -> Dict[str, Any]:
    """
    Check if API is running and healthy
    
    Returns:
        dict: Health status or error info
    """
    try:
        client = get_api_client()
        health = client.health_check()
        client.close()
        return {
            "status": "healthy",
            "data": health
        }
    except requests.ConnectionError:
        return {
            "status": "error",
            "message": "Cannot connect to API. Is it running on http://localhost:8000?"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        
def format_probability(prob: float) -> str:
    """
    Format probability as percentage string
    """
    return f"{prob * 100:.1f}%"

def get_risk_color(risk_level: str) -> str:
    """
    Get color for risk level
    """
    colors = {
        "LOW": "green",
        "MEDIUM": "orange",
        "HIGH": "red"
    }
    return colors.get(risk_level, "gray")

def create_sample_transaction() -> Dict:
    """
    Create a sample transaction for testing
    """
    return {
        "Time": 0.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "Amount": 100.0
    }
    
def load_sample_transactions() -> pd.DataFrame:
    """
    Load sample transactions from library.
    """
    try:
        from api.sample_transactions import SAMPLE_TRANSACTIONS
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {**tx, "type": name}
            for name, tx in SAMPLE_TRANSACTIONS.items()
        ])
        return df
    except:
        return pd.DataFrame()