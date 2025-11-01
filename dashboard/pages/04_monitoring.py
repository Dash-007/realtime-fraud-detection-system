"""
Monitoring Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils import check_api_health

# Page config
st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")
st.markdown("""
Track model performance, prediction history, and system health metrics over time
""")
st.markdown("---")

# Initialize session state for tracking predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
    
# Sidebar - Settings
st.sidebar.header("Monitoring Settings")

show_demo_data = st.sidebar.checkbox(
    "Show Demo Data",
    value=True,
    help="Display simulated historical data for demonstration"
)

time_range = st.sidebar.selectbox(
    "Time Range",
    options=["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
    index=1
)

# Generate demo data
def generate_demo_data(n_days=7):
    """Generate simulated prediction history for demonstration"""
    
    np.random.seed(42)  # Reproducible demo data
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=n_days)
    
    # Generate hourly predictions
    hours = int((end_time - start_time).total_seconds() / 3600)
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    predictions = []
    
    for ts in timestamps:
        # Simulate varying traffic patterns
        hour = ts.hour
        
        # More transactions during business hours
        if 9 <= hour <= 17:
            n_predictions = np.random.poisson(50)
        else:
            n_predictions = np.random.poisson(20)
        
        # Generate predictions for this hour
        for _ in range(n_predictions):
            # 85% normal, 10% suspicious, 5% fraud
            rand = np.random.random()
            
            if rand < 0.85:
                prob = np.random.beta(2, 10)  # Low probability
                risk = "LOW"
            elif rand < 0.95:
                prob = np.random.beta(5, 5)   # Medium probability
                risk = "MEDIUM" if prob < 0.704 else "HIGH"
            else:
                prob = np.random.beta(8, 2)   # High probability
                risk = "HIGH"
            
            is_fraud = prob >= 0.704
            
            predictions.append({
                'timestamp': ts + timedelta(minutes=np.random.randint(0, 60)),
                'fraud_probability': prob,
                'is_fraud': is_fraud,
                'risk_level': risk,
                'amount': np.random.exponential(50) + 1,
                'response_time_ms': np.random.gamma(2, 10)
            })
    
    return pd.DataFrame(predictions)