# dashboard/app.py
"""
Fraud Detection System - Main Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from dashboard.utils import check_api_health

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .danger-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚨 Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Introduction
st.markdown("""
### Welcome to the Real-Time Fraud Detection Dashboard

This dashboard provides an interactive interface for our fraud detection system, which uses an ensemble 
of machine learning models to identify fraudulent credit card transactions in real-time.

####  Key Features:
- **Single Transaction Prediction**: Analyze individual transactions with instant results
- **SHAP Explainability**: Understand why the model made its decision
- **Batch Processing**: Analyze multiple transactions at once
- **Performance Monitoring**: Track model performance over time

####  Model Performance:
""")

# Display model performance metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Precision", "91.9%", help="When we flag fraud, we're right 92% of the time")
with col2:
    st.metric("Recall", "80.6%", help="We catch 81% of all fraud")
with col3:
    st.metric("F1-Score", "85.9%", help="Harmonic mean of precision and recall")
with col4:
    st.metric("False Alarm Ratio", "0.089", help="Only 0.089 false alarms per fraud caught")

st.markdown("---")

# API Health Check
st.markdown("###  System Status")

health_status = check_api_health()

if health_status["status"] == "healthy":
    with st.container():
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("API is healthy and ready to process transactions")
        
        data = health_status["data"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", data["status"].upper())
        with col2:
            st.metric("Model Version", data["model_version"])
        with col3:
            st.metric("Uptime", f"{data['uptime_seconds']:.1f}s")
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error(f"{health_status['message']}")
    st.info("Make sure the API is running: `docker-compose up` or `uvicorn api.main:app --reload`")

st.markdown("---")

# Navigation Guide
st.markdown("""
###  Navigation

Use the sidebar to navigate between different sections:

1. ** Single Prediction** - Analyze individual transactions
2. ** SHAP Explainer** - See how the model makes decisions
3. ** Batch Prediction** - Process multiple transactions
4. ** Monitoring** - View performance metrics and history

---

###  Getting Started

1. Ensure the API is running (check status above)
2. Select a page from the sidebar
3. Input transaction data or use sample transactions
4. View instant predictions with explanations

**Need help?** Check the documentation at [http://localhost:8000/docs](http://localhost:8000/docs)
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with Streamlit, FastAPI, and ❤️</p>
    <p>Model: Ensemble (Random Forest + XGBoost) | Threshold: 0.704</p>
</div>
""", unsafe_allow_html=True)