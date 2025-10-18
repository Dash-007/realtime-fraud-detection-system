"""
SHAP Explainability Page - Model Interpretability
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils import get_api_client, format_probability
from api.sample_transactions import SAMPLE_TRANSACTIONS

# Page config
st.set_page_config(page_title="SHAP Explainer", page_icon="🧠", layout="wide")

st.title("Model Explainability with SHAP")
st.markdown("""
Understand **why** the model made its prediction using SHAP (SHapely Additive exPlanations).

SHAP values show how much each feature contributed to pushing the prediction toward fraud or legitimate.
""")
st.markdown("---")

# Load model
@st.cache_resource
def load_model_and_explainer():
    """
    Load model and create SHAP explainer (cached for performance)
    """
    try:
        # Load the model package
        model_path = Path(__file__).parent.parent.parent / "models" / "production_model_ensemble.pkl"
        model_package = joblib.load(model_path)
        
        # Extract components
        ensemble_model = model_package['ensemble_model']
        scaler = model_package['scaler']
        feature_engineer = model_package['feature_engineer']
        feature_names = model_package['feature_name']
        
        # Create SHAP explainer
        # Using TreeExplainer for tree-based models (RF, XGBoost)
        explainer = SHAP.TreeExplainer(ensemble_model)
        
        return {
            'model': ensemble_model,
            'scaler': scaler,
            'engineer': feature_engineer,
            'explainer': explainer,
            'feature_names': feature_names
        }
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# Load model components
with st.spinner("Loading model and SHAP explainer..."):
    components = load_model_and_explainer()
    
if components is None:
    st.error("Failed to load model. Please check that the model file exists.")
    st.stop()
    
st.success("Model and SHAP explainer loaded successfully")

# Sidebar - Sample selection
st.sidebar.header("Select Transaction")
st.sidebar.markdown("Choose a sample to transaction to explain:")

sample_options = {
    "Normal Transaction": "normal",
    "Fraudulent Transaction": "fraud",
    "Suspicious Transaction": "suspicious",
    "High Amount Normal": "high_amount"
}

selected_sample = st.sidebar.selectbox(
    "Sample Type",
    options=list(sample_options.keys())
)

if st.sidebar.button("Load Sample", type="primary"):
    sample_key = sample_options[selected_sample]
    st.session_state.explain_transaction = SAMPLE_TRANSACTIONS[sample_key].copy()
    st.success(f"Loaded: {selected_sample}")
    
# Initialize
if 'explain_transaction' not in st.session_state:
    st.session_state.explain_transaction = SAMPLE_TRANSACTIONS['fraud'].copy()
    
transaction = st.session_state.explain_transaction

# Display transaction info
st.header("Transaction Details")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Amount", f"${transaction['Amount']:.2f}")
with col2:
    st.metric("Time", f"{transaction['Time']:.0f}s")
with col3:
    # Highlight key fraud indicators
    key_features = ['V10', 'V14', 'V16', 'V17']
    negative_count = sum(1 for f in key_features if transaction[f] < -2)
    st.metric("High-Risk Features", f"{negative_count}/4")
    
st.markdown("---")
