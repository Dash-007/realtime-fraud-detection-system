"""
Batch Prediction Page - Analyze Multiple Transactions
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils import get_api_client, format_probability
from api.sample_transactions import SAMPLE_TRANSACTIONS

# Page Config
st.set_page_config(page_title="Batch Prediction", page_icon="📦", layout="wide")

st.title("Batch Transaction Analysis")
st.markdown("""
Upload a CSV file with multiple transactions or use sample data to analyze many transactions at once.
""")
st.markdown("---")

# Sidebar - Options
st.sidebar.header("batch Options")

# Sample data option
use_sample = st.sidebar.checkbox("Use Sample Data", value=True)

if use_sample:
    st.sidebar.markdown("**Sample Data Settings:**")
    n_samples = st.sidebar.slider(
        "Number of samples",
        min_value=5,
        max_value=50,
        value=10,
        help="Generate random mix of normal and fraud samples"
    )
    
# Main content
if use_sample:
    st.header("Sample Data")
    
    if st.button("Generate Sample Batch", type="primary"):
        # Create mix of sample transactions
        samples = []
        sample_types = list(SAMPLE_TRANSACTIONS.keys())
        
        for i in range(n_samples):
            # Random selection with higher probability for normal
            if i % 5 == 0: # 20% fraud/suspicious
                sample_type = np.random.choice(['fraud', 'suspicious'])
            else: # 80% normal/high_amount
                sample_type = np.random.choice(['normal', 'high_amount', 'zero_amount'])
                
            sample = SAMPLE_TRANSACTIONS[sample_type].copy()
            # Add slight variations
            sample['Amount'] *= np.random.uniform(0.8, 1.2)
            sample['Time'] += i * 100
            
            samples.append(sample)
            
        st.session_state.batch_data = pd.DataFrame(samples)
        st.success(f"Generated {n_samples} sample transactions")
        
