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
        
else:
    st.header("Upload Transaction Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should contain columns: Time, Amount, V1-V28"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.batch_data = df
            st.success(f"Loaded {len(df)} transactions from file")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
# Display and analyze data
if 'batch_data' in st.session_state and st.session_state.batch_data is not None:
    df = st.session_state.batch_data
    
    st.markdown("---")
    st.header("Data Preview")
    
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Avg Amount", f"${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
        
    # Show data preview
    with st.expander("View Transaction Data", expanded=False):
        st.dataframe(df.head(20), width='stretch')
        
    # Analyze button
    st.markdown("---")
    
    if st.button("Analyze All Transactions", type="primary", width="stretch"):
        with st.spinner(f"Analyzing {len(df)} transactions..."):
            try:
                client = get_api_client()
                
                # Convert to list of dicts
                transactions = df.to_dict('records')
                
                # Make batch prediction
                results = client.predict_batch(transactions)
                client.close()
                
                # Store results
                st.session_state.batch_results = results
                
                # Create results dataframe
                results_df = pd.DataFrame([
                    {
                        'Transaction_ID': i+1,
                        'Amount': transactions[i]['Amount'],
                        'Fraud_Probability': r.fraud_probability,
                        'Is_Fraud': r.is_fraud,
                        'Risk_Level': r.risk_level,
                        'Prediction_ID': r.prediction_id
                    } for i, r in enumerate(results)
                ])
                
                st.session_state.results_df = results_df
                
                st.success(f"Analysis complete! Processed {len(results)} transactions")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Make sure the API is running")