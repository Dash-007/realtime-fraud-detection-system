"""
Single Transaction Prediction Page
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils import get_api_client, format_probability, get_risk_color
from api.sample_transactions import SAMPLE_TRANSACTIONS
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Single Prediction", page_icon="🔍", layout="wide")

st.title("Single Transaction Prediction")
st.markdown("Analyze individual credit card transactions for fraud detection")
st.markdown("---")

# Sidebar - Sample Transactions
st.sidebar.header("Quick Load Samples")
st.sidebar.markdown("Load pre-configured transaction examples:")

sample_options = {
    "Normal Transaction": "normal",
    "Fraudulent Transaction": "fraud",
    "Suspicious Transaction": "suspicious",
    "High Amount Normal": "high_amount",
    "Minimal Amount": "zero_amount"
}

selected_sample = st.sidebar.selectbox(
    "Select Sample",
    options=list(sample_options.keys())
)

if st.sidebar.button("Load Sample Transaction", type="primary"):
    sample_key = sample_options[selected_sample]
    st.session_state.transaction = SAMPLE_TRANSACTIONS[sample_key].copy()
    st.success(f"Loaded: {selected_sample}")

# Initialize transaction in session state
if 'transaction' not in st.session_state:
    st.session_state.transaction = SAMPLE_TRANSACTIONS['normal'].copy()

# Main form
st.header("Transaction Details")

# Create two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    
    time = st.number_input(
        "Time (seconds since first transaction)",
        min_value=0.0,
        value=float(st.session_state.transaction['Time']),
        help="Seconds elapsed between first transaction and this one"
    )
    
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=float(st.session_state.transaction['Amount']),
        help="Transaction amount in dollars"
    )
    
    st.info("**Tip:** Fraudulent transactions often have small amounts (~$2-10)")

with col2:
    st.subheader("PCA Features")
    st.markdown("*Principal Component Analysis transformed features (V1-V28)*")
    
    # Show only key features for input
    st.markdown("**Key Fraud Indicators:**")
    
    v10 = st.number_input(
        "V10",
        value=float(st.session_state.transaction['V10']),
        help="Important: Fraud often shows very negative values"
    )
    
    v14 = st.number_input(
        "V14",
        value=float(st.session_state.transaction['V14']),
        help="Important: Fraud often shows very negative values"
    )
    
    v16 = st.number_input(
        "V16",
        value=float(st.session_state.transaction['V16']),
        help="Important: Fraud indicator"
    )
    
    v17 = st.number_input(
        "V17",
        value=float(st.session_state.transaction['V17']),
        help="Important: Fraud indicator"
    )

# Expandable section for all other features
with st.expander("All PCA Features (V1-V28)", expanded=False):
    st.markdown("*Adjust all 28 PCA-transformed features*")
    
    # Create grid layout for all features
    cols = st.columns(4)
    
    feature_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            # Skip the ones we already have inputs for
            if i in [10, 14, 16, 17]:
                feature_values[f'V{i}'] = locals()[f'v{i}']
            else:
                feature_values[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=float(st.session_state.transaction[f'V{i}']),
                    key=f'v{i}_input'
                )

# Update transaction dict
transaction = {
    "Time": time,
    "Amount": amount,
    "V1": feature_values.get('V1', st.session_state.transaction['V1']),
    "V2": feature_values.get('V2', st.session_state.transaction['V2']),
    "V3": feature_values.get('V3', st.session_state.transaction['V3']),
    "V4": feature_values.get('V4', st.session_state.transaction['V4']),
    "V5": feature_values.get('V5', st.session_state.transaction['V5']),
    "V6": feature_values.get('V6', st.session_state.transaction['V6']),
    "V7": feature_values.get('V7', st.session_state.transaction['V7']),
    "V8": feature_values.get('V8', st.session_state.transaction['V8']),
    "V9": feature_values.get('V9', st.session_state.transaction['V9']),
    "V10": v10,
    "V11": feature_values.get('V11', st.session_state.transaction['V11']),
    "V12": feature_values.get('V12', st.session_state.transaction['V12']),
    "V13": feature_values.get('V13', st.session_state.transaction['V13']),
    "V14": v14,
    "V15": feature_values.get('V15', st.session_state.transaction['V15']),
    "V16": v16,
    "V17": v17,
    "V18": feature_values.get('V18', st.session_state.transaction['V18']),
    "V19": feature_values.get('V19', st.session_state.transaction['V19']),
    "V20": feature_values.get('V20', st.session_state.transaction['V20']),
    "V21": feature_values.get('V21', st.session_state.transaction['V21']),
    "V22": feature_values.get('V22', st.session_state.transaction['V22']),
    "V23": feature_values.get('V23', st.session_state.transaction['V23']),
    "V24": feature_values.get('V24', st.session_state.transaction['V24']),
    "V25": feature_values.get('V25', st.session_state.transaction['V25']),
    "V26": feature_values.get('V26', st.session_state.transaction['V26']),
    "V27": feature_values.get('V27', st.session_state.transaction['V27']),
    "V28": feature_values.get('V28', st.session_state.transaction['V28'])
}

st.markdown("---")

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "Predict Fraud",
        type="primary",
        use_container_width=True
    )

# Make prediction
if predict_button:
    with st.spinner("Analyzing transaction..."):
        try:
            client = get_api_client()
            result = client.predict(transaction)
            client.close()
            
            st.markdown("---")
            st.header("Prediction Results")
            
            # Results display
            if result.is_fraud:
                st.error("###  FRAUD DETECTED")
            else:
                st.success("###  LEGITIMATE TRANSACTION")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Fraud Probability",
                    format_probability(result.fraud_probability),
                    help="Likelihood this transaction is fraudulent"
                )
            
            with col2:
                st.metric(
                    "Risk Level",
                    result.risk_level,
                    help="Risk category: LOW, MEDIUM, or HIGH"
                )
            
            with col3:
                st.metric(
                    "Decision Threshold",
                    f"{result.threshold_used:.3f}",
                    help="Model threshold for classification"
                )
            
            with col4:
                st.metric(
                    "Confidence",
                    result.confidence,
                    help="Human-readable confidence level"
                )
            
            # Visual gauge
            st.markdown("### Risk Gauge")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result.fraud_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)"},
                delta={'reference': result.threshold_used * 100},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result.threshold_used * 100
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.markdown("###  Interpretation")
            
            if result.fraud_probability > result.threshold_used:
                st.warning(f"""
                **Action Required:** This transaction exceeds the fraud threshold of {result.threshold_used:.1%}.
                
                **Recommended Action:** 
                - **HIGH risk** ({result.fraud_probability:.1%}): Block transaction and alert security
                - Review transaction details
                - Contact customer for verification
                """)
            else:
                st.info(f"""
                **No Action Required:** This transaction is below the fraud threshold of {result.threshold_used:.1%}.
                
                **Status:**
                - Transaction appears legitimate
                - Fraud probability: {result.fraud_probability:.1%}
                - Safe to proceed
                """)
            
            # Transaction summary
            st.markdown("###  Transaction Summary")
            
            summary_data = {
                "Field": ["Amount", "Time", "V10", "V14", "V16", "V17"],
                "Value": [
                    f"${transaction['Amount']:.2f}",
                    f"{transaction['Time']:.0f}s",
                    f"{transaction['V10']:.3f}",
                    f"{transaction['V14']:.3f}",
                    f"{transaction['V16']:.3f}",
                    f"{transaction['V17']:.3f}"
                ],
                "Note": [
                    "Transaction amount",
                    "Time since first transaction",
                    "Fraud indicator" if transaction['V10'] < -2 else "Normal range",
                    "Fraud indicator" if transaction['V14'] < -2 else "Normal range",
                    "Fraud indicator" if transaction['V16'] < -2 else "Normal range",
                    "Fraud indicator" if transaction['V17'] < -2 else "Normal range"
                ]
            }
            
            import pandas as pd
            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Request ID for tracking
            st.markdown(f"**Request ID:** `{result.prediction_id}`")
            st.caption("Use this ID to trace the prediction in logs")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Make sure the API is running: `docker-compose up`")

# Footer
st.markdown("---")
st.markdown("""
**Next Steps:**
- Try different sample transactions from the sidebar
- Use the **SHAP Explainer** page to understand why the model made its decision
- Process multiple transactions with **Batch Prediction**
""")