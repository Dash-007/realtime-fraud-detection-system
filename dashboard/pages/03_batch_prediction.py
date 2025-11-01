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
                
# Display results
if 'results_df' in st.session_state and st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    
    st.markdown("---")
    st.header("Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    fraud_count = results_df['Is_Fraud'].sum()
    high_risk = (results_df['Risk_Level'] == 'HIGH').sum()
    medium_risk = (results_df['Risk_Level'] == 'MEDIUM').sum()
    avg_prob = results_df['Fraud_Probability'].mean()
    
    with col1:
        st.metric(
            "Frauds Detected",
            fraud_count,
            delta=f"{fraud_count/len(results_df)*100:.1f}%"
        )
    with col2:
        st.metric("High Risk", high_risk)
    with col3:
        st.metric("Medium Risk", medium_risk)
    with col4:
        st.metric("Avg Fraud Probability", f"{avg_prob*100:.1f}%")
        
    st.markdown("---")
    
    # Visualizations
    st.subheader("Visual Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Distribution",
        "Probability Distribution",
        "Amount vs Risk",
        "Detailed Results"
    ])
    
    with tab1:
        # Risk level pie chart
        risk_counts = results_df['Risk_Level'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            hole=0.3
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level summary
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Risk Breakdown:**")
            for level in ['LOW', 'MEDIUM', 'HIGH']:
                count = (results_df['Risk_Level'] == level).sum()
                pct = count / len(results_df) * 100
                st.write(f"- {level}: {count} ({pct:.1f}%)")
                
        with col2:
            st.markdown("**Recommendations:**")
            if high_risk > 0:
                st.warning(f"{high_risk} high-risk transactions require immediate review")
            if medium_risk > 0:
                st.info(f"{medium_risk} medium-risk transactions should be monitored")
            if fraud_count == 0:
                st.success("No fraudulent transactions detected")
                
    with tab2:
        # Fraud probability histogram
        fig = px.histogram(
            results_df,
            x='Fraud_Probability',
            nbins=30,
            title='Fraud Probability Distribution',
            labels={'Fraud_Probability': 'Fraud Probability', 'count': 'Number of Transactions'}
        )
        
        # Add threshold line
        fig.add_vline(
            x=0.704,
            line_dash='dash',
            line_color='red',
            annotation_text='threshold (0.704)'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("**Probability Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minimum", f"{results_df['Fraud_Probability'].min()*100:.1f}%")
        with col2:
            st.metric("Average", f"{results_df['Fraud_Probability'].mean()*100:.1f}%")
        with col3:
            st.metric("Maximum", f"{results_df['Fraud_Probability'].max()*100:.1f}%")
    
    with tab3:
        # Amount vs Risk scatter plot
        fig = px.scatter(
            results_df,
            x='Amount',
            y='Fraud_Probability',
            color='Risk_Level',
            color_discrete_map={'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'},
            title='Transaction Amount vs Fraud Probability',
            labels={
                'Amount': 'Transaction Amount ($)',
                'Fraud_Probability': 'Fraud Probability'
            },
            hover_data=['Transaction_ID']
        )
        
        # Add threshold line
        fig.add_hline(
            y=0.704,
            line_dash="dash",
            line_color="red",
            annotation_text="Fraud Threshold"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("**Insights:**")
        
        # High amount frauds
        high_amount_frauds = results_df[
            (results_df['Is_Fraud'] == True) & 
            (results_df['Amount'] > results_df['Amount'].median())
        ]
        
        if len(high_amount_frauds) > 0:
            st.warning(f"{len(high_amount_frauds)} high-value fraudulent transactions detected!")
            
        # Small amounts frauds
        small_amount_frauds = results_df[
            (results_df['Is_Fraud'] == True) & (results_df['Amount'] < 10)
        ]
        
        if len(small_amount_frauds) > 0:
            st.info(f"{len(small_amount_frauds)} small-amount frauds detected (classic fraud pattern)")
            
    with tab4:
        # Detailed results table
        st.markdown("**Complete Results:**")
        
        # Add color colding
        def highlight_fraud(row):
            if row['Is_Fraud']:
                return ['background-color: #ffebee'] * len(row)
            elif row['Risk_Level'] == 'HIGH':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return [''] * len(row)
            
        # Format dataframe
        display_df = results_df.copy()
        display_df['Fraud_Probability'] = display_df['Fraud_Probability'].apply(
            lambda x: f"{x*100:.2f}%"
        )
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            display_df.style.apply(highlight_fraud, axis=1),
            width='stretch',
            height=400
        )
        
        # DOwnload results
        st.markdown("---")
        st.markdown("**Download Results:**")
        
        # Create downloadable CSV
        csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Create detailed report
        col1, col2 = st.columns(2)
        
        with col1:
            if fraud_count > 0:
                st.markdown("**Flagged Transactions:**")
                flagged = results_df[results_df['Is_Fraud'] == True][
                    ['Transaction_ID', 'Amount', 'Fraud_Probability']]
                st.dataframe(flagged, width='stretch', hide_index=True)
                
        with col2:
            if medium_risk + high_risk > 0:
                st.markdown("**High-Risk Transactions:**")
                high_risk_tx = results_df[results_df['Risk_Level'].isin(['MEDIUM', 'HIGH'])][
                    ['Transaction_ID', 'Amount', 'Risk_Level', 'Fraud_Probability']].head(10)
                st.dataframe(high_risk_tx, width='stretch', hide_index=True)
                
else:
    # Instructions
    st.info("""
    **Get Started:**
    
    1.  **Option A - Use Sample Data:**
        - Check "Use Sample Data" in sidebar
        - Adjust number of samples
        - Click "Generate Sample Batch"
        
    2.  **Option B - Upload Your Data:**
        - Uncheck "Use Sample Data"
        - Upload CSV file with columns: Time, Amount, V1-V28
        - Click "Analyze All Transactions"
    """)
    
    # Show expected format
    with st.expander("Expected CSV Format", expanded=False):
        st.markdown("""
Your CSV file should have these columns:
- 'Time': Seconds since first transaction (float)
- 'Amount': Transaction amount in dollars (float)
- 'V1' through 'V28': PCA-transformed features (float)
        
```
Time,Amount,V1,V2,V3,...,V28
0.0,149.62,-1.359807,-0.072781,2.536347,...,-0.021053
1.0,2.69,1.191857,0.266151,0.166480,...,0.014724
```
    """)
        
# Footer
st.markdown("---")
st.markdown("""
**Tips:**
- Use sample data to quickly test the batch analysis
- Download results for further analysis in Excel/Python
- High-risk transactions should be reviewed manually
- Monitor patterns across multiple batches
""")