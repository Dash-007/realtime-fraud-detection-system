"""
Monitoring Dashboard - Track Model Performance Over Time
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

st.title("Model Performance Monitoring")
st.markdown("""
Track model performance, prediction history, and system health metrics over time.
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

# Get data
if show_demo_data:
    # Map time range to days
    days_map = {
        "Last Hour": 1/24,
        "Last 24 Hours": 1,
        "Last 7 Days": 7,
        "Last 30 Days": 30
    }
    
    demo_df = generate_demo_data(days_map[time_range])
    st.session_state.monitoring_data = demo_df
    
    st.info(f"Displaying {len(demo_df):,} simulated predictions from demo data")
else:
    # Use actual prediction history from this session
    if len(st.session_state.prediction_history) > 0:
        st.session_state.monitoring_data = pd.DataFrame(st.session_state.prediction_history)
        st.success(f"Displaying {len(st.session_state.prediction_history)} predictions from this session")
    else:
        st.warning("No prediction history yet. Make some predictions to see monitoring data!")
        st.session_state.monitoring_data = None

# Display metrics if we have data
if 'monitoring_data' in st.session_state and st.session_state.monitoring_data is not None:
    df = st.session_state.monitoring_data
    
    # Overall Metrics
    st.header("Overall Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_predictions = len(df)
    fraud_detected = df['is_fraud'].sum()
    fraud_rate = fraud_detected / total_predictions if total_predictions > 0 else 0
    avg_prob = df['fraud_probability'].mean()
    avg_response = df['response_time_ms'].mean() if 'response_time_ms' in df else 0
    
    with col1:
        st.metric("Total Predictions", f"{total_predictions:,}")
    with col2:
        st.metric("Frauds Detected", fraud_detected, delta=f"{fraud_rate*100:.2f}%")
    with col3:
        st.metric("Avg Fraud Prob", f"{avg_prob*100:.1f}%")
    with col4:
        high_risk = (df['risk_level'] == 'HIGH').sum()
        st.metric("High Risk", high_risk)
    with col5:
        if avg_response > 0:
            st.metric("Avg Response", f"{avg_response:.0f}ms")
        else:
            st.metric("Avg Response", "N/A")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trends Over Time",
        "Distribution Analysis", 
        "Performance Metrics",
        "Model Health"
    ])
    
    with tab1:
        st.subheader("Prediction Trends")
        
        # Aggregate by hour for cleaner visualization
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        hourly_stats = df.groupby('hour').agg({
            'is_fraud': 'sum',
            'fraud_probability': 'mean',
            'timestamp': 'count'
        }).reset_index()
        hourly_stats.columns = ['hour', 'frauds', 'avg_probability', 'total_predictions']
        
        # Predictions over time
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['total_predictions'],
            name='Total Predictions',
            line=dict(color='#3498db', width=2)
        ))
        
        fig1.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['frauds'],
            name='Frauds Detected',
            line=dict(color='#e74c3c', width=2),
            yaxis='y2'
        ))
        
        fig1.update_layout(
            title='Predictions and Fraud Detection Over Time',
            xaxis_title='Time',
            yaxis_title='Total Predictions',
            yaxis2=dict(
                title='Frauds Detected',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Fraud probability trend
        fig2 = px.line(
            hourly_stats,
            x='hour',
            y='avg_probability',
            title='Average Fraud Probability Over Time',
            labels={'hour': 'Time', 'avg_probability': 'Avg Fraud Probability'}
        )
        
        # Add threshold line
        fig2.add_hline(
            y=0.704,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold"
        )
        
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Traffic Patterns:**")
            
            # Peak hour
            peak_hour = hourly_stats.loc[hourly_stats['total_predictions'].idxmax()]
            st.write(f"- Peak traffic: {peak_hour['total_predictions']:.0f} predictions at {peak_hour['hour'].strftime('%H:%M')}")
            
            # Average per hour
            avg_per_hour = hourly_stats['total_predictions'].mean()
            st.write(f"- Average: {avg_per_hour:.1f} predictions/hour")
        
        with col2:
            st.markdown("**Fraud Patterns:**")
            
            # Fraud rate trend
            if len(hourly_stats) > 1:
                recent_rate = hourly_stats.tail(24)['frauds'].sum() / hourly_stats.tail(24)['total_predictions'].sum()
                st.write(f"- Recent fraud rate: {recent_rate*100:.2f}%")
                st.write(f"- Total frauds: {hourly_stats['frauds'].sum():.0f}")
    
    with tab2:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = df['risk_level'].value_counts()
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Level Distribution',
                color=risk_counts.index,
                color_discrete_map={
                    'LOW': '#2ecc71',
                    'MEDIUM': '#f39c12',
                    'HIGH': '#e74c3c'
                }
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud probability histogram
            fig = px.histogram(
                df,
                x='fraud_probability',
                nbins=50,
                title='Fraud Probability Distribution',
                labels={'fraud_probability': 'Fraud Probability'}
            )
            
            fig.add_vline(
                x=0.704,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Amount analysis
        st.markdown("**Transaction Amount Analysis:**")
        
        fig = px.box(
            df,
            x='risk_level',
            y='amount',
            color='risk_level',
            title='Transaction Amount by Risk Level',
            color_discrete_map={
                'LOW': '#2ecc71',
                'MEDIUM': '#f39c12',
                'HIGH': '#e74c3c'
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("System Performance Metrics")
        
        if 'response_time_ms' in df and df['response_time_ms'].notna().any():
            # Response time statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Response", f"{df['response_time_ms'].min():.0f}ms")
            with col2:
                st.metric("Avg Response", f"{df['response_time_ms'].mean():.0f}ms")
            with col3:
                st.metric("P95 Response", f"{df['response_time_ms'].quantile(0.95):.0f}ms")
            with col4:
                st.metric("Max Response", f"{df['response_time_ms'].max():.0f}ms")
            
            # Response time distribution
            fig = px.histogram(
                df,
                x='response_time_ms',
                nbins=50,
                title='Response Time Distribution',
                labels={'response_time_ms': 'Response Time (ms)'}
            )
            
            # Add SLA lines
            fig.add_vline(x=50, line_dash="dash", line_color="green", annotation_text="Target (50ms)")
            fig.add_vline(x=200, line_dash="dash", line_color="red", annotation_text="SLA (200ms)")
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            sla_violations = (df['response_time_ms'] > 200).sum()
            sla_compliance = (1 - sla_violations / len(df)) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SLA Compliance", f"{sla_compliance:.1f}%")
            with col2:
                st.metric("SLA Violations", sla_violations)
        else:
            st.info("Response time data not available for this dataset")
        
        # API Health
        st.markdown("---")
        st.markdown("**Current API Health:**")
        
        health = check_api_health()
        
        if health['status'] == 'healthy':
            col1, col2, col3 = st.columns(3)
            
            data = health['data']
            with col1:
                st.metric("Status", data['status'].upper())
            with col2:
                st.metric("Model Version", data['model_version'])
            with col3:
                st.metric("Uptime", f"{data['uptime_seconds']:.0f}s")
        else:
            st.error(f"API Health Check Failed: {health.get('message', 'Unknown error')}")
    
    with tab4:
        st.subheader("Model Health Indicators")
        
        # Calculate health metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Prediction Quality:**")
            
            # Check for prediction distribution
            prob_std = df['fraud_probability'].std()
            prob_mean = df['fraud_probability'].mean()
            
            st.write(f"- Mean probability: {prob_mean*100:.2f}%")
            st.write(f"- Std deviation: {prob_std*100:.2f}%")
            
            # Check for healthy distribution
            if prob_std < 0.05:
                st.warning("Low variance - model might be too conservative")
            elif prob_std > 0.3:
                st.warning("High variance - check for data quality issues")
            else:
                st.success("Healthy prediction distribution")
            
            # Fraud detection rate
            fraud_rate_pct = fraud_rate * 100
            if fraud_rate_pct < 0.1:
                st.info("Very low fraud rate - expected for real-world data")
            elif fraud_rate_pct > 5:
                st.warning("High fraud rate - investigate potential data issues")
            else:
                st.success("Normal fraud detection rate")
        
        with col2:
            st.markdown("**Model Behavior:**")
            
            # Check threshold crossings
            near_threshold = ((df['fraud_probability'] > 0.65) & (df['fraud_probability'] < 0.75)).sum()
            near_threshold_pct = near_threshold / len(df) * 100
            
            st.write(f"- Predictions near threshold: {near_threshold} ({near_threshold_pct:.1f}%)")
            
            if near_threshold_pct > 10:
                st.info("Many predictions near threshold - consider threshold tuning")
            else:
                st.success("Clear decision boundary")
            
            # Risk distribution check
            low_pct = (df['risk_level'] == 'LOW').sum() / len(df) * 100
            high_pct = (df['risk_level'] == 'HIGH').sum() / len(df) * 100
            
            st.write(f"- Low risk: {low_pct:.1f}%")
            st.write(f"- High risk: {high_pct:.1f}%")
            
            if low_pct > 95:
                st.info("Mostly low-risk - expected for normal traffic")
            elif high_pct > 20:
                st.warning("High percentage of high-risk transactions")
            else:
                st.success("Balanced risk distribution")
        
        # Model metrics (from training)
        st.markdown("---")
        st.markdown("**Production Model Performance:**")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Precision", "91.9%", help="Training performance")
        with metrics_col2:
            st.metric("Recall", "80.6%", help="Training performance")
        with metrics_col3:
            st.metric("F1-Score", "85.9%", help="Training performance")
        with metrics_col4:
            st.metric("False Alarm Ratio", "0.089", help="Training performance")
        
        st.caption("*Metrics from model training on test set*")

else:
    # No data available
    st.info("""
    **No monitoring data available**
    
    **Option 1 - View Demo Data:**
    - Check "Show Demo Data" in the sidebar to see simulated historical data
    
    **Option 2 - Generate Real Data:**
    - Make predictions using the Single Prediction or Batch Prediction pages
    - Prediction history will appear here automatically
    
    **In Production:**
    - This dashboard would connect to a database storing all predictions
    - Real-time metrics would be calculated from production traffic
    - Alerts would trigger on anomalies
    """)

# Footer
st.markdown("---")
st.markdown("""
**Future Production Enhancements:**
- Connect to database for persistent storage
- Add alerting for model drift or performance degradation
- Add real-time streaming metrics
- Include business metrics (fraud prevented, false alarm costs)
""")