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

# Sidebar - Model selection
st.sidebar.header("Explanation Settings")


explain_option = st.sidebar.radio(
    "Explain which model?",
    options=["Random Forest (Fast)", "XGBoost (Fast)", "Full Ensemble (Slow)"],
    help="Individual models are faster. Full Ensemble uses KernelExplainer (~30 seconds)."
)

# Load model
@st.cache_resource
def load_model_and_explainer(explain_option):
    """Load model and create SHAP explainer (cached for performance)"""
    try:
        # Load the model package
        model_path = Path(__file__).parent.parent.parent / "models" / "production_model_ensemble.pkl"
        model_package = joblib.load(model_path)
        
        # Extract components
        ensemble_model = model_package['ensemble_model']
        scaler = model_package['scaler']
        feature_engineer = model_package['feature_engineer']
        feature_names = model_package['feature_names']
        
        from sklearn.ensemble import VotingClassifier
        
        if isinstance(ensemble_model, VotingClassifier):
            individual_models = ensemble_model.named_estimators_
            
            # Select model based on user choice
            if "Random Forest" in explain_option and 'rf' in individual_models:
                model_to_explain = individual_models['rf']
                model_name = "Random Forest"
                explainer = shap.TreeExplainer(model_to_explain)
                
            elif "XGBoost" in explain_option and 'xgb' in individual_models:
                model_to_explain = individual_models['xgb']
                model_name = "XGBoost"
                
                # For XGBoost SHAP compatibility
                try:
                    booster = model_to_explain.get_booster()
                    booster.set_param({'base_score': '0.5'})
                    explainer = shap.TreeExplainer(model_to_explain)
                except Exception as e:
                    # Fallback: create dummy background for KernelExplainer
                    st.warning(f"Using slower SHAP method for XGBoost: {str(e)}")
                    
                    def predict_fn(X):
                        return model_to_explain.predict_proba(X)[:, 1]
                    
                    # Create simple background (neutral values)
                    n_features = len(feature_names)
                    background = np.zeros((10, n_features))
                    explainer = shap.KernelExplainer(predict_fn, background)
                
            elif "Full Ensemble" in explain_option:
                model_to_explain = ensemble_model
                model_name = "Full Ensemble (VotingClassifier)"
                
                # For VotingClassifier, we need to use KernelExplainer
                # Create prediction function
                def predict_fn(X):
                    """Prediction function for KernelExplainer"""
                    return ensemble_model.predict_proba(X)[:, 1]
                
                # Create background dataset
                # Use zeros as a simple baseline (represents "neutral" transaction)
                n_features = len(feature_names)
                background = np.zeros((1, n_features))
                
                # Create KernelExplainer
                explainer = shap.KernelExplainer(predict_fn, background)
                
            else:
                # Fallback to first available
                model_to_explain = list(individual_models.values())[0]
                model_name = list(individual_models.keys())[0]
                explainer = shap.TreeExplainer(model_to_explain)
        else:
            model_to_explain = ensemble_model
            model_name = "Model"
            explainer = shap.TreeExplainer(model_to_explain)
        
        return {
            'model': ensemble_model,
            'explain_model': model_to_explain,
            'model_name': model_name,
            'scaler': scaler,
            'engineer': feature_engineer,
            'explainer': explainer,
            'feature_names': feature_names,
            'is_ensemble': "Full Ensemble" in explain_option
        }
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Load model components
with st.spinner("Loading model and SHAP explainer..."):
    components = load_model_and_explainer(explain_option)
    
if components is None:
    st.error("Failed to load model. Please check that the model file exists.")
    st.stop()
    
st.success(f"{components['model_name']} loaded successfully")
st.info(f"Explaining predictions using {components['model_name']} component")

# Sidebar - Sample Selection
st.sidebar.markdown("---")
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

# Analyze button
if st.button("Explain Prediction", type="primary", width="stretch"):
    with st.spinner("Computing SHAP values..."):
        try:
            # Prepare features
            df = pd.DataFrame([transaction])
            df_engineered = components['engineer'].transform(df)
            X_prepared = df_engineered[components['feature_names']]
            X_scaled = components['scaler'].transform(X_prepared)
            
            # Get prediction
            pred_proba = components['model'].predict_proba(X_scaled)[0, 1]
            
            # Calculate SHAP values using individual model
            if components.get('is_ensemble', False):
                # KernelExplainer for full ensemble (slower)
                with st.spinner("Computing SHAP values with KernelExplainer (30-60 seconds)..."):
                    shap_values = components['explainer'].shap_values(X_scaled, nsamples=100)
                
                # KernelExplainer returns 1D array for single class probability
                if len(shap_values.shape) > 1:
                    shap_values = shap_values.flatten()
                
                # Base value for KernelExplainer
                base_value = components['explainer'].expected_value
                
            else:
                # TreeExplainer for individual models (fast)
                shap_values = components['explainer'].shap_values(X_scaled)
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get fraud class
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]
                elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                    shap_values = shap_values[:, 1]
                
                # Ensure 1D array
                if len(shap_values.shape) > 1:
                    shap_values = shap_values.flatten()
                
                # Get base value
                base_value = components['explainer'].expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if len(base_value) > 1:
                        base_value = base_value[1]
                    else:
                        base_value = base_value[0]
                
            st.markdown("---")
            st.header("Prediction Results")
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", format_probability(pred_proba), help="Full ensemble prediction")
                
            with col2:
                classification = "FRAUD" if round(pred_proba, 3) >= 0.704 else "LEGITIMATE"
                st.metric("Classification", classification)
                
            with col3:
                risk = "HIGH" if pred_proba > 0.8 else "MEDIUM" if pred_proba > 0.5 else "LOW"
                st.metric("Risk Level", risk)
                
            st.caption(f"*Prediction from full ensemble. SHAP explanation from {components['model_name']}.*")
            
            st.markdown("---")
            st.header("SHAP Explanation")
            
            # Create SHAP explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=X_scaled[0],
                feature_names=components['feature_names']
            )
            
            # Waterfall plot
            st.subheader("Waterfall Plot: Feature Contributions")
            st.markdown("""
            This plot shows how each feature pushes the prediction from the base value toward the final prediction.
            
            - **Red bars** push toward FRAUD (increase probability)
            - **Blue bars** push toward LEGITIMATE (decrease probability)
            - Bar length shows strenth of contribution
            """)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Feature importance plot
            st.subheader("Feature Importance: Top Contributors")
            st.markdown("Features ranked by absolute contribution to this prediction:")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            # Detailed breakdown
            st.subheader("Detailed Feature Analysis")
            
            # Dataframe for SHAP values
            shap_df = pd.DataFrame({
                'Feature': components['feature_names'],
                'Feature Value': X_scaled[0],
                'SHAP Value': shap_values,
                'Absolute Impact': np.abs(shap_values)
            }).sort_values('Absolute Impact', ascending=False)
            
            # Top 10 features
            st.markdown("**Top 10 Most Influential Features:**")
            
            top_features = shap_df.head(10).copy()
            top_features['Impact Direction'] = top_features['SHAP Value'].apply(
                lambda x: 'Toward Fraud' if x > 0 else 'Toward Legitimate'
            )
            top_features['SHAP Value'] = top_features['SHAP Value'].round(4)
            top_features['Feature Value'] = top_features['Feature Value'].round(4)
            
            st.dataframe(
                top_features[['Feature', 'Feature Value', 'SHAP Value', 'Impact Direction']],
                width='stretch',
                hide_index=True
            )
            
            # Interpretation
            st.markdown("---")
            st.subheader("Interpretation")
            
            # Analyze top contributors
            top_fraud_features = shap_df[shap_df['SHAP Value'] > 0].head(3)
            top_legit_features = shap_df[shap_df['SHAP Value'] < 0].head(3)
            
            if len(top_fraud_features) > 0:
                st.markdown('**Key Fraud Indicators:**')
                for _, row in top_fraud_features.iterrows():
                    st.write(f"- **{row['Feature']}** (value: {row['Feature Value']:.3f}) contributes +{row['SHAP Value']:.4f} toward fraud")
                    
            if len(top_legit_features) > 0:
                st.markdown("**Key Legitimate Indicators:**")
                for _, row in top_legit_features.iterrows():
                    st.write(f"- **{row['Feature']}** (value: {row['Feature Value']:.3f}) contributes {row['SHAP Value']:.4f} toward legitimate")
                    
            # Summary
            st.info(f"""
            **Summary:**
            
            The model's base prediction (average across all data) is {format_probability(base_value)}.
            
            After considering all features, the final prediction is {format_probability(pred_proba)}.
            
            The features shown above had the strongest influence on moving the prediction from the base rate the final predicition.
            """)
            
        except Exception as e:
            st.error(f"Error computing SHAP values: {str(e)}")
            st.exception(e)
            
# Info section
st.markdown("---")
st.header("Understanding SHAP")

with st.expander("What are SHAP values?", expanded=False):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values are a unified measure of feature importance based on game theory.
    
    **Key concepts:**
    
    1. **Base Value**: The average prediction across all training data (baseline)
    
    2. **SHAP Value**: How much a feature pushed the prediction away from the base value
    - Positive SHAP = pushed toward fraud
    - Negative SHAP = pushed toward legitimate
    
    3. **Final Prediction**: Base value + sum of all SHAP values
    
    **Example:**
    - Base value: 17% (average fraud rate)
    - V14 contribution: +30% (very negative value indicates fraud)
    - V2 contribution: -5% (normal value indicates legitimate)
    - Amount contribution: +10% (small amount indicates fraud)
    - Final prediction: 17% + 30% - 5% = 10% + ... = 68%
    """)
    
with st.expander("How to interpret the visualizations", expanded=False):
    st.markdown("""
    **Waterfall Plot:**
    - Shows the cumulative effect of features
    - Starts at base value, adds contributions sequentially
    - Ends at final prediction
    - Red bars push UP (toward fraud), blue bars push DOWN (toward legitimate)
    
    **Bar Plot:**
    - Shows absolute impact of each feature
    - Longer bars = stronger influence
    - Doesn't show direction, just magnitude
    
    **Feature Tables:**
    - Lists features in order of impact
    - Shows actual feature values
    - Shows SHAP values (contributions)
    - Direction indicator (toward fraud or legitimate)
    """)
    
# Footer
st.markdown("---")
st.markdown("""
**Next Steps:**
- Try different sample transactions to see how SHAP values change
- Use **Batch Prediction** to analyze multiple transactions
- Check **Monitoring** to tract model performace over time (coming soon)
""")