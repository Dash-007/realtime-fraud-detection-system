"""
Unit tests for model loading and prediction
"""

import pytest
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import (
    get_sample_transaction,
    get_fraud_transaction,
    get_normal_transaction
)

class TestModelLoading:
    """Test model loading functionality"""
    
    @pytest.fixture
    def model_path(self):
        """Get path to prodution model"""
        return Path(__file__).parent.parent.parent / "models" / "production_model_ensemble.pkl"
    
    def test_model_file_exists(self, model_path):
        """Test that model file exists"""
        model_package = joblib.load(model_path)
        
        assert model_package is not None
        assert isinstance(model_package, dict)
        
    def test_model_has_required_components(self, model_path):
        """Test that model package has all required components"""
        model_package = joblib.load(model_path)
        
        required_keys = ['ensemble_model', 'scaler', 'feature_engineer', 'feature_names', 'optimal_threshold']
        
        for key in required_keys:
            assert key in model_package, f"Missing required key: {key}"
            
    def test_model_feature_count(self, model_path):
        """Test that model expects correct number of features"""
        model_package = joblib.load(model_path)
        feature_names = model_package['feature_names']
        
        # Should have approximately 40 features
        assert len(feature_names) >=30
        assert len(feature_names) <= 50
        
class TestModelPrediction:
    """
    Test model prediction functionality
    """
    
    @pytest.fixture
    def model_components(self):
        """Load model componenets"""
        model_path = Path(__file__).parent.parent.parent / "models" / "production_model_ensemble.pkl"
        return joblib.load(model_path)
    
    def test_predict_normal_transaction(self, model_components):
        """Test prediction on normal transaction"""
        tx = get_normal_transaction()
        df = pd.DataFrame([tx])
        
        # Engineer features
        df_eng = model_components['feature_engineer'].transform(df)
        X = df_eng[model_components['feature_names']]
        X_scaled = model_components['scaler'].transform(X)
        
        # Predict
        prob = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
        
        # Should be lower probability
        assert 0 <= prob <= 1
        assert prob < 0.5
        
    def test_predict_fraud_transaction(self, model_components):
        """Test prediction on fraudulent transaction"""
        tx = get_fraud_transaction()
        df = pd.DataFrame([tx])
        
        # Engineer features
        df_eng = model_components['feature_engineer'].transform(df)
        X = df_eng[model_components['feature_names']]
        X_scaled = model_components['scaler'].transform(X)
        
        # Predict
        prob = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
        
        # Should be high probability
        assert 0 <= prob <= 1
        assert prob > 0.3
        
    def test_prediction_consistency(self, model_components):
        """Test that predictions are consistent"""
        tx = get_fraud_transaction()
        df = pd.DataFrame([tx])
        
        # Engineer features
        df_eng = model_components['feature_engineer'].transform(df)
        X = df_eng[model_components['feature_names']]
        X_scaled = model_components['scaler'].transform(X)
        
        # Predict
        prob1 = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
        prob2 = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
        
        # Should be identical
        assert abs(prob1 - prob2) < 0.001 # Allow tiny floating point differences
        
    def test_threshold_classification(self, model_components):
        """Test that threshold classification works"""
        threshold = model_components['optimal_threshold']
        
        # Test transaction above threshold
        tx = get_fraud_transaction()
        df = pd.DataFrame([tx])
        df_eng = model_components['feature_engineer'].transform(df)
        X = df_eng[model_components['feature_names']]
        X_scaled = model_components['scaler'].transform(X)
        
        prob = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
        
        # Classification should match threshold
        is_fraud = prob >= threshold
        assert isinstance(is_fraud, (bool, np.bool_))
        
class TestModelPerformance:
    """
    Test model performance characteristics
    """
    
    @pytest.fixture
    def model_components(self):
        """Load model components"""
        model_path = Path(__file__).parent.parent.parent / "models" / "production_model_ensemble.pkl"
        return joblib.load(model_path)
    
    def test_threshold_in_valid_range(self, model_components):
        """Test that threshold is in valid range"""
        threshold = model_components['optimal_threshold']
        
        assert 0 < threshold < 1
        assert threshold > 0.5 # Should be conservative
        
    def test_predicition_in_valid_range(self, model_components):
        """Test that all predicitions are valid probabilities"""
        transactions = [
            get_normal_transaction(),
            get_fraud_transaction(),
            get_sample_transaction()
        ]
        
        for tx in transactions:
            df = pd.DataFrame([tx])
            df_eng = model_components['feature_engineer'].transform(df)
            X = df_eng[model_components['feature_names']]
            X_scaled = model_components['scaler'].transform(X)
            
            prob = model_components['ensemble_model'].predict_proba(X_scaled)[0, 1]
            
            assert 0 <= prob <= 1