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
        
        required_keys = ['ensemble_model', 'scaler', 'feature_engineer', 'feature_names', 'threshold']
        
        for key in required_keys:
            assert key in model_package, f"Missing required key: {key}"
            
    def test_model_feature_count(self, model_path):
        """Test that model expects correct number of features"""
        model_package = joblib.load(model_path)
        feature_names = model_package['feature_names']
        
        # Should have approximately 40 features
        assert len(feature_names) >=30
        assert len(feature_names) <= 50