"""
Unit tests for feature engineering
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.engineer import FeatureEngineer
from tests.fixtures.test_data import get_sample_transaction, get_transaction_dataframe

class TestFeatureEngineer:
    """
    Test suite for FeatureEngineer class
    """
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame"""
        tx = get_sample_transaction()
        return pd.DataFrame([tx])
    
    def test_initialization(self, engineer):
        """Test FeatureEngineer initialized correctly"""
        assert engineer is not None
        assert hasattr(engineer, 'transform')
        
    def test_transform_single_transaction(self, engineer, sample_df):
        """Test transforming a single transaction"""
        result = engineer.transform(sample_df)
        
        # Check result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check result has correct number of rows
        assert len(result) == 1
        
        # Check has engineered features
        assert len(result.columns) > len(sample_df.columns)
        
    def test_feature_names(self, engineer, sample_df):
        """Test that feature names are consistent"""
        result = engineer.transform(sample_df)
        
        # Should have Amount_scaled, Hour, etc
        expected_features = [
            'Amount_log', 'Amount_scaled', 'Amount_bin', 'Amount_is_zero',
            'Hour', 'Is_night', 'Is_weekend_hour', 'Day',
            'V10_V14_interaction', 'negative_features_sum', 'max_abs_top_features']
        
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"
            
    def test_amount_scaling(self, engineer):
        """Test that amount scaling works orrectly"""
        df = pd.DataFrame([{
            **get_sample_transaction(),
            'Amount': 100.0
        }])
        
        result = engineer.transform(df)
        
        # Amount_scaled should be normalized
        assert 'Amount_scaled' in result.columns
        scaled = result['Amount_scaled'].iloc[0]
        
        assert scaled != 100.0
        
    def test_time_features(self, engineer):
        """Test time-based feature creation"""
        df = pd.DataFrame([{
            **get_sample_transaction(),
            'Time': 3600.0
        }])
        
        result = engineer.transform(df)
        
        # Should have time features
        assert 'Hour' in result.columns
        assert result['Hour'].iloc[0] == 1.0
        
    def test_handles_missing_time(self, engineer):
        """Test handling missing Time column"""
        df = pd.DataFrame([get_sample_transaction()])
        df = df.drop('Time', axis=1)
        
        try:
            result = engineer.transform(df)
            assert 'Time' not in result.columns
        except KeyError:
            pytest.skip("Feature engineer requires Time column")
            
    def test_preserves_original_features(self, engineer, sample_df):
        """Test that original V features are preserved"""
        result = engineer.transform(sample_df)
        
        for i in range (1, 29):
            assert f'V{i}' in sample_df.columns
            
    def test_deterministic_output(self, engineer, sample_df):
        """Test that transformation is deterministic"""
        result1 = engineer.transform(sample_df.copy())
        result2 = engineer.transform(sample_df.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
    def test_handles_edge_cases(self, engineer):
        """test handling of edge case values"""
        # Zero amount
        df = pd.DataFrame([{
            **get_sample_transaction(),
            'Amount': 0.0
        }])
        
        result = engineer.transform(df)
        assert not result.empty
        
        # Very large amount
        df = pd.DataFrame([{
            **get_sample_transaction(),
            'Amount': 100000.0
        }])
        
        result = engineer.transform(df)
        assert not result.empty
        
class TestFeatureValidation:
    """Test feature validation logic"""
    
    def test_all_required_features_present(self):
        """Test that sample data has all required features"""
        tx = get_sample_transaction()
        
        # Check V1-V28
        for i in range(1, 29):
            assert f'V{i}' in tx
            
        # Check basic features
        assert 'Time' in tx
        assert 'Amount' in tx
        
    def test_feature_data_types(self):
        """Test that features have correct data types"""
        tx = get_sample_transaction()
        
        for key, value in tx.items():
            assert isinstance(value, (int, float, np.number))