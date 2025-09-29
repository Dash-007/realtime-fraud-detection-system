"""
Feature engineering for faud detection.
Creates additional features to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Create and transform features for fraud detection.
    """
    
    def __init__(self):
        """
        Initialize feature engineering pipeline.
        """
        self.feature_names = None
        self.amount_stats = None
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn statistics from training data.
        
        
        Args:
            df: Training DataFrame
            
        Returns:
            self: Fitted FeatureEngineer
        """
        # Store amount statistics for scaling
        self.amount_stats = {
            'mean': df['Amount'].mean(),
            'std': df['Amount'].std(),
            'median': df['Amount'].median(),
            'q1': df['Amount'].quantile(0.25),
            'q3': df['Amount'].quantile(0.75)
        }
        
        logger.info(f"Fitted on {len(df)} samples.")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Dataframe with new features
        """
        # Create a copy to avoid modifying the original
        df_new = df.copy()
        
        # 1. Amount-based features
        df_new = self._create_amount_features(df_new)
        
        # 2. Time-based features
        df_new = self._create_time_features(df_new)
        
        # 3. Statistical features
        df_new = self._create_statistical_features(df_new)
        
        # Store feature names
        self.feature_names = [col for col in df_new.columns
                              if col not in ['Class', 'Time']]
        
        logger.info(f"Created {len(self.feature_names)} total features (excluding 'Time').")
        
        return df_new
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        return self.fit(df).transform(df)
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on transaction amount.
        """
        
        # Log transform (handles skewed distribution - reduces impact of extreme values)
        # Add 1 to avoid log(0)
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Normalized amount (using training stats - for models that need scaling)
        if self.amount_stats:
            df['Amount_scaled'] = (df['Amount'] - self.amount_stats['mean']) / self.amount_stats['std']
        else:
            # If not fitted yet
            df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            
        # Amount bins (categorical ranges - captures non-linear patterns in ranges)
        df['Amount_bin'] = pd.cut(df['Amount'],
                                  bins=[0, 10, 50, 100, 500, 30000],
                                  labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Convert to numeric
        df['Amount_bin'] = df['Amount_bin'].cat.codes
        
        # Is zero amount?
        df['Amount_is_zero'] = (df['Amount'] == 0).astype(int)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on transaction time.
        """
        # Hour of day (0-23)
        df['Hour'] = (df['Time'] / 3600) % 24
        
        # Part of the day
        df['Is_night'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
        df['Is_weekend_hour'] = df['Hour'].apply(lambda x: 1 if x in [0, 1, 22, 23] else 0)
        
        # Day from start (assuming Time is seconds from start)
        df['Day'] = df['Time'] // (24 * 3600)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical combination features.
        """
        # Interaction between most important PCA features
        # Based on the EDA, V10 and V14 were most important
        df['V10_V14_interaction'] = df['V10'] * df['V14']
        
        # Sum of top negative features (fraud indicators from EDA)
        df['negative_features_sum'] = df[['V10', 'V14', 'V16', 'V17']].sum(axis=1)
        
        # Max absolute value among top features
        top_features = ['V10', 'V14', 'V16', 'V17', 'V11', 'V12']
        df['max_abs_top_features'] = df[top_features].abs().max(axis=1)
        
        return df
    
# Test the feature engineer
if __name__ == "__main__":
    print("Testing FeatureEngineer...")
    
    # import our data loader
    import sys
    import os
    sys.path.append('../..')
    from src.data.loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Test feature engineering
    engineer = FeatureEngineer()
    
    # Original features
    original_features = df.shape[1] - 1
    print(f"Original features: {original_features} features")
    
    # Create new features
    df_engineered = engineer.fit_transform(df)
    
    # New features
    new_features = df_engineered.shape[1] - 1
    print(f"After engineering: {new_features} features")
    print(f"New features created: {new_features - original_features}")
    
    # Show some new features
    new_features_names = [col for col in df_engineered.columns if col not in df.columns]
    print(f"\nNew features: {new_features_names}")