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
        
        logger.info(f"Created {len(self.feature_names)} total features.")
        
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
    
    