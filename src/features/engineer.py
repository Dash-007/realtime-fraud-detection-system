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