"""
Data loading utilities for fraud detection system.
This module handles all data ingestion with proper error handling and logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
import yaml

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data loading and initial validation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize with configuration.
        """
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {
                'data': {
                    'raw_path': 'data/raw',
                    'processed_path': 'data/processed',
                    'filename': 'creditcard.csv',
                    'test_size': 0.2,
                    'random_state': 42
                }
            }
            
    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load credit card fraud dataset with validation.
        
        Returns:
            DataFrame with transaction data
        """
        if filename is None:
            filename = self.config['data']['filename']
            
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {file_path}."
            )
            
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate data
        self._validate_data(df)
        
        # Add metadata
        self.add_metadata(df)
        
        logger.info(f"Loaded {len(df):,} transactions")
        logger.info(f"Fraud rate: {df['Class'].mean():.2%}")
        logger.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data structure and quality.
        """
        # Check expected columns
        expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        # Check for nulls in target
        if df['Class'].isnull().any():
            raise ValueError("Target column 'Class' contains null values")
        
        # Check target values
        if not set(df['Class'].unique()).issubset({0, 1}):
            raise ValueError("Target should only contain 0 (normal) and 1 (fraud)")
        
        logger.info("Data validation passed!")
        
    