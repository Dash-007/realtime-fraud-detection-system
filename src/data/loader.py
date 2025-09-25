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