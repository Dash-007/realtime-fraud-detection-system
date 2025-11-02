"""
Test data fixtures for unit and integration tests
"""

import pandas as pd
import numpy as np

def get_sample_transaction():
    """
    Get a single sample transaction for testing
    """
    return {
        'Time': 0.0,
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 149.62
    }

def get_fraud_transaction():
    """
    Get a fraudulent transaction for testing
    """
    return {
        'Time': 12000.0,
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': -5.5, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -7.2, 'V15': 1.468177, 'V16': -3.8,
        'V17': -2.5, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.129539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 1.99
    }

def get_normal_transaction():
    """
    Get a normal (legitimate) transaction for testing
    """
    return {
        'Time': 5000.0,
        'V1': 0.144, 'V2': 0.358, 'V3': 1.220, 'V4': 0.331,
        'V5': -0.273, 'V6': 0.635, 'V7': 0.463, 'V8': -0.215,
        'V9': -0.168, 'V10': 0.125, 'V11': -0.322, 'V12': 0.179,
        'V13': 0.507, 'V14': -0.287, 'V15': -0.631, 'V16': 0.123,
        'V17': -0.294, 'V18': 0.661, 'V19': 0.738, 'V20': 0.868,
        'V21': 0.255, 'V22': 0.662, 'V23': -0.103, 'V24': 0.502,
        'V25': 0.401, 'V26': -0.128, 'V27': -0.018, 'V28': 0.028,
        'Amount': 125.50
    }

def get_batch_transactions(n=5):
    """
    Get multiple transactions for batch testing
    """
    transactions = []
    
    for i in range(n):
        if i % 3 == 0:
            tx = get_fraud_transaction().copy()
        else:
            tx = get_normal_transaction().copy()
        
        # Add variation
        tx['Time'] = i * 100.0
        tx['Amount'] *= np.random.uniform(0.8, 1.2)
        
        transactions.append(tx)
    
    return transactions

def get_invalid_transaction():
    """
    Get an invalid transaction (missing fields) for error testing
    """
    return {
        'Time': 0.0,
        'Amount': 100.0,
        # Missing V1-V28 features
    }

def get_transaction_dataframe(n=10):
    """
    Get a DataFrame of transactions for testing
    """
    transactions = get_batch_transactions(n)
    return pd.DataFrame(transactions)