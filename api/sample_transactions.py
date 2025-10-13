"""
Sample transactions for testing the fraud detection API.
Based on patterns discovered in EDA.
"""


# Normal transaction - typical legitimate purchase
NORMAL_TRANSACTION = {
    "Time": 5000.0,
    "Amount": 125.50,
    "V1": 0.144, "V2": 0.358, "V3": 1.220, "V4": 0.331, "V5": -0.273,
    "V6": 0.635, "V7": 0.463, "V8": -0.215, "V9": -0.168, "V10": 0.125,
    "V11": -0.322, "V12": 0.179, "V13": 0.507, "V14": -0.287, "V15": -0.631,
    "V16": 0.123, "V17": -0.294, "V18": 0.661, "V19": 0.738, "V20": 0.868,
    "V21": 0.255, "V22": 0.662, "V23": -0.103, "V24": 0.502, "V25": 0.401,
    "V26": -0.128, "V27": -0.018, "V28": 0.028
}


# High-risk fraud pattern - based on EDA findings
# Characteristics: Very negative V10, V14, V16, V17, small amount
FRAUDULENT_TRANSACTION = {
    "Time": 12000.0,
    "Amount": 1.99,  # Small amount (fraud pattern)
    "V1": -1.359, "V2": -0.073, "V3": 2.536, "V4": 1.378, "V5": -0.338,
    "V6": 0.462, "V7": 0.240, "V8": 0.099, "V9": 0.364, "V10": -5.5,  # Very negative
    "V11": -0.552, "V12": -0.618, "V13": -0.991, "V14": -7.2,  # Very negative
    "V15": 1.468, "V16": -3.8,  # Very negative
    "V17": -2.5,  # Very negative
    "V18": 0.026, "V19": 0.404, "V20": 0.251, "V21": -0.018, "V22": 0.278,
    "V23": -0.110, "V24": 0.067, "V25": 0.129, "V26": -0.189, "V27": 0.134,
    "V28": -0.021
}


# Suspicious but not definite fraud - borderline case
SUSPICIOUS_TRANSACTION = {
    "Time": 8000.0,
    "Amount": 45.00,
    "V1": -0.876, "V2": -0.234, "V3": 1.234, "V4": 0.567, "V5": -0.123,
    "V6": 0.234, "V7": 0.123, "V8": 0.045, "V9": 0.167, "V10": -2.5,  # Moderately negative
    "V11": -0.234, "V12": -0.345, "V13": -0.456, "V14": -3.2,  # Moderately negative
    "V15": 0.678, "V16": -1.8,  # Moderately negative
    "V17": -1.2,  # Moderately negative
    "V18": 0.012, "V19": 0.234, "V20": 0.123, "V21": -0.009, "V22": 0.156,
    "V23": -0.067, "V24": 0.034, "V25": 0.078, "V26": -0.098, "V27": 0.067,
    "V28": -0.011
}


# High amount normal transaction - expensive legitimate purchase
HIGH_AMOUNT_NORMAL = {
    "Time": 15000.0,
    "Amount": 2500.00,  # High amount but legitimate pattern
    "V1": 0.234, "V2": 0.567, "V3": 0.890, "V4": 0.123, "V5": -0.456,
    "V6": 0.789, "V7": 0.345, "V8": -0.123, "V9": -0.234, "V10": 0.345,
    "V11": -0.456, "V12": 0.234, "V13": 0.678, "V14": -0.123, "V15": -0.789,
    "V16": 0.234, "V17": -0.345, "V18": 0.567, "V19": 0.890, "V20": 1.123,
    "V21": 0.345, "V22": 0.789, "V23": -0.123, "V24": 0.567, "V25": 0.456,
    "V26": -0.234, "V27": -0.012, "V28": 0.034
}


# Zero/very small amount - sometimes fraud, sometimes legitimate
ZERO_AMOUNT_TRANSACTION = {
    "Time": 2000.0,
    "Amount": 0.01,  # Minimal amount
    "V1": 0.123, "V2": 0.234, "V3": 0.345, "V4": 0.456, "V5": 0.567,
    "V6": 0.678, "V7": 0.789, "V8": 0.890, "V9": 0.123, "V10": 0.234,
    "V11": 0.345, "V12": 0.456, "V13": 0.567, "V14": 0.678, "V15": 0.789,
    "V16": 0.890, "V17": 0.123, "V18": 0.234, "V19": 0.345, "V20": 0.456,
    "V21": 0.567, "V22": 0.678, "V23": 0.789, "V24": 0.890, "V25": 0.123,
    "V26": 0.234, "V27": 0.345, "V28": 0.456
}


# All samples in a dictionary for easy access
SAMPLE_TRANSACTIONS = {
    "normal": NORMAL_TRANSACTION,
    "fraud": FRAUDULENT_TRANSACTION,
    "suspicious": SUSPICIOUS_TRANSACTION,
    "high_amount": HIGH_AMOUNT_NORMAL,
    "zero_amount": ZERO_AMOUNT_TRANSACTION
}


def get_sample(transaction_type: str) -> dict:
    """
    Get a sample transaction by type.
    
    Args:
        transaction_type: One of 'normal', 'fraud', 'suspicious', 
                         'high_amount', 'zero_amount'
    
    Returns:
        dict: Transaction data
    """
    if transaction_type not in SAMPLE_TRANSACTIONS:
        raise ValueError(
            f"Unknown transaction type: {transaction_type}. "
            f"Choose from: {list(SAMPLE_TRANSACTIONS.keys())}"
        )
    
    return SAMPLE_TRANSACTIONS[transaction_type].copy()


def get_all_samples() -> dict:
    """
    Get all sample transactions.
    """
    return {k: v.copy() for k, v in SAMPLE_TRANSACTIONS.items()}


# Testing feature - generate variations
def create_variation(base_transaction: dict, amount: float = None) -> dict:
    """
    Create a variation of a transaction with different amount.
    
    Args:
        base_transaction: Base transaction to modify
        amount: New amount value
    
    Returns:
        dict: Modified transaction
    """
    transaction = base_transaction.copy()
    if amount is not None:
        transaction['Amount'] = amount
    return transaction


if __name__ == "__main__":
    # Demo the sample transactions
    print("Sample Transactions Library")
    print("=" * 50)
    
    for name, transaction in SAMPLE_TRANSACTIONS.items():
        print(f"\n{name.upper()}:")
        print(f"  Amount: ${transaction['Amount']:.2f}")
        print(f"  V10: {transaction['V10']:.2f} (fraud indicator if very negative)")
        print(f"  V14: {transaction['V14']:.2f} (fraud indicator if very negative)")