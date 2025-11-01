"""
Generate a test CSV file for batch prediction upload
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from api.sample_transactions import SAMPLE_TRANSACTIONS

def generate_test_batch(n_transactions=50, output_file='test_batch.csv'):
    """
    Generate a CSV file with test transactions.
    
    Args:
        n_transactions: Number of transactions to generate
        output_file: Output CSV filename
    """
    
    transactions = []
    
    # Distribution: 70% normal, 15% suspicious, 15% fraud
    for i in range(n_transactions):
        rand = np.random.random()
        
        if rand < 0.70:
            # Normal transaction
            base = SAMPLE_TRANSACTIONS['normal'].copy()
        elif rand < 0.85:
            # Suspicious transaction
            base = SAMPLE_TRANSACTIONS['suspicious'].copy()
        else:
            # Fraudulent transaction
            base = SAMPLE_TRANSACTIONS['fraud'].copy()
        
        # Add variations to make each transaction unique
        transaction = base.copy()
        
        # Vary amount
        transaction['Amount'] *= np.random.uniform(0.5, 2.0)
        
        # Increment time
        transaction['Time'] = i * np.random.uniform(50, 200)
        
        # Add small noise to PCA features
        for feature in [f'V{i}' for i in range(1, 29)]:
            transaction[feature] += np.random.normal(0, 0.1)
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Ensure correct column order
    columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    df = df[columns]
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "generated" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {n_transactions} transactions")
    print(f"Saved to: {output_path}")
    print(f"\nExpected Distribution:")
    print(f"   Normal: ~{int(n_transactions * 0.70)} transactions")
    print(f"   Suspicious: ~{int(n_transactions * 0.15)} transactions")
    print(f"   Fraud: ~{int(n_transactions * 0.15)} transactions")
    print(f"\nUsage:")
    print(f"   1. Go to Batch Prediction page")
    print(f"   2. Uncheck 'Use Sample Data'")
    print(f"   3. Upload '{output_file}'")
    print(f"   4. Click 'Analyze All Transactions'")
    
    return df


if __name__ == "__main__":
    # Generate different batch sizes
    
    print("=" * 60)
    print("GENERATING TEST BATCH FILES")
    print("=" * 60)
    print()
    
    # Small batch for quick testing
    print("Small Batch (20 transactions):")
    generate_test_batch(20, 'test_batch_small.csv')
    
    print("\n" + "-" * 60 + "\n")
    
    # Medium batch
    print("Medium Batch (50 transactions):")
    generate_test_batch(50, 'test_batch_medium.csv')
    
    print("\n" + "-" * 60 + "\n")
    
    # Large batch
    print("Large Batch (100 transactions):")
    generate_test_batch(100, 'test_batch_large.csv')
    
    print("\n" + "=" * 60)
    print("ALL TEST FILES GENERATED")
    print("=" * 60)