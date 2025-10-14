"""
Quick demo of the Fraud Detection API
"""

import requests
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.sample_transactions import get_sample

BASE_URL = "http://localhost:8000"


def print_separator():
    print("\n" + "=" * 50 + "\n")


def demo_single_prediction():
    """Demo: Single transaction prediction"""
    print("DEMO: Single Transaction Prediction")
    print_separator()
    
    # Try a normal transaction
    print("Testing NORMAL transaction...")
    normal = get_sample("normal")
    
    response = requests.post(f"{BASE_URL}/predict", json=normal)
    result = response.json()
    
    print(f"Amount: ${normal['Amount']:.2f}")
    print(f"Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"Confidence: {result['fraud_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    
    time.sleep(1)
    
    # Try a fraudulent transaction
    print("\n" + "-" * 50 + "\n")
    print("Testing FRAUDULENT transaction...")
    fraud = get_sample("fraud")
    
    response = requests.post(f"{BASE_URL}/predict", json=fraud)
    result = response.json()
    
    print(f"Amount: ${fraud['Amount']:.2f}")
    print(f"Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"Confidence: {result['fraud_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")


def demo_batch_prediction():
    """Demo: Batch prediction"""
    print_separator()
    print("DEMO: Batch Prediction")
    print_separator()
    
    batch = {
        "transactions": [
            get_sample("normal"),
            get_sample("fraud"),
            get_sample("suspicious")
        ]
    }
    
    print("Processing 3 transactions in batch...")
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    results = response.json()
    
    for i, result in enumerate(results, 1):
        print(f"\nTransaction {i}:")
        print(f"  Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"  Probability: {result['fraud_probability']:.1%}")
        print(f"  Risk: {result['risk_level']}")


def demo_explainability():
    """Demo: Model explainability"""
    print_separator()
    print("DEMO: Model Explainability")
    print_separator()
    
    fraud = get_sample("fraud")
    
    print("Analyzing a high-risk transaction...")
    response = requests.post(f"{BASE_URL}/analyze", json=fraud)
    result = response.json()
    
    print(f"\nFraud Probability: {result['fraud_probability']:.1%}")
    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
    
    if 'individual_model_predictions' in result:
        print(f"\nIndividual Model Predictions:")
        for model, prob in result['individual_model_predictions'].items():
            print(f"  {model}: {prob:.1%}")
    
    if 'feature_analysis' in result:
        high_risk = result['feature_analysis'].get('high_risk_features', [])
        if high_risk:
            print(f"\nHigh-Risk Features Detected: {', '.join(high_risk)}")


def run_demo():
    """Run complete demo"""
    print("\n" + "=" * 50)
    print("FRAUD DETECTION API - DEMO")
    print("=" * 50)
    
    try:
        # Check health
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"\nAPI Status: {health['status'].upper()}")
        print(f"Model Version: {health['model_version']}")
        print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
        
        # Run demos
        demo_single_prediction()
        demo_batch_prediction()
        demo_explainability()
        
        print_separator()
        print("DEMO COMPLETE!")
        print("\nNext steps:")
        print("  1. Try the interactive docs: http://localhost:8000/docs")
        print("  2. Run full tests: python api/test_samples.py")
        print("  3. Explore sample transactions: python api/sample_transactions.py")
        
    except requests.ConnectionError:
        print("\nCannot connect to API.")
        print("Please start the API first: uvicorn api.main:app --reload")


if __name__ == "__main__":
    run_demo()