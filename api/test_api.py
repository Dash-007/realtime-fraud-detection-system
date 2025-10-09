"""
Test script for the Fraud Detection API
"""

import requests
import json
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# API endpoint
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)
    return response.status_code == 200


def test_single_prediction():
    """Test single transaction prediction"""
    
    # Sample transaction (can modify these values)
    transaction = {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": -3.090794,  # Suspicious value (negative)
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -5.311169,  # Very suspicious (very negative)
        "V15": 1.468177,
        "V16": -2.470401,  # Suspicious
        "V17": -1.207971,  # Suspicious
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 9.99  # Small amount (typical for fraud)
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=transaction
    )
    
    print("Single Prediction Result:")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Interpret results
    print("\nInterpretation:")
    if result.get('is_fraud'):
        print(f"FRAUD DETECTED! Probability: {result['fraud_probability']:.1%}")
    else:
        print(f"Transaction appears NORMAL. Fraud probability: {result['fraud_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print("-" * 50)
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction"""
    
    # Create 3 different transactions
    transactions = {
        "transactions": [
            # Normal transaction
            {
                "Time": 100.0,
                "V1": 0.5, "V2": 0.3, "V3": -0.2, "V4": 0.1, "V5": 0.4,
                "V6": -0.1, "V7": 0.2, "V8": -0.3, "V9": 0.1, "V10": 0.2,
                "V11": -0.1, "V12": 0.3, "V13": -0.2, "V14": 0.1, "V15": -0.4,
                "V16": 0.2, "V17": -0.1, "V18": 0.3, "V19": -0.2, "V20": 0.1,
                "V21": -0.3, "V22": 0.2, "V23": -0.1, "V24": 0.4, "V25": -0.2,
                "V26": 0.1, "V27": -0.3, "V28": 0.2,
                "Amount": 125.50
            },
            # Suspicious transaction
            {
                "Time": 200.0,
                "V1": -1.5, "V2": -0.8, "V3": 2.1, "V4": 1.2, "V5": -0.5,
                "V6": 0.3, "V7": 0.1, "V8": 0.2, "V9": 0.4, "V10": -4.5,
                "V11": -0.6, "V12": -0.8, "V13": -1.2, "V14": -6.3, "V15": 1.5,
                "V16": -3.2, "V17": -2.1, "V18": 0.1, "V19": 0.5, "V20": 0.3,
                "V21": -0.1, "V22": 0.3, "V23": -0.2, "V24": 0.1, "V25": 0.2,
                "V26": -0.3, "V27": 0.2, "V28": -0.1,
                "Amount": 2.99
            },
            # Another normal transaction
            {
                "Time": 300.0,
                "V1": 0.2, "V2": -0.1, "V3": 0.3, "V4": -0.2, "V5": 0.1,
                "V6": 0.4, "V7": -0.3, "V8": 0.2, "V9": -0.1, "V10": 0.3,
                "V11": 0.2, "V12": -0.4, "V13": 0.3, "V14": -0.2, "V15": 0.1,
                "V16": -0.3, "V17": 0.2, "V18": -0.1, "V19": 0.4, "V20": -0.2,
                "V21": 0.1, "V22": -0.3, "V23": 0.2, "V24": -0.1, "V25": 0.3,
                "V26": -0.2, "V27": 0.1, "V28": -0.3,
                "Amount": 89.00
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=transactions
    )
    
    print("Batch Prediction Results:")
    results = response.json()
    
    for i, result in enumerate(results, 1):
        print(f"\nTransaction {i}:")
        print(f"  Fraud: {'YES' if result['is_fraud'] else 'NO'}")
        print(f"  Probability: {result['fraud_probability']:.1%}")
        print(f"  Risk Level: {result['risk_level']}")
    
    print("-" * 50)
    return response.status_code == 200


def test_model_info():
    """Get model information"""
    response = requests.get(f"{BASE_URL}/model/info")
    print("Model Information:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)
    return response.status_code == 200


def test_analyze():
    """Test transaction analysis endpoint"""
    
    # Suspicious transaction for analysis
    transaction = {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": -4.5,  # High risk
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -7.2,  # Very high risk
        "V15": 1.468177,
        "V16": -3.8,  # High risk
        "V17": -2.5,  # High risk
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 1.99
    }
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json=transaction
    )
    
    print("Transaction Analysis:")
    result = response.json()
    print(json.dumps(result, indent=2))
    print("-" * 50)
    
    return response.status_code == 200


def run_all_tests():
    """Run all API tests"""
    print("="*50)
    print("RUNNING API TESTS")
    print("="*50)
    
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Transaction Analysis", test_analyze)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"Error: {e}")
            results.append((test_name, "ERROR"))
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, result in results:
        print(f"{test_name}: {result}")


if __name__ == "__main__":
    # Make sure the API is running first
    print("Make sure the API is running on http://localhost:8000")
    print("Run: cd api && uvicorn main:app --reload\n")
    
    try:
        # Check if API is accessible
        response = requests.get(f"{BASE_URL}/")
        print(f"API is running: {response.json()['message']}\n")
        
        # Run all tests
        run_all_tests()
        
    except requests.ConnectionError:
        print("Cannot connect to API. Please start it with:")
        print("cd api")
        print("uvicorn main:app --reload")