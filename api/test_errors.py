"""
Test error handling in the API
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_invalid_amount():
    """
    Test that negative amounts are rejected
    """
    print("\nTesting: Invalid Amount (Negative)")
    
    transaction = {
        "Time": 0.0,
        "Amount": -100.0,  # Invalid: negative amount
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Should return 422 (Unprocessable Entity)
    assert response.status_code == 422, "Should reject negative amount"
    print("PASSED: Correctly rejects invalid amount")


def test_missing_features():
    """Test that missing features are rejected"""
    print("\nTesting: Missing Features")
    
    incomplete_transaction = {
        "Time": 0.0,
        "Amount": 100.0,
        "V1": 0.0
        # Missing V2-V28
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=incomplete_transaction)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Should return 422
    assert response.status_code == 422, "Should reject incomplete data"
    print("PASSED: Correctly rejects missing features")


def test_invalid_data_types():
    """Test that wrong data types are rejected"""
    print("\nTesting: Invalid Data Types")
    
    transaction = {
        "Time": "not_a_number",  # Invalid: should be float
        "Amount": 100.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422, "Should reject wrong data type"
    print("PASSED: Correctly rejects invalid data type")


if __name__ == "__main__":
    print("=" * 50)
    print("ERROR HANDLING TESTS")
    print("=" * 50)
    
    try:
        # Check if API is running
        response = requests.get(f"{BASE_URL}/health")
        print(f"API is running\n")
        
        # Run error tests
        test_invalid_amount()
        test_missing_features()
        test_invalid_data_types()
        
        print("=" * 50)
        print("ALL ERROR HANDLING TESTS PASSED")
        print("=" * 50)
        
    except requests.ConnectionError:
        print("Cannot connect to API. Please start it first.")