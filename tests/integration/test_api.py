"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi import status
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestHealthEndpoint:
    """
    Test /health endpoint
    """
    
    def test_health_check_returns_200(self, api_client):
        """Test health endpoint return 200"""
        response = api_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
    def test_health_check_response_format(self, api_client):
        """Test health check has correct format"""
        response = api_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        
    def test_health_check_status_health(self, api_client):
        """Test that health status is healthy"""
        response = api_client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        
    def test_health_check_uptime_positive(self, api_client):
        """Test that uptime is a positive number"""
        response = api_client.get("/health")
        data = response.json()
        
        assert["uptime_seconds"] >= 0
        
class TestPredictEndpoint:
    """
    Test /predict endpoint
    """
    
    def test_predict_returns_200_for_valid_input(self, api_client, normal_transaction):
        """Test predict endpoint with valid transaction"""
        response = api_client.post("/predict", json=normal_transaction)
        assert response.status_code == status.HTTP_200_OK
        
    def test_predict_response_format(self, api_client, normal_transaction):
        """Test predict response has correct format"""
        response = api_client.post("/predict", json=normal_transaction)
        data = response.json()
        
        required_fields = ["is_fraud", "fraud_probability", "risk_level", "threshold_used", "model_version", "prediction_id", "timestamp"]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
            
    def test_predict_fraud_posibility_range(self, api_client, normal_transaction):
        """Test fraud probability is between 0 and 1"""
        response = api_client.post("/predict", json=normal_transaction)
        data = response.json()
        
        prob = data["fraud_probability"]
        assert 0 <= prob <= 1
        
    def test_predict_risk_level_values(self, api_client, normal_transaction):
        """Test risk level is valid"""
        response = api_client.post("/predict", json=normal_transaction)
        data = response.json()
        
        risk = data["risk_level"]
        assert risk in ["LOW", "MEDIUM", "HIGH"]
        
    def test_predict_is_fraud_boolean(self, api_client, normal_transaction):
        """Test is_fraud is boolean"""
        response = api_client.post("/predict", json=normal_transaction)
        data = response.json()
        
        assert isinstance(data["is_fraud"], bool)
        
    def test_predict_normal_transaction(self, api_client, normal_transaction):
        """Test prediction on normal transaction"""
        response = api_client.post("/predict", json=normal_transaction)
        data = response.json()
        
        # Normal transaction should have lower probability
        assert data["fraud_probability"] < 0.5
        assert data["risk_level"] in ["LOW", "MEDIUM"]
        
    def test_predict_fraud_transaction(self, api_client, fraud_transaction):
        """Test prediction on fraud transaction"""
        response = api_client.post("/predict", json=fraud_transaction)
        data = response.json()
        
        # Fraud transaction should have lower probability
        assert data["fraud_probability"] > 0.3
        assert data["risk_level"] in ["MEDIUM", "HIGH"]
        
    def test_predict_missing_field_returns_422(self, api_client):
        """Test that missing fields return validation error"""
        invalid_data = {"Time": 0.0, "Amount": 100.0} # Missing V1-V28
        
        response = api_client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        
    def test_predict_prediction_id_unique(self, api_client, normal_transaction):
        """Test that prediction IDs are unique"""
        response1 = api_client.post("/predict", json=normal_transaction)
        response2 = api_client.post("/predict", json=normal_transaction)
        
        id1 = response1.json()["prediction_id"]
        id2 = response2.json()["prediction_id"]
        
        assert id1 != id2
        
    def test_predict_consistency(self, api_client, normal_transaction):
        """Test that same input gives consistent probability"""
        response1 = api_client.post("/predict", json=normal_transaction)
        response2 = api_client.post("/predict", json=normal_transaction)
        
        prob1 = response1.json()["fraud_probability"]
        prob2 = response2.json()["fraud_probability"]
        
        assert abs(prob1 - prob2) < 0.001 # Allow tiny floating point differences