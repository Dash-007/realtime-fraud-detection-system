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
        
class TestBatchPredictionEndpoint:
    """
    Test /predict/batch endpoint
    """
    
    def test_batch_predict_returns_200(self, api_client, batch_transactions):
        """Test batch predict with valid transactions"""
        response = api_client.post(
            "/predict/batch",
            json={"transactions": batch_transactions}
        )
        assert response.status_code == status.HTTP_200_OK
        
    def test_batch_prediction_return_list(self, api_client, batch_transactions):
        """Test batch predict returns list of predictions"""
        response = api_client.post(
            "/predict/batch",
            json={"transactions": batch_transactions}
        )
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == len(batch_transactions)
        
    def test_batch_predict_each_has_required_fields(self, api_client, batch_transactions):
        """Test each prediction has required fields"""
        response = api_client.post(
            "/predict/batch",
            json={"transactions": batch_transactions}
        )
        predictions = response.json()
        
        required_fields = ["is_fraud", "fraud_probability", "risk_level"]
        
        for pred in predictions:
            for field in required_fields:
                assert field in pred
                
    def test_batch_predict_empty_list(self, api_client, batch_transactions):
        """Test batch predict with empty list"""
        response = api_client.post(
            "/predict/batch",
            json={"transactions": batch_transactions}
        )
        
        # Should either return 200 with empty list or 422
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_CONTENT]
        
    def test_batch_predict_single_transaction(self, api_client, normal_transaction):
        """Test batch with single transaction"""
        response = api_client.post(
            "/predict/batch",
            json={"transactions": normal_transaction}
        )
        data = response.json()
        
        assert response.status_code == status.HTTP_200_OK
        assert len(data) == 1
        
class TestAnalyzeEndpoint:
    """
    Test /analyze endpoint
    """
    
    def test_analyze_returns_200(self, api_client, normal_transaction):
        """Test analyze endpoint returns 200"""
        response = api_client.post("/analyze", json=normal_transaction)
        assert response.status_code == status.HTTP_200_OK
        
    def test_analyze_response_format(self, api_client, normal_transaction):
        """Test analyze has detailed breakdown"""
        response = api_client.post("/analyze", json=normal_transaction)
        data = response.json()
        
        assert "prediction" in data
        assert "feature_analysis" in data
        assert "model_breakdown" in data
        
    def test_analyze_has_feature_contributions(self, api_client, normal_transaction):
        """Test analyze includes feature importance"""
        response = api_client.post("/analyze", json=normal_transaction)
        data = response.json()
        
        assert "high_risk_features" in data["feature_analysis"]
        assert "low_risk_features" in data["feature_analysis"]
        
    def test_analyze_has_recommendation(self, api_client, normal_transaction):
        """Test analyze includes recommendation"""
        response = api_client.post("/analyze", json=normal_transaction)
        data = response.json()
        
        assert "recommendation" in data
        
class TestModelInfoEndpoint:
    """
    Test /model/info endpoint
    """
    
    def test_model_info_returns_200(self, api_client):
        """Test model info endpoint"""
        response = api_client.get("/model/info")
        assert response.status_code == status.HTTP_200_OK
        
    def test_model_info_format(self, api_client):
        """Test model info has correct format"""
        response = api_client.get("/model/info")
        data = response.json()
        
        assert "model_version" in data
        assert "model_type" in data
        assert "features_count" in data
        assert "threshold" in data
        
    def test_model_info_has_performance_metrics(self, api_client):
        """Test model info includes performace metrics"""
        response = api_client.get("/model/info")
        data = response.json()
        
        assert "performance" in data
        perf = data["performace"]
        
        assert "precision" in perf
        assert "recall" in perf
        assert "f1_score" in perf
        
class TestDocsEndpoint:
    """
    Test API documentation endpoints
    """
    
    def test_docs_endpoint_accessible(self, api_client):
        """Test that /docs is accessible"""
        response = api_client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        
    def test_openai_json_accessible(self, api_client):
        """test that OpenAPI spec is accessible"""
        response = api_client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "openapi" in data
        assert "indo" in data
        assert "paths" in data