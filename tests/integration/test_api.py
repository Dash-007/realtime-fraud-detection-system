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
        
    