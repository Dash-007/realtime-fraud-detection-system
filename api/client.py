"""
Python client SDK for Fraud Detection API
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    threshold_used: float
    prediction_id: str
    
    @property
    def confidence(self) -> str:
        """Human-readable confidence level"""
        if self.fraud_probability > 0.8:
            return "Very High"
        elif self.fraud_probability > 0.5:
            return "High"
        elif self.fraud_probability > 0.2:
            return "Medium"
        else:
            return "Low"


class FraudDetectionClient:
    """
    Client for interacting with the Fraud Detection API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize client.
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def health_check(self) -> Dict:
        """
        Check API health.
        
        Returns:
            dict: Health status information
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
        
    def predict(self, transaction: Dict) -> FraudPrediction:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction data with all required features
            
        Returns:
            FraudPrediction: Prediction result
            
        Raises:
            requests.HTTPError: If API returns error
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=transaction,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return FraudPrediction(
            is_fraud=data['is_fraud'],
            fraud_probability=data['fraud_probability'],
            risk_level=data['risk_level'],
            threshold_used=data['threshold_used'],
            prediction_id=data['prediction_id']
        )
        
    def predict_batch(self, transactions: List[Dict]) -> List[FraudPrediction]:
        """
        Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List[FraudPrediction]: List of predictions
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json={"transactions": transactions},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        results = response.json()
        return [
            FraudPrediction(
                is_fraud=r['is_fraud'],
                fraud_probability=r['fraud_probability'],
                risk_level=r['risk_level'],
                threshold_used=r['threshold_used'],
                prediction_id=r['prediction_id']
            )
            for r in results
        ]
        
    def analyze(self, transaction: Dict) -> Dict:
        """
        Get detailed analysis for a transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            dict: Detailed analysis including model breakdowns
        """
        response = self.session.post(
            f"{self.base_url}/analyze",
            json=transaction,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
        
    def close(self):
        """Close the session"""
        self.session.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()


# Example usage
if __name__ == "__main__":
    from api.sample_transactions import get_sample
    
    # Initialize client
    client = FraudDetectionClient("http://localhost:8000")
    
    # Check health
    print("Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model: {health['model_version']}")
    
    # Single prediction
    print("\nSingle Prediction:")
    transaction = get_sample("normal")
    result = client.predict(transaction)
    
    print(f"   Result: {'FRAUD' if result.is_fraud else 'LEGITIMATE'}")
    print(f"   Probability: {result.fraud_probability:.1%}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Request ID: {result.prediction_id}")
    
    # Batch prediction
    print("\nBatch Prediction:")
    transactions = [
        get_sample("normal"),
        get_sample("fraud"),
        get_sample("suspicious")
    ]
    results = client.predict_batch(transactions)
    
    for i, result in enumerate(results, 1):
        print(f"   Transaction {i}: {result.risk_level} risk ({result.fraud_probability:.1%})")
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    fraud_tx = get_sample("fraud")
    analysis = client.analyze(fraud_tx)
    print(f"   Recommendation: {analysis.get('recommendation')}")
    print(f"   High-risk features: {analysis['feature_analysis'].get('high_risk_features', [])}")
    
    client.close()
    print("\nClient demo complete!")