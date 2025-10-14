"""
Structured logging configuration for production monitoring
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Dict
import uuid


class StructuredLogger:
    """
    Production-grade structured logger for API requests.
    Outputs JSON logs for easy parsing by monitoring tools.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
    def log_request(
        self, 
        method: str,
        path: str,
        request_id: str,
        client_ip: str = None,
        extra: Dict[str, Any] = None
    ):
        """Log incoming request"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_ip": client_ip,
            **(extra or {})
        }
        self.logger.info(json.dumps(log_data))
        
    def log_response(
        self,
        request_id: str,
        status_code: int,
        duration_ms: float,
        extra: Dict[str, Any] = None
    ):
        """Log API response"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response",
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            **(extra or {})
        }
        self.logger.info(json.dumps(log_data))
        
    def log_prediction(
        self,
        request_id: str,
        fraud_probability: float,
        is_fraud: bool,
        risk_level: str,
        amount: float = None
    ):
        """Log fraud prediction details"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "prediction",
            "request_id": request_id,
            "fraud_probability": round(fraud_probability, 4),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "amount": amount
        }
        self.logger.info(json.dumps(log_data))
        
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        extra: Dict[str, Any] = None
    ):
        """Log error details"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            **(extra or {})
        }
        self.logger.error(json.dumps(log_data))


# Global logger instance
structured_logger = StructuredLogger("fraud_detection")