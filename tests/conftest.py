# tests/conftest.py
"""
Shared pytest fixtures for all tests
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def load_model():
    """Load model once for all tests (session scope)"""
    import joblib
    from pathlib import Path
    from datetime import datetime
    
    model_path = Path(__file__).parent.parent / "models" / "production_model_ensemble.pkl"
    
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")
    
    # Load model into global state
    model_package = joblib.load(model_path)
    
    # Store in api.main's global state
    import api.main as main_module
    main_module.model_package = model_package
    main_module.ensemble_model = model_package['ensemble_model']
    main_module.scaler = model_package['scaler']
    main_module.feature_engineer = model_package['feature_engineer']
    main_module.feature_names = model_package['feature_names']
    
    # Handle threshold - use from package or default
    if 'threshold' in model_package:
        main_module.threshold = model_package['threshold']
    else:
        # Use default threshold (same as in api/main.py)
        main_module.threshold = 0.703956557422205
        print("Warning: Using default threshold (not in model package)")
    
    # Initialize app_start_time for tests
    main_module.app_start_time = datetime.utcnow()
    
    return model_package


@pytest.fixture
def api_client(load_model):
    """Create FastAPI test client with model loaded"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    """Sample transaction fixture"""
    from tests.fixtures.test_data import get_sample_transaction
    return get_sample_transaction()


@pytest.fixture
def fraud_transaction():
    """Fraud transaction fixture"""
    from tests.fixtures.test_data import get_fraud_transaction
    return get_fraud_transaction()


@pytest.fixture
def normal_transaction():
    """Normal transaction fixture"""
    from tests.fixtures.test_data import get_normal_transaction
    return get_normal_transaction()


@pytest.fixture
def batch_transactions():
    """Batch of transactions fixture"""
    from tests.fixtures.test_data import get_batch_transactions
    return get_batch_transactions(n=5)