"""
Shared pytest fixtures for all tests
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def api_client():
    """Create fast api test client"""
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