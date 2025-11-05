import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Fixture réutilisable pour tous les tests"""
    np.random.seed(42)
    x_train = np.random.rand(100, 10)
    x_test = np.random.rand(20, 10)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)
    return x_train, x_test, y_train, y_test