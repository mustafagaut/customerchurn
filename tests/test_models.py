"""Unit tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.models.train_model import create_pipeline, train_pipeline, save_model
from src.models.predict_model import load_model, predict, predict_proba


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 100

    X = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20, 100, n),
        "TotalCharges": np.random.uniform(100, 5000, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n)
    })
    y = pd.Series(np.random.choice([0, 1], n))

    return X, y


def test_create_pipeline(sample_train_data):
    """Test pipeline creation."""
    X, y = sample_train_data
    pipeline = create_pipeline(X)

    assert pipeline is not None
    assert "prep" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_train_pipeline(sample_train_data):
    """Test model training."""
    X, y = sample_train_data
    pipeline = train_pipeline(X, y)

    # Check that pipeline is fitted
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


def test_save_and_load_model(sample_train_data):
    """Test model saving and loading."""
    X, y = sample_train_data
    pipeline = train_pipeline(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.joblib"
        save_model(pipeline, filepath)

        loaded = load_model(filepath)
        assert loaded is not None

        # Check predictions match
        original_pred = pipeline.predict(X)
        loaded_pred = loaded.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)


def test_predict_functions(sample_train_data):
    """Test prediction functions."""
    X, y = sample_train_data
    pipeline = train_pipeline(X, y)

    predictions = predict(pipeline, X)
    assert predictions.shape == (len(X),)
    assert set(predictions).issubset({0, 1})

    probabilities = predict_proba(pipeline, X)
    assert probabilities.shape == (len(X),)
    assert all(0 <= p <= 1 for p in probabilities)
