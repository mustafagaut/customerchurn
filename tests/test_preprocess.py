"""Unit tests for preprocessing functions."""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from src.data.preprocess import get_column_types, get_preprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "tenure": [1, 12, 24, 48],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
        "gender": ["Female", "Male", "Male", "Female"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"]
    })


def test_get_column_types(sample_data):
    """Test column type identification."""
    cat_cols, num_cols = get_column_types(sample_data)

    assert set(cat_cols) == {"gender", "Contract"}
    assert set(num_cols) == {"tenure", "MonthlyCharges"}


def test_get_preprocessor_returns_column_transformer(sample_data):
    """Test that get_preprocessor returns a ColumnTransformer."""
    preprocessor = get_preprocessor(sample_data)

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_transforms_data(sample_data):
    """Test that preprocessor can transform data."""
    preprocessor = get_preprocessor(sample_data)
    y = pd.Series([0, 1, 1, 0])

    transformed = preprocessor.fit_transform(sample_data, y)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] == 4  # 2 num + 2 cat columns
