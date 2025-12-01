"""Unit tests for feature engineering functions."""

import pytest
import pandas as pd

from src.features.build_features import create_tenure_groups, create_service_count


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "tenure": [6, 15, 30, 60],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "Yes", "No internet service", "Yes"]
    })


def test_create_tenure_groups(sample_data):
    """Test tenure group creation."""
    result = create_tenure_groups(sample_data)

    assert "TenureGroup" in result.columns
    assert result["TenureGroup"].iloc[0] == "0-1yr"  # tenure=6
    assert result["TenureGroup"].iloc[1] == "1-2yr"  # tenure=15
    assert result["TenureGroup"].iloc[2] == "2-4yr"  # tenure=30
    assert result["TenureGroup"].iloc[3] == "4-6yr"  # tenure=60


def test_create_tenure_groups_preserves_original(sample_data):
    """Test that original data is not modified."""
    original_cols = sample_data.columns.tolist()
    create_tenure_groups(sample_data)

    assert sample_data.columns.tolist() == original_cols


def test_create_service_count(sample_data):
    """Test service count calculation."""
    result = create_service_count(sample_data)

    assert "ServiceCount" in result.columns
    # Row 0: PhoneService=Yes, InternetService=DSL, OnlineSecurity=Yes, OnlineBackup=No -> 3
    assert result["ServiceCount"].iloc[0] == 3
    # Row 2: PhoneService=Yes, InternetService=No -> 1
    assert result["ServiceCount"].iloc[2] == 1
