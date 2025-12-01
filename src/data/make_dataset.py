"""Data loading, cleaning, and splitting functions."""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw telco churn dataset."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "data" / "raw" / "telco-churn.csv"
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset."""
    df = df.copy()

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Fix TotalCharges - convert to numeric and fill missing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert Churn to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
