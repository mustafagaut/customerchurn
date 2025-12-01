"""Model prediction functions."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_model(filepath: str = None):
    """Load trained pipeline from disk."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "models" / "churn_model.joblib"
    return joblib.load(filepath)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Make binary predictions."""
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions for churn class."""
    return model.predict_proba(X)[:, 1]
