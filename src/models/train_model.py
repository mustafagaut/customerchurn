"""Model training functions."""

import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import get_preprocessor


def create_pipeline(X_train, max_iter: int = 2000, class_weight: str = "balanced"):
    """Create training pipeline with preprocessing and classifier."""
    preprocessor = get_preprocessor(X_train)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver="liblinear"
        ))
    ])

    return pipeline


def train_pipeline(X_train, y_train, max_iter: int = 2000, class_weight: str = "balanced"):
    """Train the churn prediction pipeline."""
    pipeline = create_pipeline(X_train, max_iter, class_weight)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline, filepath: str = None):
    """Save trained pipeline to disk."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "models" / "churn_model.joblib"

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    return filepath
