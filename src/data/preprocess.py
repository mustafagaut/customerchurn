"""Preprocessing pipelines and transformers."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


def get_column_types(X: pd.DataFrame) -> tuple:
    """Identify categorical and numerical columns."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    return cat_cols, num_cols


def get_preprocessor(X: pd.DataFrame, smoothing: float = 10) -> ColumnTransformer:
    """Create preprocessing pipeline with StandardScaler and TargetEncoder."""
    cat_cols, num_cols = get_column_types(X)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", TargetEncoder(smoothing=smoothing), cat_cols)
    ])

    return preprocessor
