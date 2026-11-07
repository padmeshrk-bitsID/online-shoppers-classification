"""Feature transformation step (fitted inside each model pipeline).

Because the transformer lives *inside* the saved pipeline, every `model/*.joblib`
file already contains its fitted preprocessing; no separate artifact is needed
at prediction time.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def build_numeric_transformer() -> Pipeline:
    """log1p tames the heavy right tail of counts/durations; scaling puts features on one scale."""
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def build_categorical_transformer(min_frequency: int = config.RARE_CATEGORY_MIN_COUNT) -> OneHotEncoder:
    """One-hot encode categories; rare and unseen levels share one 'infrequent' column."""
    return OneHotEncoder(
        min_frequency=min_frequency,
        handle_unknown="infrequent_if_exist",
        drop="if_binary",          # Weekend becomes a single 0/1 column
        sparse_output=False,       # Gaussian Naive Bayes needs a dense matrix
    )


def build_preprocessor(
    numeric_features: list[str] = config.NUMERIC_FEATURES,
    categorical_features: list[str] = config.CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Unfitted column transformer; a fresh copy is created for every model."""
    return ColumnTransformer([
        ("num", build_numeric_transformer(), list(numeric_features)),
        ("cat", build_categorical_transformer(), list(categorical_features)),
    ])
