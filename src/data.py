"""Data access and cleaning.

`DataPreparer` is used by BOTH training and the Streamlit app, so a file uploaded
to the app is cleaned exactly the way the training data was (no train/serve skew).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

_TRUE_STRINGS = {"true", "1", "yes"}
_FALSE_STRINGS = {"false", "0", "no"}


class SchemaError(ValueError):
    """Raised when an input table does not have the columns / values the models need."""


@dataclass(frozen=True)
class SchemaReport:
    """Result of checking an uploaded table *before* scoring it (drives the app's checklist)."""

    n_rows: int
    n_columns: int
    missing_features: list[str]
    target_present: bool
    target_required: bool
    columns_with_nulls: list[str]
    extra_columns: list[str]

    @property
    def problems(self) -> list[str]:
        """Human-readable reasons the table cannot be scored (empty list = all good)."""
        issues = []
        if self.n_rows == 0:
            issues.append("The file has no data rows.")
        if self.missing_features:
            issues.append(f"Missing input column(s): {', '.join(self.missing_features)}")
        if self.target_required and not self.target_present:
            issues.append(f"Missing target column '{config.TARGET}' (needed to compute the metrics).")
        if self.columns_with_nulls:
            issues.append(f"Empty cells found in: {', '.join(self.columns_with_nulls)}")
        return issues

    @property
    def ok(self) -> bool:
        return not self.problems


def load_raw_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw UCI CSV exactly as published."""
    return pd.read_csv(path)


def remove_duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows so the same session cannot sit in both train and test."""
    return raw.drop_duplicates().reset_index(drop=True)


def split_train_test(
    raw: pd.DataFrame,
    target: str = config.TARGET,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of the *raw* rows (keeps the purchase rate equal in both parts)."""
    train, test = train_test_split(
        raw, test_size=test_size, stratify=raw[target], random_state=random_state
    )
    return train, test


def _to_binary(series: pd.Series) -> pd.Series:
    """Map True/False, 'TRUE'/'FALSE', 1/0, 'yes'/'no' to integers 1/0."""
    if series.dtype == bool:
        return series.astype(int)
    text = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - _TRUE_STRINGS - _FALSE_STRINGS)
    if unknown:
        raise SchemaError(f"Column '{series.name}' has non-boolean values: {unknown[:5]}")
    return text.isin(_TRUE_STRINGS).astype(int)


class DataPreparer:
    """Validates a raw table and converts it into model-ready features (X) and target (y).

    Works for raw UCI rows, for `test_data.csv`, and for user uploads, with or
    without the target column.
    """

    def __init__(
        self,
        numeric_features: list[str] = config.NUMERIC_FEATURES,
        categorical_features: list[str] = config.CATEGORICAL_FEATURES,
        coded_categorical_features: list[str] = config.CODED_CATEGORICAL_FEATURES,
        target: str = config.TARGET,
    ):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.coded_categorical_features = list(coded_categorical_features)
        self.target = target

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def has_target(self, raw: pd.DataFrame) -> bool:
        return self.target in raw.columns

    def inspect(self, raw: pd.DataFrame, require_target: bool = True) -> SchemaReport:
        """Check columns and empty cells without raising, so the UI can show a checklist."""
        expected = [*self.feature_columns, self.target]
        present = [c for c in expected if c in raw.columns]
        return SchemaReport(
            n_rows=len(raw),
            n_columns=raw.shape[1],
            missing_features=[c for c in self.feature_columns if c not in raw.columns],
            target_present=self.has_target(raw),
            target_required=require_target,
            columns_with_nulls=[c for c in present if raw[c].isna().any()],
            extra_columns=[c for c in raw.columns if c not in expected],
        )

    def validate(self, raw: pd.DataFrame) -> None:
        """Fail early with a readable message instead of a cryptic model error."""
        missing = [c for c in self.feature_columns if c not in raw.columns]
        if missing:
            raise SchemaError(f"Missing required column(s): {', '.join(missing)}")
        if raw.empty:
            raise SchemaError("The table has no rows.")
        has_nulls = raw[self.feature_columns].isna().any()
        if has_nulls.any():
            raise SchemaError(f"Missing values found in: {', '.join(has_nulls[has_nulls].index)}")

    def prepare_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return the feature table with consistent dtypes (extra columns are ignored)."""
        self.validate(raw)
        features = raw[self.feature_columns].copy()

        for column in self.numeric_features:
            try:
                features[column] = pd.to_numeric(features[column])
            except (ValueError, TypeError) as error:
                raise SchemaError(f"Column '{column}' must be numeric.") from error

        # integer codes -> strings, so the encoder treats them as categories
        for column in self.coded_categorical_features:
            features[column] = features[column].astype(int).astype(str)

        features["Weekend"] = _to_binary(features["Weekend"])
        features["Month"] = features["Month"].astype(str)
        features["VisitorType"] = features["VisitorType"].astype(str)
        return features

    def prepare_target(self, raw: pd.DataFrame) -> pd.Series:
        """Return the target as integers (1 = purchase, 0 = no purchase)."""
        if not self.has_target(raw):
            raise SchemaError(f"Target column '{self.target}' not found.")
        return _to_binary(raw[self.target]).rename(self.target)

    def prepare(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """Features plus target (target is None when the table has no target column)."""
        features = self.prepare_features(raw)
        target = self.prepare_target(raw) if self.has_target(raw) else None
        return features, target
