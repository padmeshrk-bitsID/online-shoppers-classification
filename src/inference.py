"""Prediction service used by the Streamlit app.

The app (UI) talks only to `PredictionService`; the service hides where models
come from (repository), how input is cleaned (preparer) and how it is scored
(evaluator). Each of those can be swapped without touching the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data import DataPreparer
from src.evaluation import (
    EvaluationResult,
    ModelEvaluator,
    comparison_table,
    positive_class_scores,
)
from src.persistence import ModelRepository

PREDICTED_COLUMN = "Predicted"
PREDICTED_LABEL_COLUMN = "Predicted label"
SCORE_COLUMN = "Purchase score"
ACTUAL_LABEL_COLUMN = "Actual label"
CORRECT_COLUMN = "Correct"


@dataclass(frozen=True)
class PredictionOutput:
    """Predictions for every input row, plus metrics when the true label was supplied."""

    predictions: pd.DataFrame
    evaluation: EvaluationResult | None


class PredictionService:
    """Load saved models on demand and run them on raw tables."""

    def __init__(self, repository: ModelRepository, preparer: DataPreparer, evaluator: ModelEvaluator):
        self.repository = repository
        self.preparer = preparer
        self.evaluator = evaluator
        self._metadata = repository.load_metadata()
        self._models: dict[str, Pipeline] = {}     # simple in-memory cache

    # ---------------------------------------------------------- catalogue
    def model_names(self) -> list[str]:
        return [entry["name"] for entry in self._metadata["models"]]

    def model_info(self, name: str) -> dict[str, Any]:
        return next(entry for entry in self._metadata["models"] if entry["name"] == name)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def recommended_model(self, metric: str = "MCC") -> str:
        """Model with the best held-out test score (MCC is the fairest single metric here)."""
        return max(self._metadata["models"], key=lambda entry: entry["test_metrics"][metric])["name"]

    def _model(self, name: str) -> Pipeline:
        if name not in self._models:
            self._models[name] = self.repository.load_model(self.model_info(name)["key"])
        return self._models[name]

    # ---------------------------------------------------------- prediction
    def run(self, name: str, raw: pd.DataFrame) -> PredictionOutput:
        """Predict every row; evaluate too if the table contains the target column."""
        X, y = self.preparer.prepare(raw)
        model = self._model(name)

        predictions = raw.copy()
        predictions[PREDICTED_COLUMN] = model.predict(X)
        predictions[PREDICTED_LABEL_COLUMN] = predictions[PREDICTED_COLUMN].map(config.CLASS_LABELS)
        predictions[SCORE_COLUMN] = positive_class_scores(model, X).round(4)
        if y is not None:
            predictions[ACTUAL_LABEL_COLUMN] = y.map(config.CLASS_LABELS).to_numpy()
            predictions[CORRECT_COLUMN] = (y.to_numpy() == predictions[PREDICTED_COLUMN].to_numpy())

        evaluation = self.evaluator.evaluate(model, X, y) if y is not None else None
        return PredictionOutput(predictions=predictions, evaluation=evaluation)

    def compare_all(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Metrics of every saved model on the same labelled table."""
        X, y = self.preparer.prepare(raw)
        if y is None:
            raise ValueError(f"Comparison needs the '{self.preparer.target}' column.")
        results = {name: self.evaluator.evaluate(self._model(name), X, y) for name in self.model_names()}
        return comparison_table(results)
