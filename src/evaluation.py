"""Evaluation metrics.

Metrics are plain data (`Metric` objects), so adding a seventh metric means
adding one line to `DEFAULT_METRICS`; `ModelEvaluator` itself never changes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


@dataclass(frozen=True)
class Metric:
    """A named metric. `needs_scores=True` -> called with scores instead of hard labels."""

    name: str
    func: Callable[..., float]
    needs_scores: bool = False


DEFAULT_METRICS: tuple[Metric, ...] = (
    Metric("Accuracy", accuracy_score),
    Metric("AUC", roc_auc_score, needs_scores=True),
    Metric("Precision", partial(precision_score, zero_division=0)),
    Metric("Recall", partial(recall_score, zero_division=0)),
    Metric("F1", partial(f1_score, zero_division=0)),
    Metric("MCC", matthews_corrcoef),
)
METRIC_NAMES = [metric.name for metric in DEFAULT_METRICS]


def positive_class_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Score for the 'purchase' class: probability if available, else the SVM decision value.

    AUC and ROC only need a ranking of the rows, so either score works.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


@dataclass(frozen=True)
class EvaluationResult:
    """Everything needed to report one model's performance on one dataset."""

    metrics: dict[str, float]
    confusion: np.ndarray            # [[TN, FP], [FN, TP]]
    report: pd.DataFrame             # per-class precision / recall / F1 / support
    y_pred: np.ndarray
    y_score: np.ndarray


class ModelEvaluator:
    """Computes metrics, confusion matrix and classification report for a fitted model."""

    def __init__(self, metrics: tuple[Metric, ...] = DEFAULT_METRICS,
                 class_labels: dict[int, str] = config.CLASS_LABELS):
        self.metrics = metrics
        self.class_labels = class_labels

    def _score(self, metric: Metric, y_true, y_pred, y_score) -> float:
        try:
            return float(metric.func(y_true, y_score if metric.needs_scores else y_pred))
        except ValueError:
            # e.g. AUC is undefined when the uploaded file contains only one class
            return float("nan")

    def _report_table(self, y_true, y_pred, labels: list[int]) -> pd.DataFrame:
        """Per-class precision / recall / F1 / support, laid out like scikit-learn's text report."""
        names = [self.class_labels[label] for label in labels]
        report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                       output_dict=True, zero_division=0)
        accuracy = report.pop("accuracy", float("nan"))
        table = pd.DataFrame(report).T
        # accuracy is a single number: show it under F1 with the total support (as sklearn prints it)
        table.loc["accuracy"] = [np.nan, np.nan, accuracy, table.loc["macro avg", "support"]]
        table = table.loc[[*names, "accuracy", "macro avg", "weighted avg"]]
        table["support"] = table["support"].astype(int)
        return table

    def evaluate(self, model, X: pd.DataFrame, y_true: pd.Series) -> EvaluationResult:
        y_pred = model.predict(X)
        y_score = positive_class_scores(model, X)

        metrics = {m.name: self._score(m, y_true, y_pred, y_score) for m in self.metrics}
        labels = sorted(self.class_labels)

        return EvaluationResult(
            metrics=metrics,
            confusion=confusion_matrix(y_true, y_pred, labels=labels),
            report=self._report_table(y_true, y_pred, labels),
            y_pred=y_pred,
            y_score=y_score,
        )


def comparison_table(results: dict[str, EvaluationResult], decimals: int = 4) -> pd.DataFrame:
    """One row per model, one column per metric."""
    table = pd.DataFrame({name: result.metrics for name, result in results.items()}).T
    table.index.name = "ML Model"
    return table.round(decimals)
