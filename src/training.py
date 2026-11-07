"""Hyper-parameter tuning.

`HyperparameterTuner` depends only on the `ModelSpec` abstraction and on an
injected preprocessor factory (Dependency Inversion), so it can tune any model
with any preprocessing without modification.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.models import ModelSpec
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

PREPROCESS_STEP = "prep"
CLASSIFIER_STEP = "clf"

_UNKNOWN_CATEGORY_WARNING = "Found unknown categories"


def silence_expected_warnings() -> None:
    """Hide one harmless, expected warning.

    A rare Browser / TrafficType code can be absent from a CV training fold; the
    encoder then maps it to the 'infrequent' column (intended behaviour) but warns
    every time. Call this once from an entry point (script / notebook / app).
    The environment variable carries the filter into parallel CV worker processes.
    """
    warnings.filterwarnings("ignore", message=_UNKNOWN_CATEGORY_WARNING)
    os.environ["PYTHONWARNINGS"] = f"ignore:{_UNKNOWN_CATEGORY_WARNING}:UserWarning"


@dataclass(frozen=True)
class TrainedModel:
    """A tuned, fitted pipeline plus what we learned while tuning it."""

    key: str
    name: str
    pipeline: Pipeline
    best_params: dict[str, Any]
    cv_scores: dict[str, float] = field(default_factory=dict)
    tuning_seconds: float = 0.0


def _to_plain_python(value: Any) -> Any:
    """numpy scalars -> plain Python, so parameters print cleanly and serialise to JSON."""
    return value.item() if isinstance(value, np.generic) else value


class HyperparameterTuner:
    """Grid-search a model specification with stratified k-fold cross-validation."""

    def __init__(
        self,
        preprocessor_factory: Callable[[], TransformerMixin] = build_preprocessor,
        cv_folds: int = config.CV_FOLDS,
        refit_metric: str = config.TUNING_METRIC,
        extra_metrics: tuple[str, ...] = ("roc_auc",),
        random_state: int = config.RANDOM_STATE,
        n_jobs: int = -1,
    ):
        self.preprocessor_factory = preprocessor_factory
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.refit_metric = refit_metric
        self.scoring = {name: name for name in (refit_metric, *extra_metrics)}
        self.n_jobs = n_jobs

    def build_pipeline(self, spec: ModelSpec) -> Pipeline:
        """Preprocessing + classifier in one object, so CV never leaks test-fold statistics."""
        return Pipeline([
            (PREPROCESS_STEP, self.preprocessor_factory()),
            (CLASSIFIER_STEP, spec.build_estimator()),
        ])

    @staticmethod
    def _prefixed_grid(spec: ModelSpec) -> dict[str, list[Any]]:
        """Specs use plain parameter names; the pipeline needs the 'clf__' prefix."""
        return {f"{CLASSIFIER_STEP}__{name}": values for name, values in spec.param_grid().items()}

    def tune(self, spec: ModelSpec, X: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """Search the grid, refit the best combination on all of X, and return it."""
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=self.build_pipeline(spec),
            param_grid=self._prefixed_grid(spec),
            scoring=self.scoring,
            refit=self.refit_metric,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        search.fit(X, y)

        best = search.best_index_
        best_params = {
            name.removeprefix(f"{CLASSIFIER_STEP}__"): _to_plain_python(value)
            for name, value in search.best_params_.items()
        }
        cv_scores = {
            metric: round(float(search.cv_results_[f"mean_test_{metric}"][best]), 4)
            for metric in self.scoring
        }
        elapsed = round(time.perf_counter() - started, 1)
        logger.info("%-24s tuned in %6.1fs  best=%s", spec.name, elapsed, best_params)

        return TrainedModel(
            key=spec.key,
            name=spec.name,
            pipeline=search.best_estimator_,
            best_params=best_params,
            cv_scores=cv_scores,
            tuning_seconds=elapsed,
        )
