"""End-to-end training workflow.

`TrainingWorkflow` only *coordinates*: it receives its collaborators (preparer,
tuner, evaluator, model specs) through the constructor (Dependency Inversion) and
delegates the real work to them. Saving is a separate step (`save_artifacts`),
so a run can be inspected, for example in the notebook, before anything is written.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import DataPreparer, remove_duplicates, split_train_test
from src.evaluation import EvaluationResult, ModelEvaluator, comparison_table
from src.models import MODEL_SPECS, ModelSpec
from src.persistence import ModelRepository
from src.training import HyperparameterTuner, TrainedModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Everything produced by one training run (nothing is written to disk yet)."""

    trained: list[TrainedModel]
    evaluations: dict[str, EvaluationResult]   # test-set results, keyed by model name
    comparison: pd.DataFrame                   # models x metrics (test set)
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series

    def tuning_summary(self) -> pd.DataFrame:
        """Best hyper-parameters and cross-validation scores per model."""
        rows = [{
            "Model": m.name,
            "Best hyper-parameters": m.best_params,
            **{f"CV {metric.upper()}": score for metric, score in m.cv_scores.items()},
            "Tuning time (s)": m.tuning_seconds,
        } for m in self.trained]
        return pd.DataFrame(rows).set_index("Model")


class TrainingWorkflow:
    """dedupe -> split -> clean -> tune every model -> evaluate on the held-out test set."""

    def __init__(
        self,
        preparer: DataPreparer,
        tuner: HyperparameterTuner,
        evaluator: ModelEvaluator,
        specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    ):
        self.preparer = preparer
        self.tuner = tuner
        self.evaluator = evaluator
        self.specs = specs

    def run(self, raw: pd.DataFrame) -> TrainingResult:
        unique_rows = remove_duplicates(raw)
        train_raw, test_raw = split_train_test(unique_rows, target=self.preparer.target)
        X_train, y_train = self.preparer.prepare(train_raw)
        X_test, y_test = self.preparer.prepare(test_raw)
        logger.info("Rows after de-duplication: %d  (train %d / test %d)",
                    len(unique_rows), len(train_raw), len(test_raw))

        trained = [self.tuner.tune(spec, X_train, y_train) for spec in self.specs]
        evaluations = {m.name: self.evaluator.evaluate(m.pipeline, X_test, y_test) for m in trained}

        return TrainingResult(
            trained=trained,
            evaluations=evaluations,
            comparison=comparison_table(evaluations),
            train_raw=train_raw,
            test_raw=test_raw,
            X_test=X_test,
            y_test=y_test,
        )


def build_metadata(result: TrainingResult, preparer: DataPreparer) -> dict[str, Any]:
    """Describe the saved models: schema, split, tuned parameters, scores and library versions."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "problem": "Binary classification: will the session end in a purchase?",
        "target": preparer.target,
        "class_labels": config.CLASS_LABELS,
        "numeric_features": preparer.numeric_features,
        "categorical_features": preparer.categorical_features,
        "split": {
            "train_rows": len(result.train_raw),
            "test_rows": len(result.test_raw),
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "stratified": True,
        },
        "tuning": {"method": "GridSearchCV", "cv_folds": config.CV_FOLDS, "refit_metric": config.TUNING_METRIC},
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "models": [{
            "key": m.key,
            "name": m.name,
            "file": f"{m.key}.joblib",
            "best_params": m.best_params,
            "cv_scores": m.cv_scores,
            "test_metrics": result.comparison.loc[m.name].to_dict(),
        } for m in result.trained],
    }


def save_artifacts(
    result: TrainingResult,
    preparer: DataPreparer,
    repository: ModelRepository,
    test_data_path: Path = config.TEST_DATA_PATH,
) -> list[Path]:
    """Write every model, the metadata, the comparison table and test_data.csv."""
    written = [repository.save_model(m.key, m.pipeline) for m in result.trained]
    written.append(repository.save_comparison(result.comparison))
    written.append(repository.save_metadata(build_metadata(result, preparer)))

    # test rows in the original raw format -> can be uploaded to the app as-is
    result.test_raw.to_csv(test_data_path, index=False)
    written.append(Path(test_data_path))
    for path in written:
        logger.info("Saved %s", path)
    return written
