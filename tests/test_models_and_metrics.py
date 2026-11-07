"""Tests for model specifications and the evaluation metrics."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluation import METRIC_NAMES, ModelEvaluator
from src.models import MODEL_SPECS, ModelSpec


def test_registry_has_the_five_required_models():
    assert [spec.name for spec in MODEL_SPECS] == [
        "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes (Gaussian)", "SVM"]
    assert len({spec.key for spec in MODEL_SPECS}) == len(MODEL_SPECS)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: s.key)
def test_every_grid_parameter_exists_on_its_estimator(spec: ModelSpec):
    estimator = spec.build_estimator()
    assert set(spec.param_grid()) <= set(estimator.get_params())


def test_evaluator_returns_the_six_metrics_in_order():
    X = pd.DataFrame({"x": np.arange(20)})
    y = pd.Series([0] * 10 + [1] * 10)
    model = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)
    result = ModelEvaluator().evaluate(model, X, y)
    assert list(result.metrics) == METRIC_NAMES
    assert result.confusion.shape == (2, 2)


def test_perfect_model_scores_one():
    X = pd.DataFrame({"x": [0, 0, 1, 1]})
    y = pd.Series([0, 0, 1, 1])

    class Perfect:                       # minimal stand-in exposing the sklearn API
        def predict(self, data):
            return data["x"].to_numpy()

        def predict_proba(self, data):
            p = data["x"].to_numpy(dtype=float)
            return np.column_stack([1 - p, p])

    metrics = ModelEvaluator().evaluate(Perfect(), X, y).metrics
    assert all(value == pytest.approx(1.0) for value in metrics.values())


@pytest.mark.filterwarnings("ignore::UserWarning")   # sklearn warns that AUC is undefined: expected here
def test_auc_is_nan_when_uploaded_data_has_one_class():
    # model trained on both classes, but the evaluated file contains only non-buyers
    model = DummyClassifier(strategy="prior").fit(pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]))
    X_upload = pd.DataFrame({"x": [1, 2, 3]})
    y_upload = pd.Series([0, 0, 0])
    assert np.isnan(ModelEvaluator().evaluate(model, X_upload, y_upload).metrics["AUC"])
