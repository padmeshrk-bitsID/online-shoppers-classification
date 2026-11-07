"""Classifier specifications.

Each model is described by a small class that knows only two things:
how to build its estimator and which hyper-parameters to search.

* Open/Closed: adding a sixth model means adding one subclass to `MODEL_SPECS`;
  the trainer, evaluator, persistence layer and app need no changes.
* Liskov: every spec is interchangeable wherever a `ModelSpec` is expected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config


class ModelSpec(ABC):
    """Contract that every classifier specification must follow."""

    #: human-readable name shown in reports and in the app
    name: str
    #: file-safe identifier -> saved as model/<key>.joblib
    key: str

    @abstractmethod
    def build_estimator(self) -> ClassifierMixin:
        """Return a new, unfitted estimator with fixed (non-tuned) settings."""

    @abstractmethod
    def param_grid(self) -> dict[str, list[Any]]:
        """Hyper-parameters to search, using the estimator's own parameter names."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key='{self.key}')"


class LogisticRegressionSpec(ModelSpec):
    name, key = "Logistic Regression", "logistic_regression"

    def build_estimator(self) -> ClassifierMixin:
        return LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [0.001, 0.003, 0.01, 0.1, 1, 10, 100],  # inverse L2 strength
            "class_weight": [None, "balanced"],          # balanced -> up-weight buyers
        }


class DecisionTreeSpec(ModelSpec):
    name, key = "Decision Tree", "decision_tree"

    def build_estimator(self) -> ClassifierMixin:
        return DecisionTreeClassifier(random_state=config.RANDOM_STATE)

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "max_depth": [3, 4, 5, 6, 8, 10, None],      # depth & leaf size limit over-fitting
            "min_samples_leaf": [1, 10, 30, 60, 100, 150],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"],
        }


class KNearestNeighborsSpec(ModelSpec):
    name, key = "KNN", "knn"

    def build_estimator(self) -> ClassifierMixin:
        return KNeighborsClassifier()

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "n_neighbors": [5, 9, 15, 25, 35, 51],       # small k = high variance, large k = high bias
            "weights": ["uniform", "distance"],
            "p": [1, 2],                                 # 1 = Manhattan, 2 = Euclidean
        }


class GaussianNaiveBayesSpec(ModelSpec):
    """Gaussian (not Multinomial) NB: features are continuous and can be negative after scaling."""

    name, key = "Naive Bayes (Gaussian)", "naive_bayes"

    def build_estimator(self) -> ClassifierMixin:
        return GaussianNB()

    def param_grid(self) -> dict[str, list[Any]]:
        return {"var_smoothing": list(np.logspace(-9, 2, 12))}  # variance stabiliser


class SupportVectorMachineSpec(ModelSpec):
    """RBF-kernel SVM. AUC uses `decision_function`, so `probability=True` is not needed."""

    name, key = "SVM", "svm"

    def build_estimator(self) -> ClassifierMixin:
        return SVC(kernel="rbf", random_state=config.RANDOM_STATE)

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [0.1, 1, 10],                           # margin softness
            "gamma": ["scale", 0.001, 0.003, 0.01, 0.1], # RBF width
            "class_weight": [None, "balanced"],
        }


# Registry: the single place that decides which models are trained.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    LogisticRegressionSpec(),
    DecisionTreeSpec(),
    KNearestNeighborsSpec(),
    GaussianNaiveBayesSpec(),
    SupportVectorMachineSpec(),
)
