"""Reusable matplotlib figures (used by the notebook and the Streamlit app).

Functions only *draw*: they receive ready-made numbers and return a Figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from src import config

# Fixed colour order: a model keeps the same colour in every chart.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL_GRAY = "#8a8984"
TEXT_DARK = "#0b0b0b"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
})


def color_map(names: list[str]) -> dict[str, str]:
    """Assign colours to names in order (never cycled: max 8 series)."""
    return dict(zip(names, SERIES_COLORS))


def plot_confusion_matrix(cm: np.ndarray, title: str = "", ax=None) -> Figure:
    """2x2 confusion matrix with counts written in each cell."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, f"{value:,}", ha="center", va="center", fontsize=11,
                color="white" if value > cm.max() / 2 else TEXT_DARK)
    labels = [config.CLASS_LABELS[k] for k in sorted(config.CLASS_LABELS)]
    ax.set_xticks([0, 1], [f"Pred: {label}" for label in labels], fontsize=8.5)
    ax.set_yticks([0, 1], [f"True: {label}" for label in labels], fontsize=8.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(False)
    return fig


def plot_confusion_matrices(matrices: dict[str, np.ndarray]) -> Figure:
    """One confusion matrix per model, side by side."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.7 * len(matrices), 3.6))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        plot_confusion_matrix(cm, name, ax=ax)
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true: pd.Series, scores: dict[str, np.ndarray], auc: dict[str, float]) -> Figure:
    """ROC curve of every model on one chart, plus the random-guess diagonal."""
    colors = color_map(list(scores))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name}  (AUC = {auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=NEUTRAL_GRAY, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    return fig


def plot_metric_heatmap(table: pd.DataFrame, title: str = "Test-set metrics (bold = best in column)",
                        as_percent: bool = False) -> Figure:
    """Models x metrics grid; the best value in each column is bold. `as_percent` labels 0.882 as 88.20%."""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(table) + 1.4))
    ax.imshow(table.values, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.xaxis.tick_top()
    ax.grid(False)
    column_best = table.max()
    for (row, col), value in np.ndenumerate(table.values):
        label = f"{value * 100:.2f}%" if as_percent else f"{value:.3f}"
        ax.text(col, row, label, ha="center", va="center", fontsize=10,
                color="white" if value > 0.78 else TEXT_DARK,
                fontweight="bold" if value == column_best.iloc[col] else "normal")
    ax.set_title(title, loc="left", pad=28)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, title: str) -> Figure:
    """Horizontal bars, largest at the top, value printed at the bar end."""
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(importance) + 0.9))
    ordered = importance.sort_values()
    ax.barh(ordered.index, ordered.values, color=SERIES_COLORS[0], height=0.6)
    for i, value in enumerate(ordered.values):
        ax.text(value + ordered.max() * 0.01, i, f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Importance")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
