"""Small, pure HTML building blocks for the app.

Every function takes plain data and returns an HTML string (rendered with
`st.markdown(..., unsafe_allow_html=True)`). Rules followed throughout:
  * any text that could come from a user (e.g. a file name) is HTML-escaped;
  * output is a single line, because Markdown would turn indented lines into code blocks.
"""
from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd

from src.ui.content import METRIC_HELP

# Performance bands: (lower bound, css class, symbol, label, range text). Colour is never used alone.
BANDS = (
    (0.80, "osc-good", "▲", "Good", "80% and above"),
    (0.60, "osc-moderate", "●", "Moderate", "60% to 80%"),
    (float("-inf"), "osc-low", "▼", "Low", "below 60%"),
)

# Outcome colours (validated for colour-blind separation): blues = correct, oranges = mistakes
OUTCOME_COLORS = {"tn": "#174f94", "tp": "#3d8fe6", "fp": "#eb6834", "fn": "#a3401a"}


def _is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_percent(value: float | None, decimals: int = 2) -> str:
    """0.882 -> '88.20%': percentages are easier to read than decimals. Missing -> 'n/a'."""
    return "n/a" if _is_missing(value) else f"{value * 100:.{decimals}f}%"


def _pct(value: float) -> float:
    """0-1 score -> 0-100 position, clamped (MCC can be negative)."""
    return max(0.0, min(1.0, value)) * 100


def _stagger(index: int, step_ms: int = 70) -> str:
    """Inline style that delays an element's entry animation (cards appear one after another)."""
    return f"animation-delay:{index * step_ms}ms"


# ----------------------------------------------------------------------------- page furniture
def author_credit(name: str, identifier: str) -> str:
    """'Made by <name> · <id>' line (used in the header)."""
    return (f'<div class="osc-byline"><span>👤 Made by <b>{escape(name)}</b></span>'
            f'<span class="osc-sep">·</span><span>{escape(identifier)}</span></div>')


def hero(title: str, tagline: str, chips: list[str], byline: str = "") -> str:
    """Page header; `byline` is optional ready-made HTML (e.g. from `author_credit`)."""
    chip_html = "".join(f'<span class="osc-chip">{escape(chip)}</span>' for chip in chips)
    return f'<div class="osc-hero"><h1>{escape(title)}</h1><p>{escape(tagline)}</p>{chip_html}{byline}</div>'


def watermark(name: str, identifier: str) -> str:
    """Small badge pinned to the bottom-right corner of the window (stays visible while scrolling)."""
    return f'<div class="osc-watermark">Made by {escape(name)} · {escape(identifier)}</div>'


def step_cards(steps: list[tuple[str, str, str]]) -> str:
    """Numbered 'how it works' cards: (icon, title, text)."""
    cards = "".join(
        f'<div class="osc-card" style="{_stagger(i)}"><span class="osc-step-icon">{icon}</span>'
        f'<h4><span class="osc-step-num">{i + 1}</span>{escape(title)}</h4><p>{escape(text)}</p></div>'
        for i, (icon, title, text) in enumerate(steps)
    )
    return f'<div class="osc-grid-3">{cards}</div>'


def empty_state(icon: str, title: str, text: str) -> str:
    return (f'<div class="osc-empty"><div class="osc-empty-icon">{icon}</div>'
            f'<h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def badge(text: str, tone: str = "blue") -> str:
    """Pill label; tone is one of blue / green / orange."""
    return f'<span class="osc-badge {escape(tone)}">{escape(text)}</span>'


def section_nav(links: list[tuple[str, str]]) -> str:
    """Row of pill links that scroll to the page sections: (anchor, label)."""
    pills = "".join(f'<a href="#{escape(anchor)}">{escape(label)}</a>' for anchor, label in links)
    return f'<div class="osc-nav">{pills}</div>'


def callout(html_text: str) -> str:
    """Highlighted sentence. `html_text` must already be safe (built from our own numbers)."""
    return f'<div class="osc-callout">{html_text}</div>'


# ----------------------------------------------------------------------------- performance bands
def performance_band(value: float | None) -> tuple[str, str, str]:
    """(css class, symbol, label) of the band a 0-1 score falls in; ('', '-', 'n/a') if missing."""
    if _is_missing(value):
        return "", "-", "n/a"
    return next((css, symbol, label) for lower, css, symbol, label, _ in BANDS if value >= lower)


def band_chip(value: float | None) -> str:
    css, symbol, label = performance_band(value)
    return f'<span class="osc-band {css}"><span class="osc-dot"></span>{symbol} {label}</span>'


def band_legend() -> str:
    items = "".join(f'<span class="osc-band {css}"><span class="osc-dot"></span>{symbol} {label}: {escape(text)}</span>'
                    for _, css, symbol, label, text in BANDS)
    return f'<div class="osc-legend"><b>Colour key</b>{items}</div>'


# ----------------------------------------------------------------------------- metrics
def metric_cards(metrics: dict[str, float], best: dict[str, bool] | None = None) -> str:
    """One animated card per metric: value, a bar that grows to the value, and a tooltip.

    `best` marks metrics on which this model beats (or ties) every other model.
    """
    best = best or {}
    cards = []
    for i, (name, value) in enumerate(metrics.items()):
        meaning, formula = METRIC_HELP.get(name, ("", ""))
        tooltip = escape(f"{meaning} Formula: {formula}", quote=True)
        shown = format_percent(value)
        width = 0.0 if _is_missing(value) else _pct(value)        # MCC < 0 -> empty bar
        band_css = performance_band(value)[0]
        star = '<span class="osc-best">best</span>' if best.get(name) else ""
        cards.append(
            f'<div class="osc-card osc-metric {band_css}" title="{tooltip}" style="{_stagger(i)}">'
            f'{star}<div class="osc-label">{escape(name)}</div><div class="osc-value">{shown}</div>'
            f'<div class="osc-bar"><span style="width:{width:.1f}%;{_stagger(i, 90)}"></span></div>'
            f'<div style="margin-top:7px">{band_chip(value)}</div></div>'
        )
    return f'<div class="osc-grid">{"".join(cards)}</div>'


def best_model_cards(table: pd.DataFrame) -> str:
    """For each metric: which model scored highest on this data (ties listed together)."""
    cards = []
    for i, metric in enumerate(table.columns):
        column = table[metric].dropna()
        if column.empty:
            winner, value = "not defined for this data", ""
        else:
            winner = " & ".join(column.index[column == column.max()])
            value = format_percent(column.max())
        cards.append(
            f'<div class="osc-card" style="{_stagger(i)}"><div class="osc-muted">BEST {escape(metric.upper())}</div>'
            f'<h4>{escape(winner)}</h4><p>{value}</p></div>'
        )
    return f'<div class="osc-grid">{"".join(cards)}</div>'


# ----------------------------------------------------------------------------- comparison graphics
def _tip(text: str) -> str:
    return f'<span class="osc-tooltip">{escape(text)}</span>'


def position_chart(table: pd.DataFrame, selected: str) -> str:
    """Per metric, on a 0-100 % scale: the selected model (dot coloured by its band) against
    the other models (grey dots) and the range they span (grey bar). Every dot has a hover tooltip."""
    rows = []
    for metric in table.columns:
        column = table[metric].dropna()
        value = table[metric].get(selected)
        band_css, symbol, label = performance_band(value)
        marks = []
        if not column.empty:
            low, high = _pct(column.min()), _pct(column.max())
            marks.append(f'<div class="osc-range" style="left:{low:.1f}%;width:{max(high - low, 0.8):.1f}%"></div>')
            marks += [f'<div class="osc-mark other osc-tip" style="left:{_pct(score):.1f}%">'
                      f'{_tip(f"{name}: {format_percent(score)}")}</div>'
                      for name, score in column.items() if name != selected]
        if not _is_missing(value):
            marks.append(f'<div class="osc-mark selected osc-tip" style="left:{_pct(value):.1f}%">'
                         f'{_tip(f"{selected} (selected): {format_percent(value)} · {symbol} {label}")}</div>')
        rows.append(
            f'<div class="osc-pos-row {band_css}"><div class="osc-pos-name">{escape(metric)}</div>'
            f'<div class="osc-track">{"".join(marks)}</div>'
            f'<div class="osc-pos-value">{format_percent(value)}</div></div>'
        )
    scale = "".join(f"<span>{tick}%</span>" for tick in (0, 25, 50, 75, 100))
    axis = f'<div class="osc-axis"><div></div><div class="osc-axis-scale">{scale}</div><div></div></div>'
    return f'<div class="osc-card">{"".join(rows)}{axis}</div>'


def ranking_bars(table: pd.DataFrame, metric: str, selected: str) -> str:
    """All models ranked on one metric; bars coloured by band, selected model outlined, best gets a trophy."""
    column = table[metric].sort_values(ascending=False)          # missing values go last
    best = column.max()
    meaning = METRIC_HELP.get(metric, ("", ""))[0]
    rows = []
    for i, (name, value) in enumerate(column.items()):
        band_css, symbol, label = performance_band(value)
        width = 0.0 if _is_missing(value) else _pct(value)
        trophy = " 🏆" if not _is_missing(value) and value == best else ""
        is_selected = " selected" if name == selected else ""
        suffix = " (selected)" if is_selected else ""
        rows.append(
            f'<div class="osc-rank-row {band_css}{is_selected}">'
            f'<div class="osc-rank-name">{escape(name)}{trophy}</div>'
            f'<div class="osc-rank-track osc-tip"><div class="osc-fill" style="width:{width:.1f}%;{_stagger(i, 90)}"></div>'
            f'{_tip(f"{name}{suffix}: {format_percent(value)} · {symbol} {label}")}</div>'
            f'<div class="osc-rank-value">{format_percent(value)}</div></div>'
        )
    return f'<div class="osc-card"><div class="osc-muted">{escape(metric)}: {escape(meaning)}</div>{"".join(rows)}</div>'


def outcome_mix(cm: np.ndarray) -> str:
    """Stacked bar of where every prediction landed (correct vs mistakes), with a labelled legend."""
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    parts = [  # key, label, count, explanation
        ("tn", "Correct: no purchase", tn, "predicted no purchase, didn't buy"),
        ("tp", "Correct: purchase", tp, "predicted purchase, did buy"),
        ("fp", "False alarm", fp, "predicted purchase, didn't buy"),
        ("fn", "Missed buyer", fn, "predicted no purchase, but bought"),
    ]
    segments, legend = [], []
    for i, (key, label, count, explanation) in enumerate(parts):
        share = count / total if total else 0.0
        color = OUTCOME_COLORS[key]
        if count:
            segments.append(
                f'<div class="osc-seg osc-tip" style="width:{share * 100:.2f}%;background:{color};{_stagger(i, 120)}">'
                f'{_tip(f"{label}: {count:,} sessions ({share:.1%}), {explanation}")}</div>')
        legend.append(f'<span class="osc-band"><span class="osc-dot" style="background:{color}"></span>'
                      f'{escape(label)}: <b>{count:,}</b> ({share:.1%})</span>')
    return (f'<div class="osc-card"><div class="osc-muted">WHERE THE {total:,} PREDICTIONS LANDED</div>'
            f'<div class="osc-stack">{"".join(segments)}</div><div class="osc-legend">{"".join(legend)}</div></div>')


# ----------------------------------------------------------------------------- confusion matrix
def confusion_matrix_html(cm: np.ndarray, labels: tuple[str, str] = ("No purchase", "Purchase")) -> str:
    """2x2 matrix as coloured cells: correct predictions in blue, mistakes in orange."""
    (tn, fp), (fn, tp) = cm
    actual_negative, actual_positive = tn + fp, fn + tp

    def cell(count: int, row_total: int, actual: str, tag: str, correct: bool, order: int) -> str:
        share = f"{count / row_total:.1%} of actual {actual.lower()}" if row_total else "-"
        kind = "correct" if correct else "wrong"
        return (f'<div class="osc-cm-cell {kind}" style="{_stagger(order, 90)}">'
                f'<div class="osc-count">{count:,}</div><div class="osc-share">{share}</div>'
                f'<div class="osc-tag">{tag}</div></div>')

    negative, positive = (escape(label) for label in labels)
    return (
        '<div class="osc-card"><div class="osc-cm">'
        f'<div></div><div class="osc-cm-head">Predicted<b>{negative}</b></div>'
        f'<div class="osc-cm-head">Predicted<b>{positive}</b></div>'
        f'<div class="osc-cm-side">Actual<b>{negative}</b></div>'
        f'{cell(tn, actual_negative, negative, "True negative", True, 0)}'
        f'{cell(fp, actual_negative, negative, "False positive", False, 1)}'
        f'<div class="osc-cm-side">Actual<b>{positive}</b></div>'
        f'{cell(fn, actual_positive, positive, "False negative", False, 2)}'
        f'{cell(tp, actual_positive, positive, "True positive", True, 3)}'
        '</div></div>'
    )


def describe_confusion(cm: np.ndarray) -> str:
    """One plain-English sentence summarising the confusion matrix."""
    (tn, fp), (fn, tp) = cm
    buyers, non_buyers = tp + fn, tn + fp
    caught = f"{tp / buyers:.1%}" if buyers else "n/a"
    false_alarm = f"{fp / non_buyers:.1%}" if non_buyers else "n/a"
    return (f"Caught <b>{tp:,} of {buyers:,}</b> real buyers ({caught}) and wrongly flagged "
            f"<b>{fp:,} of {non_buyers:,}</b> non-buyers as buyers ({false_alarm}).")


# ----------------------------------------------------------------------------- setup panel
def model_card(name: str, blurb: str, params: dict, cv_scores: dict, file_name: str, recommended: bool) -> str:
    """Selected model: description, tuned hyper-parameters and where it was loaded from."""
    star = badge("Recommended", "green") if recommended else ""
    chips = "".join(f'<span class="osc-param">{escape(str(k))}={escape(str(v))}</span>' for k, v in params.items())
    cv = " · ".join(f"CV {escape(k.replace('roc_auc', 'AUC').upper())} {format_percent(v)}"
                    for k, v in cv_scores.items())
    return (
        f'<div class="osc-card osc-model"><h4>{escape(name)} {star}</h4><p>{escape(blurb)}</p>'
        f'<div style="margin-top:8px">{chips}</div>'
        f'<div class="osc-hint">{cv}</div>'
        f'<div class="osc-hint">Loaded from <code>model/{escape(file_name)}</code>: no retraining.</div></div>'
    )
