"""Tests for the upload checklist and the HTML building blocks used by the app."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import DataPreparer
from src.ui import components as ui
from src.ui.content import COLUMN_GUIDE, METRIC_HELP


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH).head(20)


# ----------------------------------------------------------------- schema checklist
def test_complete_table_passes_inspection(sample):
    report = DataPreparer().inspect(sample)
    assert report.ok and report.n_rows == 20 and not report.extra_columns


def test_missing_target_is_a_problem_when_required(sample):
    report = DataPreparer().inspect(sample.drop(columns=config.TARGET))
    assert not report.ok and any("Revenue" in p for p in report.problems)
    assert DataPreparer().inspect(sample.drop(columns=config.TARGET), require_target=False).ok


def test_extra_columns_and_empty_cells_are_reported(sample):
    table = sample.assign(SessionID=1)
    table.loc[0, "ExitRates"] = None
    report = DataPreparer().inspect(table)
    assert report.extra_columns == ["SessionID"]
    assert report.columns_with_nulls == ["ExitRates"] and not report.ok


# ----------------------------------------------------------------- help content stays in sync
def test_column_guide_covers_every_required_column():
    assert list(COLUMN_GUIDE) == [*config.FEATURES, config.TARGET]


def test_every_metric_has_help_text():
    from src.evaluation import METRIC_NAMES
    assert set(METRIC_HELP) == set(METRIC_NAMES)


# ----------------------------------------------------------------- HTML components
def test_user_text_is_escaped():
    markup = ui.hero("<script>x</script>", "tag & line", ["a<b"])
    assert "<script>" not in markup and "&lt;script&gt;" in markup and "a&lt;b" in markup


def test_components_are_single_line():
    # Markdown would turn indented continuation lines into code blocks
    assert "\n" not in ui.metric_cards({"F1": 0.5}) + ui.confusion_matrix_html(np.array([[5, 1], [2, 3]]))


def test_metric_cards_show_percentages():
    markup = ui.metric_cards({"Accuracy": 0.88201, "AUC": float("nan")}, best={"Accuracy": True})
    assert "88.20%" in markup and "n/a" in markup and "osc-best" in markup
    assert "width:88.2%" in markup


@pytest.mark.parametrize("value, expected", [(0.88201, "88.20%"), (1.0, "100.00%"), (-0.05, "-5.00%"),
                                             (float("nan"), "n/a"), (None, "n/a")])
def test_format_percent(value, expected):
    assert ui.format_percent(value) == expected


def test_confusion_matrix_and_sentence_use_the_counts():
    cm = np.array([[1849, 210], [78, 304]])
    markup = ui.confusion_matrix_html(cm)
    for count in ["1,849", "210", "78", "304"]:
        assert count in markup
    assert "Caught <b>304 of 382</b>" in ui.describe_confusion(cm)


def test_best_model_cards_handle_ties_and_undefined_metrics():
    table = pd.DataFrame({"Recall": [0.8, 0.8, 0.5], "AUC": [np.nan] * 3},
                         index=["Logistic Regression", "SVM", "KNN"])
    markup = ui.best_model_cards(table)
    assert "Logistic Regression &amp; SVM" in markup and "not defined for this data" in markup
    assert "80.00%" in markup


# ----------------------------------------------------------------- colour-coded graphics
@pytest.mark.parametrize("value, label", [(0.80, "Good"), (0.7999, "Moderate"), (0.60, "Moderate"),
                                          (0.5999, "Low"), (-0.2, "Low"), (float("nan"), "n/a")])
def test_performance_band_boundaries(value, label):
    assert ui.performance_band(value)[2] == label


def test_metric_cards_are_colour_coded_with_a_text_label():
    markup = ui.metric_cards({"AUC": 0.9255, "Precision": 0.5914})
    assert "osc-metric osc-good" in markup and "▲ Good" in markup
    assert "osc-metric osc-low" in markup and "▼ Low" in markup


@pytest.fixture
def comparison() -> pd.DataFrame:
    return pd.DataFrame({"F1": [0.68, 0.66, 0.62], "AUC": [0.93, np.nan, 0.92]},
                        index=["SVM", "Decision Tree", "KNN"])


def test_position_chart_has_a_hover_tooltip_per_model(comparison):
    markup = ui.position_chart(comparison, "SVM")
    assert "SVM (selected): 68.00%" in markup
    assert "Decision Tree: 66.00%" in markup and "KNN: 62.00%" in markup
    assert markup.count("osc-pos-row") == 2 and "\n" not in markup


def test_ranking_bars_sorted_best_first_with_selection(comparison):
    markup = ui.ranking_bars(comparison, "F1", "KNN")
    assert markup.index("SVM") < markup.index("Decision Tree") < markup.index("KNN")
    assert "osc-rank-row osc-moderate selected" in markup and "🏆" in markup


def test_outcome_mix_shares_add_up():
    markup = ui.outcome_mix(np.array([[1849, 210], [78, 304]]))
    for text in ["1,849</b> (75.7%)", "304</b> (12.5%)", "210</b> (8.6%)", "78</b> (3.2%)", "2,441 PREDICTIONS"]:
        assert text in markup


def test_section_nav_links_to_each_anchor():
    markup = ui.section_nav([("performance", "📊 Model performance"), ("help", "Help & <FAQ>")])
    assert 'href="#performance"' in markup and 'href="#help"' in markup
    assert "&lt;FAQ&gt;" in markup and "\n" not in markup
