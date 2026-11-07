"""Streamlit app: evaluate the five pre-trained classifiers on uploaded test data.

Run locally:
    streamlit run app.py

Where each required feature lives in the UI
    a. Dataset upload (CSV, test data only)  -> sidebar, step 1
    b. Model selection drop-down             -> sidebar, step 2
    c. Display of evaluation metrics         -> "Model performance" tab (six animated cards)
    d. Confusion matrix + classification report -> "Model performance" tab
    +  Results of all five models on the test data -> "Compare all models" tab

The app NEVER retrains: it loads the saved pipelines from `model/` and only predicts.
This file is presentation only; every calculation is delegated to `src/`.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.data import DataPreparer, SchemaError, SchemaReport
from src.evaluation import METRIC_NAMES, ModelEvaluator
from src.inference import (
    ACTUAL_LABEL_COLUMN,
    CORRECT_COLUMN,
    PREDICTED_COLUMN,
    PREDICTED_LABEL_COLUMN,
    SCORE_COLUMN,
    PredictionOutput,
    PredictionService,
)
from src.persistence import ModelRepository
from src.training import silence_expected_warnings
from src.ui import components as ui
from src.ui import content
from src.ui.theme import page_css
from src.visualization import plot_metric_heatmap

# ------------------------------------------------------------------ constants
DATA_SOURCE_KEY = "data_source"
SOURCE_UPLOAD = "📤 Upload my test CSV"
SOURCE_SAMPLE = "📦 Use bundled test_data.csv"
PREVIEW_ROWS = 500
PERCENT_FORMAT = "%.2f%%"   # tables show 0.882 as 88.20%


def percent_column(label: str, help_text: str | None = None):
    """Table column showing a 0-100 value as a percentage with a progress bar."""
    return st.column_config.ProgressColumn(label, help=help_text, format=PERCENT_FORMAT, min_value=0.0, max_value=100.0)
UCI_URL = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"

st.set_page_config(page_title=content.APP_TITLE, page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
silence_expected_warnings()
st.markdown(page_css(), unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render one of our own HTML components."""
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------ cached work
@st.cache_resource(show_spinner="Loading the saved models...")
def get_service() -> PredictionService:
    """One prediction service per server process; models are loaded lazily and kept in memory."""
    return PredictionService(ModelRepository(config.MODEL_DIR), DataPreparer(), ModelEvaluator())


@st.cache_data(show_spinner=False)
def read_csv(content_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content_bytes))


@st.cache_data(show_spinner=False)
def sample_csv_bytes() -> bytes:
    return config.TEST_DATA_PATH.read_bytes()


# A leading underscore tells Streamlit not to hash that argument (the service is shared).
@st.cache_data(show_spinner=False)
def evaluate_model(_service: PredictionService, model_name: str, raw: pd.DataFrame) -> PredictionOutput:
    return _service.run(model_name, raw)


@st.cache_data(show_spinner=False)
def compare_models(_service: PredictionService, raw: pd.DataFrame) -> pd.DataFrame:
    return _service.compare_all(raw)


def use_sample_data() -> None:
    """Button callback: switch the data-source selector to the bundled test file."""
    st.session_state[DATA_SOURCE_KEY] = SOURCE_SAMPLE


def notify_once(event: tuple, message: str) -> None:
    """Show a toast only when something actually changed (not on every re-run)."""
    if st.session_state.get("last_event") != event:
        st.session_state["last_event"] = event
        st.toast(message, icon="✅")


# ------------------------------------------------------------------ sidebar: inputs
def sidebar_data_step(service: PredictionService) -> tuple[pd.DataFrame | None, str, str | None]:
    """Step 1. Returns (table or None, label for the UI, error message or None)."""
    st.sidebar.markdown("### 1 · Test data")
    source = st.sidebar.radio(
        "Data source", [SOURCE_UPLOAD, SOURCE_SAMPLE], key=DATA_SOURCE_KEY,
        help="Upload your own test split, or use the held-out test split saved during training.",
    )
    if source == SOURCE_SAMPLE:
        rows = service.metadata["split"]["test_rows"]
        st.sidebar.caption(f"Held-out 20 % test split saved during training ({rows:,} sessions).")
        return read_csv(sample_csv_bytes()), "bundled test_data.csv", None

    uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"], help=content.UPLOAD_HELP)
    st.sidebar.download_button(
        "⬇️ Download sample test_data.csv", data=sample_csv_bytes(), file_name="test_data.csv",
        mime="text/csv", help="A correctly formatted test file: upload it as-is or use it as a template.",
    )
    if uploaded is None:
        st.sidebar.caption("📎 Drag & drop a **.csv** with the 17 features + `Revenue`.")
        return None, "", None
    try:
        return read_csv(uploaded.getvalue()), uploaded.name, None
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return None, uploaded.name, "The file could not be read as a CSV. Please export it as comma-separated UTF-8."


def sidebar_model_step(service: PredictionService) -> str:
    """Step 2: model drop-down plus a card describing the selected model."""
    st.sidebar.markdown("### 2 · Model")
    names = service.model_names()
    recommended = service.recommended_model()
    selected = st.sidebar.selectbox(
        "Classifier", names, index=names.index(recommended),
        format_func=lambda name: f"{name}  ⭐" if name == recommended else name,
        placeholder="Choose a classifier...",
        help=f"{content.MODEL_HELP} ⭐ = best MCC on the held-out test set.",
    )
    info = service.model_info(selected)
    html(ui.model_card(selected, content.MODEL_BLURBS.get(info["key"], ""), info["best_params"],
                       info["cv_scores"], info["file"], recommended=selected == recommended))
    return selected


def sidebar_help() -> None:
    st.sidebar.markdown("---")
    with st.sidebar.expander("❓ Need help?"):
        st.markdown(
            "- Your CSV needs the **17 input features + `Revenue`**. Order doesn't matter.\n"
            "- `Revenue` / `Weekend` accept **TRUE/FALSE, 1/0 or yes/no**.\n"
            "- Models are **loaded, never retrained**.\n"
            "- Full column guide, glossary and FAQ: open the **📘 Help** tab."
        )


# ------------------------------------------------------------------ main panel: states
def column_guide_table() -> pd.DataFrame:
    return pd.DataFrame(
        [(column, kind, example) for column, (kind, example) in content.COLUMN_GUIDE.items()],
        columns=["Column", "Type", "Accepted values / example"],
    )


def show_empty_state() -> None:
    """What the user sees before any data is chosen: guidance instead of a blank page."""
    html(ui.empty_state("📤", "Upload your test CSV to begin",
                        "Use step 1 in the sidebar to upload a CSV with the 17 input features and the Revenue "
                        "column, or try the app instantly with the bundled test data."))
    _, middle, _ = st.columns([1, 1, 1])
    middle.button("▶  Try it with the bundled test data", on_click=use_sample_data, type="primary")
    html(ui.step_cards(content.HOW_TO_STEPS))
    with st.expander("📄 Required CSV format (18 columns)"):
        st.dataframe(column_guide_table(), hide_index=True)


def show_input_problems(label: str, problems: list[str]) -> None:
    """Explain exactly what is wrong with the file and how to fix it."""
    st.error(f"**{label}** can't be evaluated yet. Please fix the following:", icon="🚫")
    for problem in problems:
        st.markdown(f"- {problem}")
    st.info("Tip: download **sample test_data.csv** from the sidebar to see a correctly formatted file.", icon="💡")
    with st.expander("📄 Required CSV format (18 columns)", expanded=True):
        st.dataframe(column_guide_table(), hide_index=True)


def evaluate_with_status(service: PredictionService, model_name: str,
                         raw: pd.DataFrame, report: SchemaReport) -> PredictionOutput:
    """Animated status box that lists each step; it collapses into an audit trail when done."""
    model_file = service.model_info(model_name)["file"]
    with st.status(f"Evaluating {model_name} on {report.n_rows:,} rows...", expanded=False) as status:
        st.write(f"✅ File checked: {report.n_rows:,} rows, all 17 features and `{config.TARGET}` present")
        if report.extra_columns:
            st.write(f"ℹ️ Ignored extra column(s): {', '.join(report.extra_columns)}")
        st.write(f"✅ Loaded pre-trained model `model/{model_file}` (no retraining)")
        output = evaluate_model(service, model_name, raw)
        st.write("✅ Predictions and the six evaluation metrics computed")
        status.update(label=f"{model_name} evaluated on {report.n_rows:,} test rows: no retraining",
                      state="complete", expanded=False)
    return output


def show_data_summary(label: str, output: PredictionOutput) -> None:
    (tn, fp), (fn, tp) = output.evaluation.confusion
    rows, buyers = tn + fp + fn + tp, fn + tp
    html(ui.badge(f"📄 {label}", "blue") + ui.badge(f"{rows:,} sessions", "green")
         + ui.badge(f"{buyers:,} actual buyers ({buyers / rows:.1%})", "orange"))


# ------------------------------------------------------------------ main panel: tabs
def show_metric_glossary() -> None:
    st.dataframe(
        pd.DataFrame([(name, meaning, formula) for name, (meaning, formula) in content.METRIC_HELP.items()],
                     columns=["Metric", "What it tells you", "Formula"]),
        hide_index=True,
    )


def show_performance(model_name: str, output: PredictionOutput, comparison: pd.DataFrame) -> None:
    """Requirements (c) and (d): metric cards, confusion matrix and classification report."""
    evaluation = output.evaluation
    title, glossary = st.columns([5, 1], vertical_alignment="center")
    title.markdown(f"#### {model_name}: evaluation metrics")
    with glossary.popover("ℹ️ Metric guide"):
        show_metric_glossary()

    is_best = {metric: comparison.loc[model_name, metric] == comparison[metric].max() for metric in METRIC_NAMES}
    html(ui.metric_cards(evaluation.metrics, is_best))
    html(ui.band_legend())
    st.caption("All metrics are shown as percentages (higher is better). Hover a card to see what it means; "
               "**best** = highest of the five models on this data.")
    html(ui.callout(ui.describe_confusion(evaluation.confusion)))

    st.markdown(f"##### Where {model_name} stands among the five models")
    html(ui.position_chart(comparison, model_name))
    st.caption("Big coloured dot = selected model (colour = band) · grey dots = the other four models · "
               "grey bar = their range. Hover any dot for the model and its score.")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("##### Confusion matrix")
        html(ui.confusion_matrix_html(evaluation.confusion))
        st.caption("Blue = correct predictions, orange = mistakes. Percentages are shares of each actual class.")
    with right:
        st.markdown("##### Classification report")
        report = evaluation.report.reset_index(names="Class / average")
        report[["precision", "recall", "f1-score"]] *= 100      # show as percentages
        st.dataframe(
            report, hide_index=True,
            column_config={
                "precision": percent_column("Precision"),
                "recall": percent_column("Recall"),
                "f1-score": percent_column("F1-score"),
                "support": st.column_config.NumberColumn("Support", help="Number of rows in that class", format="%d"),
            },
        )
        st.caption("Per-class view: how well each class (buyers and non-buyers) is predicted.")


def show_comparison(table: pd.DataFrame, selected: str) -> None:
    """Results of all five saved models on the same test data."""
    st.markdown("#### Results of all five models on this test data")
    st.caption("Every saved model is scored on the same file. Nothing is retrained.")
    html(ui.best_model_cards(table))

    mcc = table["MCC"].dropna()
    if not mcc.empty:
        html(ui.callout(f"🏆 <b>Overall winner on this data: {mcc.idxmax()}</b> (highest MCC, the fairest single "
                        "metric when buyers are only ~15 % of sessions)."))

    st.markdown("##### Rank the models by a metric")
    metric = st.segmented_control("Metric to rank by", METRIC_NAMES, default="MCC", key="rank_metric",
                                  help="Pick a metric to re-order the models. Hover a bar for details.") or "MCC"
    html(ui.ranking_bars(table, metric, selected))
    html(ui.band_legend())

    st.markdown("##### Full comparison table")
    display = (table * 100).reset_index(names="Model")           # show as percentages
    display["Model"] = [f"{name}  ← selected" if name == selected else name for name in display["Model"]]
    st.dataframe(
        display, hide_index=True,
        column_config={metric: percent_column(metric, content.METRIC_HELP[metric][0]) for metric in METRIC_NAMES},
    )
    st.caption("All values are percentages. MCC runs from -100% to +100% (0% = no better than guessing).")
    with st.expander("🗺️ Heatmap view"):
        figure = plot_metric_heatmap(table, title="Metrics on this data (bold = best in column)", as_percent=True)
        st.pyplot(figure)
        plt.close(figure)  # free memory: Streamlit re-runs the script on every interaction
    st.download_button("⬇️ Download comparison (CSV)", table.to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")


PREDICTION_FILTERS = {
    "All rows": lambda d: d,
    "✅ Correct": lambda d: d[d[CORRECT_COLUMN]],
    "❌ Misclassified": lambda d: d[~d[CORRECT_COLUMN]],
    "Missed buyers": lambda d: d[(d[PREDICTED_COLUMN] == 0) & ~d[CORRECT_COLUMN]],
    "False alarms": lambda d: d[(d[PREDICTED_COLUMN] == 1) & ~d[CORRECT_COLUMN]],
}


def show_predictions(output: PredictionOutput) -> None:
    """Row-level results with quick filters and a CSV download."""
    predictions = output.predictions
    html(ui.outcome_mix(output.evaluation.confusion))
    choice = st.radio("Show", list(PREDICTION_FILTERS), horizontal=True,
                      help="Missed buyers = false negatives; False alarms = false positives.")
    view = PREDICTION_FILTERS[choice](predictions)

    leading = [ACTUAL_LABEL_COLUMN, PREDICTED_LABEL_COLUMN, CORRECT_COLUMN, SCORE_COLUMN]
    columns = leading + [c for c in view.columns if c not in leading and c != PREDICTED_COLUMN]
    st.dataframe(
        view[columns].head(PREVIEW_ROWS), hide_index=True,
        column_config={
            CORRECT_COLUMN: st.column_config.CheckboxColumn("Correct?"),
            SCORE_COLUMN: st.column_config.NumberColumn(
                "Purchase score", format="%.4f",
                help="Predicted purchase probability (SVM: decision value). Higher = more likely to buy."),
        },
    )
    shown = min(len(view), PREVIEW_ROWS)
    st.caption(f"Showing {shown:,} of {len(view):,} matching rows ({len(predictions):,} in total).")
    st.download_button("⬇️ Download all predictions (CSV)", predictions.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")


def show_help() -> None:
    st.markdown("#### How to use this app")
    html(ui.step_cards(content.HOW_TO_STEPS))
    st.markdown("#### Evaluation metrics")
    show_metric_glossary()
    st.markdown("#### Required CSV format")
    st.dataframe(column_guide_table(), hide_index=True)
    st.markdown("#### FAQ")
    for question, answer in content.FAQ:
        with st.expander(question):
            st.markdown(answer)
    st.caption(f"Dataset: [UCI Online Shoppers Purchasing Intention (id 468)]({UCI_URL}) · CC BY 4.0")


# ------------------------------------------------------------------ entry point
def main() -> None:
    html(ui.hero(content.APP_TITLE, content.APP_TAGLINE, content.HERO_CHIPS))
    try:
        service = get_service()
    except FileNotFoundError as error:
        st.error(f"{error}", icon="🚫")
        st.stop()

    raw, label, read_error = sidebar_data_step(service)
    model_name = sidebar_model_step(service)
    sidebar_help()

    if read_error:
        show_input_problems(label, [read_error])
        st.stop()
    if raw is None:
        show_empty_state()
        st.stop()

    report = service.preparer.inspect(raw, require_target=True)
    if not report.ok:
        show_input_problems(label, report.problems)
        st.stop()

    try:
        output = evaluate_with_status(service, model_name, raw, report)
        comparison = compare_models(service, raw)
    except SchemaError as error:
        show_input_problems(label, [str(error)])
        st.stop()

    notify_once((label, report.n_rows, model_name), f"{model_name} evaluated on {report.n_rows:,} rows")
    show_data_summary(label, output)

    performance_tab, comparison_tab, predictions_tab, help_tab = st.tabs(
        ["📊 Model performance", "🏆 Compare all models", "🔍 Predictions", "📘 Help"])
    with performance_tab:
        show_performance(model_name, output, comparison)
    with comparison_tab:
        show_comparison(comparison, model_name)
    with predictions_tab:
        show_predictions(output)
    with help_tab:
        show_help()


if __name__ == "__main__":
    main()
