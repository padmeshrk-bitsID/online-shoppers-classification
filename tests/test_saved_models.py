"""End-to-end checks on the saved artifacts in model/ (skipped until training has run)."""
import pandas as pd
import pytest

from src import config
from src.data import DataPreparer
from src.evaluation import ModelEvaluator
from src.inference import PREDICTED_COLUMN, SCORE_COLUMN, PredictionService
from src.persistence import ModelRepository

pytestmark = pytest.mark.skipif(
    not (config.MODEL_DIR / "metadata.json").exists(), reason="run `python train_models.py` first")


@pytest.fixture(scope="module")
def service() -> PredictionService:
    return PredictionService(ModelRepository(config.MODEL_DIR), DataPreparer(), ModelEvaluator())


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return pd.read_csv(config.TEST_DATA_PATH)


def test_all_five_model_files_exist(service):
    for name in service.model_names():
        assert ModelRepository().model_path(service.model_info(name)["key"]).exists()
    assert len(service.model_names()) == 5


def test_reloaded_models_reproduce_saved_test_metrics(service, test_data):
    recomputed = service.compare_all(test_data)
    saved = ModelRepository().load_comparison()
    pd.testing.assert_frame_equal(recomputed, saved, check_exact=False, atol=1e-4, check_names=False)


def test_prediction_without_target_column(service, test_data):
    output = service.run("Decision Tree", test_data.drop(columns=config.TARGET).head(25))
    assert output.evaluation is None
    assert len(output.predictions) == 25
    assert {PREDICTED_COLUMN, SCORE_COLUMN} <= set(output.predictions.columns)


def test_recommended_model_is_best_by_mcc(service):
    saved = ModelRepository().load_comparison()
    assert service.recommended_model() == saved["MCC"].idxmax()


def test_report_table_matches_sklearn_layout(service, test_data):
    report = service.run("SVM", test_data).evaluation.report
    assert list(report.index) == ["No purchase", "Purchase", "accuracy", "macro avg", "weighted avg"]
    assert report.loc["accuracy", "support"] == len(test_data)
    assert report["support"].dtype.kind == "i"
