"""Train, evaluate and save all five classifiers.

Usage (from the project folder):
    python train_models.py

Produces:
    model/<model>.joblib       fitted pipelines (preprocessing + classifier)
    model/metadata.json        feature schema, tuned parameters, scores, library versions
    model/model_comparison.csv test-set metrics of all models
    test_data.csv              the held-out 20 % test rows (upload these in the app)
"""
import logging

from src import config
from src.data import DataPreparer, load_raw_data
from src.evaluation import ModelEvaluator
from src.persistence import ModelRepository
from src.training import HyperparameterTuner, silence_expected_warnings
from src.workflow import TrainingWorkflow, save_artifacts


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("src").setLevel(logging.INFO)  # progress messages from this project only
    silence_expected_warnings()

    # Compose the workflow from its parts (each part can be replaced independently).
    preparer = DataPreparer()
    workflow = TrainingWorkflow(preparer=preparer, tuner=HyperparameterTuner(), evaluator=ModelEvaluator())

    result = workflow.run(load_raw_data(config.DATA_PATH))
    save_artifacts(result, preparer, ModelRepository(config.MODEL_DIR), config.TEST_DATA_PATH)

    print("\nBest hyper-parameters (5-fold CV on the training set)")
    print(result.tuning_summary().to_string())
    print("\nTest-set metrics")
    print(result.comparison.to_string())


if __name__ == "__main__":
    main()
