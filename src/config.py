"""Central configuration: the single source of truth for paths, columns and settings.

Every other module imports from here instead of hard-coding values, so a change
(e.g. a new test-split size) is made in exactly one place.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

# ---------------------------------------------------------------- columns
TARGET = "Revenue"  # True / False -> did the session end in a purchase?
CLASS_LABELS = {0: "No purchase", 1: "Purchase"}

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]

# Integer codes that stand for categories (browser #3 is not "more" than browser #2)
CODED_CATEGORICAL_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]

CATEGORICAL_FEATURES = ["Month", *CODED_CATEGORICAL_FEATURES, "VisitorType", "Weekend"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------- experiment settings
RANDOM_STATE = 42          # fixed seed -> reproducible split, CV folds and models
TEST_SIZE = 0.20           # 80 % train / 20 % held-out test
CV_FOLDS = 5               # stratified folds used for hyper-parameter tuning
TUNING_METRIC = "f1"       # F1 of the purchase class: robust to the 84/16 imbalance
RARE_CATEGORY_MIN_COUNT = 30  # categories seen fewer times are grouped as "infrequent"
