"""Tests for loading, cleaning and validating input tables."""
import pandas as pd
import pytest

from src import config
from src.data import (
    DataPreparer,
    SchemaError,
    load_raw_data,
    remove_duplicates,
    split_train_test,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw_data(config.DATA_PATH)


def test_raw_dataset_matches_uci_description(raw):
    assert raw.shape == (12330, 18)
    assert raw[config.TARGET].sum() == 1908          # purchases reported by UCI


def test_split_is_stratified_and_disjoint(raw):
    train, test = split_train_test(remove_duplicates(raw))
    assert len(test) == pytest.approx(0.2 * (len(train) + len(test)), abs=1)
    assert set(train.index).isdisjoint(test.index)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.005


def test_prepare_casts_types(raw):
    X, y = DataPreparer().prepare(raw.head(50))
    assert list(X.columns) == config.FEATURES
    assert X["Browser"].map(type).eq(str).all()      # codes become categories
    assert set(X["Weekend"].unique()) <= {0, 1}
    assert set(y.unique()) <= {0, 1}


@pytest.mark.parametrize("weekend", [[True, False], ["TRUE", "FALSE"], [1, 0], ["yes", "no"]])
def test_boolean_formats_are_accepted(raw, weekend):
    sample = raw.head(2).copy()
    sample["Weekend"] = weekend
    X = DataPreparer().prepare_features(sample)
    assert X["Weekend"].tolist() == [1, 0]


def test_target_is_optional(raw):
    X, y = DataPreparer().prepare(raw.head(5).drop(columns=config.TARGET))
    assert y is None and len(X) == 5


def test_missing_column_is_reported(raw):
    with pytest.raises(SchemaError, match="PageValues"):
        DataPreparer().prepare(raw.drop(columns="PageValues"))


def test_non_numeric_value_is_reported(raw):
    broken = raw.head(3).copy()
    broken["ExitRates"] = ["a", "b", "c"]
    with pytest.raises(SchemaError, match="ExitRates"):
        DataPreparer().prepare(broken)
