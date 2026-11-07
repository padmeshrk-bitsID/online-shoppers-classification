"""Saving and loading of trained models and their metadata.

All file-system knowledge about models lives here, so the training workflow and
the app never build file paths themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config

METADATA_FILE = "metadata.json"
COMPARISON_FILE = "model_comparison.csv"


class ModelRepository:
    """Reads and writes `model/<key>.joblib`, `metadata.json` and the comparison table."""

    def __init__(self, model_dir: Path = config.MODEL_DIR):
        self.model_dir = Path(model_dir)

    # ---------------------------------------------------------- paths
    def model_path(self, key: str) -> Path:
        return self.model_dir / f"{key}.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / METADATA_FILE

    @property
    def comparison_path(self) -> Path:
        return self.model_dir / COMPARISON_FILE

    # ---------------------------------------------------------- write
    def save_model(self, key: str, pipeline: Pipeline) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_path(key)
        joblib.dump(pipeline, path, compress=3)
        return path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2))
        return self.metadata_path

    def save_comparison(self, table: pd.DataFrame) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.comparison_path)
        return self.comparison_path

    # ---------------------------------------------------------- read
    def load_model(self, key: str) -> Pipeline:
        path = self.model_path(key)
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}. Run `python train_models.py` first.")
        return joblib.load(path)

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"{self.metadata_path} not found. Run `python train_models.py` first.")
        return json.loads(self.metadata_path.read_text())

    def load_comparison(self) -> pd.DataFrame:
        return pd.read_csv(self.comparison_path, index_col=0)

    def available_models(self) -> dict[str, str]:
        """Display name -> key, in training order (read from metadata)."""
        return {entry["name"]: entry["key"] for entry in self.load_metadata()["models"]}
