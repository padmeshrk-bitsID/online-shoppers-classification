"""Shared pytest setup."""
import sys
from pathlib import Path

# Make the project root importable (`import src...`) when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    # Rare category codes in test rows are mapped to "infrequent" by design (see src/training.py).
    config.addinivalue_line("filterwarnings", "ignore:Found unknown categories:UserWarning")
