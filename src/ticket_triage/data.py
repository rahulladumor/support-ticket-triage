"""Data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN, LABELS
from .validation import validate_text


def load_training_data(
    csv_path: str | Path,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Load and validate labeled training data.

    Expected schema:
    - text: customer support message
    - label: one of the four route labels
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"training CSV not found: {path}")

    df = pd.read_csv(path)
    required_columns = {text_column, label_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df[[text_column, label_column]].copy()
    df[text_column] = df[text_column].map(validate_text)
    df[label_column] = df[label_column].astype(str).str.strip()

    unknown_labels = sorted(set(df[label_column]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"unknown labels found: {unknown_labels}")

    return df


def load_unlabeled_messages(csv_path: str | Path, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """Load and validate a CSV containing inbound messages to score."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"input CSV not found: {path}")

    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"missing required text column: {text_column}")

    df = df.copy()
    df[text_column] = df[text_column].map(validate_text)
    return df
