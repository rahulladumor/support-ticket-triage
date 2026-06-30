"""Simple predict(text) interface for the trained classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .config import LABELS
from .validation import validate_text, validate_texts


@dataclass(frozen=True)
class TicketTriageService:
    """Small wrapper around a trained sklearn pipeline."""

    model: Any
    labels: list[str]

    def predict(self, text: str) -> str:
        """Predict one route label for a customer support message."""
        normalized = validate_text(text)
        prediction = str(self.model.predict([normalized])[0])
        if prediction not in self.labels:
            raise RuntimeError(f"model returned unexpected label: {prediction}")
        return prediction

    def predict_many(self, texts: list[str]) -> list[str]:
        """Predict route labels for a batch of messages."""
        normalized = validate_texts(texts)
        predictions = [str(label) for label in self.model.predict(normalized)]
        unexpected = sorted(set(predictions) - set(self.labels))
        if unexpected:
            raise RuntimeError(f"model returned unexpected labels: {unexpected}")
        return predictions


def load_service(model_path: str | Path) -> TicketTriageService:
    """Load a trained model artifact from disk."""
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact["model"]
        labels = artifact.get("labels", LABELS)
    else:
        model = artifact
        labels = LABELS
    return TicketTriageService(model=model, labels=list(labels))
