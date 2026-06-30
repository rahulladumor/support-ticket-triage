"""Evaluation helpers."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from .config import HIGH_STAKES_LABEL, LABELS


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Return metrics chosen for the take-home exercise.

    Macro F1 is the primary metric because each route matters and the dataset is
    imbalanced. Fraud recall is tracked separately because fraud-report is the
    highest-stakes class to miss.
    """
    return {
        "primary_metric": "macro_f1",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fraud_recall": recall_score(
            y_true,
            y_pred,
            labels=[HIGH_STAKES_LABEL],
            average=None,
            zero_division=0,
        )[0],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "label_order": LABELS,
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Create a short human-readable metrics summary."""
    return "\n".join(
        [
            f"Primary metric: {metrics['primary_metric']}",
            f"Accuracy:       {metrics['accuracy']:.4f}",
            f"Macro F1:       {metrics['macro_f1']:.4f}",
            f"Weighted F1:    {metrics['weighted_f1']:.4f}",
            f"Fraud recall:   {metrics['fraud_recall']:.4f}",
        ]
    )
