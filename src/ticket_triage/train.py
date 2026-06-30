"""Train and evaluate the support-ticket triage classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN, LABELS, RANDOM_STATE
from .data import load_training_data
from .evaluation import evaluate_predictions, format_metrics
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train support-ticket triage model")
    parser.add_argument("--input", default="data/train.csv", help="Training CSV path")
    parser.add_argument("--model-out", default="models/ticket_triage.joblib", help="Model output path")
    parser.add_argument("--metrics-out", default="reports/metrics.json", help="Metrics JSON output path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_training_data(args.input)

    x = df[DEFAULT_TEXT_COLUMN]
    y = df[DEFAULT_LABEL_COLUMN]
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    validation_model = build_model()
    validation_model.fit(x_train, y_train)
    val_predictions = validation_model.predict(x_val)

    metrics = evaluate_predictions(list(y_val), list(val_predictions))
    metrics.update(
        {
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "validation_split": args.test_size,
            "random_state": args.random_state,
            "class_counts_total": y.value_counts().to_dict(),
            "model_notes": {
                "algorithm": "TF-IDF word unigrams/bigrams + LinearSVC",
                "imbalance_strategy": "LinearSVC class_weight='balanced'",
                "final_training": "After validation, the saved model is refit on all labeled rows for hidden holdout scoring.",
            },
        }
    )

    model = build_model()
    model.fit(x, y)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS}, model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(format_metrics(metrics))
    print(f"Saved model:   {model_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
