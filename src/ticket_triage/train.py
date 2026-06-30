"""Train and evaluate the support-ticket triage classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from .config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN, LABELS, RANDOM_STATE
from .data import load_training_data
from .evaluation import evaluate_predictions, format_metrics
from .leakage import group_labels
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train support-ticket triage model")
    parser.add_argument("--input", default="data/train.csv", help="Training CSV path")
    parser.add_argument("--model-out", default="models/ticket_triage.joblib", help="Model output path")
    parser.add_argument("--metrics-out", default="reports/metrics.json", help="Metrics JSON output path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for out-of-fold sanity-check metrics. Use 0 to skip.",
    )
    return parser.parse_args()


def build_cross_validation_metrics(x, y, folds: int, random_state: int) -> dict:
    """Out-of-fold metrics from a plain stratified split.

    WARNING: the data is templated (~400 rows, ~71 distinct templates), so a
    random split leaks near-duplicate templates across folds and reads a
    leakage-inflated 1.0000. Kept only to make that leakage visible next to the
    honest grouped number below. Not a generalization estimate.
    """
    if folds < 2:
        return {"enabled": False, "reason": "cv_folds less than 2"}

    min_class_count = int(y.value_counts().min())
    if folds > min_class_count:
        return {
            "enabled": False,
            "reason": f"cv_folds={folds} exceeds smallest class count={min_class_count}",
        }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_predictions = cross_val_predict(build_model(), x, y, cv=cv)
    cv_metrics = evaluate_predictions(list(y), list(oof_predictions))
    cv_metrics.update(
        {
            "enabled": True,
            "folds": folds,
            "method": "StratifiedKFold out-of-fold predictions (LEAKY: templates shared across folds)",
        }
    )
    return cv_metrics


def build_grouped_cv_metrics(x, y, folds: int, random_state: int) -> dict:
    """Honest generalization estimate: out-of-fold metrics with no template
    shared between train and validation.

    Rows are grouped by template skeleton and split with StratifiedGroupKFold,
    so the model is always scored on phrasings it has not seen. This is the
    number that predicts hidden-holdout behavior, unlike the leaky split above.
    """
    if folds < 2:
        return {"enabled": False, "reason": "cv_folds less than 2"}

    groups = group_labels(x)
    n_groups = len(set(groups))
    if folds > n_groups:
        return {"enabled": False, "reason": f"cv_folds={folds} exceeds template count={n_groups}"}

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_predictions = cross_val_predict(build_model(), x, y, cv=cv, groups=groups)
    cv_metrics = evaluate_predictions(list(y), list(oof_predictions))
    cv_metrics.update(
        {
            "enabled": True,
            "folds": folds,
            "n_templates": n_groups,
            "method": "StratifiedGroupKFold out-of-fold predictions (HONEST: no template shared across folds)",
        }
    )
    return cv_metrics


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
            "cross_validation": build_cross_validation_metrics(x, y, args.cv_folds, args.random_state),
            "grouped_cross_validation": build_grouped_cv_metrics(x, y, args.cv_folds, args.random_state),
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

    print("Validation split (LEAKY - templates shared, treat as upper bound):")
    print(format_metrics(metrics))
    cv_metrics = metrics.get("cross_validation", {})
    if cv_metrics.get("enabled"):
        print("\nRandom 5-fold CV (LEAKY sanity check):")
        print(format_metrics(cv_metrics))
    grouped = metrics.get("grouped_cross_validation", {})
    if grouped.get("enabled"):
        print(f"\nGrouped 5-fold CV (HONEST, {grouped['n_templates']} templates, predicts hidden holdout):")
        print(format_metrics(grouped))
    print(f"\nSaved model:   {model_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
