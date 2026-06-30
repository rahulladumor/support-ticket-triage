"""Batch scoring entry point for hidden holdout CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TEXT_COLUMN
from .data import load_unlabeled_messages
from .predictor import load_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score support-ticket messages from CSV")
    parser.add_argument("--input", required=True, help="Input CSV containing a text column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--model", default="models/ticket_triage.joblib", help="Trained model path")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Input text column name")
    parser.add_argument("--prediction-column", default="predicted_label", help="Output prediction column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_unlabeled_messages(args.input, text_column=args.text_column)
    service = load_service(args.model)
    df[args.prediction_column] = service.predict_many(df[args.text_column].tolist())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote predictions: {output}")


if __name__ == "__main__":
    main()
