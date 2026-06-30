#!/usr/bin/env bash
set -euo pipefail
INPUT_CSV="${1:?Usage: scripts/predict_holdout.sh path/to/holdout.csv [output.csv]}"
OUTPUT_CSV="${2:-predictions.csv}"
python -m ticket_triage.predict_csv --input "$INPUT_CSV" --output "$OUTPUT_CSV" --model models/ticket_triage.joblib
