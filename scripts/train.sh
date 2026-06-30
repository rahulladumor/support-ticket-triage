#!/usr/bin/env bash
set -euo pipefail
python -m ticket_triage.train --input data/train.csv --model-out models/ticket_triage.joblib --metrics-out reports/metrics.json
