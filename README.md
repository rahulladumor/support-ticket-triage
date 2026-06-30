# Support-ticket triage classifier

Classical ML solution for routing crypto/fintech support tickets into four queues:

- `account-access`
- `transaction-dispute`
- `fraud-report`
- `general`

The goal is a small, defensible baseline that can be explained and shipped inside a short take-home window.

## Approach

I used a classical text classifier: **TF-IDF word unigrams/bigrams + LinearSVC**.

Why this choice:

- The dataset is small: 400 labeled messages.
- The labels are intent-style support routes, where lexical signals are strong.
- Classical TF-IDF models are fast, cheap, deterministic, easy to test, and easy to deploy.
- An LLM would add latency, cost, external dependency risk, and evaluation complexity without being necessary for this baseline.

## Data

Training data expected schema:

```csv
text,label
"I cannot login...",account-access
```

Class distribution in the provided training file:

| Label | Rows |
|---|---:|
| general | 160 |
| account-access | 100 |
| transaction-dispute | 90 |
| fraud-report | 50 |

## Evaluation

I use a stratified 80/20 train/validation split.

Primary metric: **macro F1**.

Reasoning:

- Accuracy can look good by favoring the majority class (`general`).
- Macro F1 gives each route equal weight, so minority classes matter.
- `fraud-report` is the highest-stakes route to miss, so I also track **fraud recall** explicitly.

Class imbalance handling:

- The model uses `class_weight="balanced"` in `LinearSVC`.
- This increases the cost of mistakes on minority classes, especially `fraud-report`.
- I would know imbalance is hurting if validation macro F1 drops while weighted F1 stays high, or if fraud recall falls below the operational threshold.

On the included stratified validation split, the current run produced:

```text
Accuracy:       1.0000
Macro F1:       1.0000
Weighted F1:    1.0000
Fraud recall:   1.0000
```

I do not treat perfect validation as proof the model is production-ready. The dataset is small and the language patterns are clean, so hidden holdout performance is the real test.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

```bash
python -m ticket_triage.train \
  --input data/train.csv \
  --model-out models/ticket_triage.joblib \
  --metrics-out reports/metrics.json
```

Or:

```bash
scripts/train.sh
```

The training script first evaluates on a stratified validation split, then refits the saved model on all labeled rows for hidden holdout scoring.

## Predict one message in Python

```python
from ticket_triage import load_service

service = load_service("models/ticket_triage.joblib")
label = service.predict("Someone withdrew BTC from my account without permission")
print(label)  # fraud-report
```

## Score a holdout CSV

Input CSV must contain a `text` column by default.

```bash
python -m ticket_triage.predict_csv \
  --input holdout.csv \
  --output predictions.csv \
  --model models/ticket_triage.joblib
```

Or:

```bash
scripts/predict_holdout.sh holdout.csv predictions.csv
```

Output CSV adds:

```csv
predicted_label
```

## Tests

```bash
pytest
```

Current tests cover:

- Input validation for empty/non-string messages.
- End-to-end `predict(text) -> label` behavior on a small fitted classifier.

## Scope and trade-offs

Prioritized:

- Clean classical baseline.
- Stratified validation split.
- Macro F1 plus fraud recall.
- Imbalance handling through class weighting.
- Simple `predict(text) -> label` interface.
- Holdout CSV scoring script.
- Basic tests and input validation.

Deliberately left out:

- LLM classifier comparison: useful later, but not needed for this dataset size and deadline.
- FastAPI service: the brief says API is optional; CLI and Python interface are enough for scoring.
- Docker/CI: good production hygiene, but lower priority than ML reasoning and testable core logic for a 3-hour practical.
- Heavy hyperparameter search: high risk of overfitting on only 400 examples.

What I was unsure about:

- I tested character n-grams as an option for typo tolerance. They did not improve validation performance, so I kept the simpler word n-gram model. The trade-off is slightly less robustness to misspellings, but better readability and lower complexity.

What I would do with more time:

- Add repeated stratified cross-validation and confidence intervals.
- Build a small error-analysis notebook or markdown report.
- Add thresholding/escalation logic for low-confidence or high-risk fraud-like messages.
- Evaluate adversarial/ambiguous examples, especially fraud vs transaction dispute.
- Add monitoring for class drift, fraud recall, and manual override rates.
- Package a small FastAPI service only if online serving is required.

Time spent: about 2.5-3 focused hours for a clean baseline, tests, evaluation, and packaging.

## If this scaled to 10,000 requests/minute

This classical model is the right default if traffic is high and ticket text is short:

- Runs locally with low latency.
- No external model dependency.
- Cheap to autoscale.
- Easy to batch.
- Deterministic behavior helps debugging.

An LLM becomes the right call when:

- Tickets become long, messy, multilingual, or require reasoning over account context.
- The route taxonomy changes frequently and labeled data is limited.
- We need explanations, extraction, or policy-aware escalation in the same flow.

For production, I would likely use a hybrid design: classical classifier for fast routing, confidence-based fallback to an LLM or human review for ambiguous/high-risk cases.

## Video walkthrough notes

1. I chose macro F1 because the classes are imbalanced and every route matters. Fraud is high-stakes, so I separately track fraud recall.
2. I used `class_weight="balanced"`. I would watch for imbalance issues by comparing macro F1 vs weighted F1 and by checking the fraud-report row in the classification report.
3. The decision I was unsure about was adding character n-grams. I left them out because they did not improve validation and made the model less simple.
4. At 10,000 requests/minute, I would prefer the classical model unless messages require deeper reasoning. If swapped for an LLM, I would add caching, rate limits, fallbacks, cost controls, and human review for risky cases.
