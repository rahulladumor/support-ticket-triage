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

Primary metric: **macro F1**, with **fraud recall** tracked separately.

Reasoning:

- Accuracy can look good by favoring the majority class (`general`).
- Macro F1 gives each route equal weight, so minority classes matter.
- `fraud-report` is the highest-stakes route to miss, so I track recall on it explicitly.

### The data is templated, so a naive split leaks

The most important thing I found in this dataset: the ~400 rows reduce to about **70 distinct sentence templates** (roughly 5-6 near-duplicate rows each). Rows differ only by a swapped coin name, device, dollar amount, or a polite prefix/suffix:

```text
"How does staking work and what rewards can I expect on Polygon? Please advise."
"Hey, How does staking work and what rewards can I expect on Ethereum? Thanks."
```

A random train/validation split puts the *same template* on both sides, TF-IDF memorizes it, and **every metric reads a perfect 1.0000**. That number is leakage, not skill. The hidden holdout will use phrasings the model has never seen, so it will behave like a split where no template is shared.

So I report two numbers (both from `train.py`):

| Split | macro F1 | fraud recall | meaning |
|---|---:|---:|---|
| Random 80/20 + random 5-fold CV | 1.0000 | 1.0000 | **leaky** upper bound, ignore |
| Grouped 5-fold CV (no template shared) | **~0.80** | **~0.72** | **honest** generalization estimate |

The honest estimate is stable across seeds (macro F1 0.80 ± 0.01; fraud recall 0.72 ± 0.05, dipping as low as 0.64). I treat the grouped number as my prediction of hidden-holdout performance. The grouping is done by [`leakage.py`](src/ticket_triage/leakage.py), which collapses each message to its template skeleton; `StratifiedGroupKFold` then keeps every template in a single fold.

The operational takeaway is the fraud row: ~0.72 recall means roughly **one in four fraud reports phrased in a new way would be misrouted**, and it is the noisiest metric. That, not the macro F1, is what I would gate on in production (see escalation note below).

### Class imbalance handling

- The model uses `class_weight="balanced"` in `LinearSVC`, raising the cost of mistakes on minority classes, especially `fraud-report`.
- I would know imbalance is hurting if the gap between weighted F1 and macro F1 widens, or if the grouped fraud recall drops below the operational threshold while overall accuracy stays high.

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
- Holdout CSV scoring smoke test that writes a `predicted_label` column.
- Template-skeleton grouping: coin/prefix/suffix variants collapse to one group, distinct intents do not (guards the leakage-aware eval).

## Scope and trade-offs

Prioritized:

- Clean classical baseline.
- A leakage-aware evaluation: catching that the data is templated and reporting the honest grouped-CV number instead of the leaky 1.0000. This is the part I would defend hardest.
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

- Whether to add character n-grams (for typo and novel-phrasing robustness). I could only evaluate this honestly *after* fixing the leakage, because against the leaky 1.0000 every variant looks identical. Under grouped CV, adding `char_wb(3,5)` lifts macro F1 slightly (~0.80 -> ~0.83) but does **not** help fraud recall and adds variance to it. Since fraud recall is the metric I most care about, I kept the simpler word-only model. The trade-off is a little less robustness to misspellings for better fraud-recall stability and lower complexity. This is the clearest example in the project of the honest eval changing a decision.

What I would do with more time:

- Add repeated stratified cross-validation with confidence intervals.
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

1. Macro F1, because the classes are imbalanced and every route matters equally; weighted F1 or accuracy would hide the minority routes. Fraud is the most expensive route to miss, so I track its recall separately. The bigger point: the metric is only meaningful on a leakage-free split. The data is templated, so a random split scores a fake 1.0000; my real number comes from grouped CV (~0.80 macro F1, ~0.72 fraud recall).
2. `class_weight="balanced"`. I would know imbalance is hurting if the weighted-vs-macro F1 gap widens, or if grouped fraud recall falls below threshold while accuracy stays high. The honest fraud recall is ~0.72 and noisy (down to 0.64), which is exactly the signal to act on.
3. Whether to add character n-grams. I could only judge this after fixing the leakage; under grouped CV they lift macro F1 a little but do not help fraud recall, so I kept the simpler word model. Best example of the honest eval changing a decision.
4. At 10,000 requests/minute I would keep the classical model: low latency, cheap, no external dependency, deterministic. An LLM earns its place on long/messy/multilingual tickets or context-dependent reasoning. In production I would run a hybrid: classical for fast routing, confidence-based fallback to an LLM or human for ambiguous and high-risk fraud-like cases.
