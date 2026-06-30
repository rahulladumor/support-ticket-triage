# Support-ticket triage classifier

Classical ML solution for routing crypto/fintech support tickets into four queues:

- `account-access`
- `transaction-dispute`
- `fraud-report`
- `general`

Aim was a small baseline I can defend line by line, sized for a short take-home and not a platform build.

## Approach

I went classical: TF-IDF word unigrams and bigrams into a LinearSVC.

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

Primary metric is macro F1, and I track fraud recall on its own.

Reasoning:

- Accuracy can look good just by favoring the majority class (`general`).
- Macro F1 weights each route equally, so the small classes still count.
- `fraud-report` is the worst route to miss, so I watch its recall directly.

### The data is templated, which makes a naive split lie to you

First thing I did was read the actual rows, and they're templated. The ~400 messages come from about 70 underlying sentence templates, 5-6 near-duplicates each, swapping a coin name, a device, a dollar amount, or a polite opener:

```text
"How does staking work and what rewards can I expect on Polygon? Please advise."
"Hey, How does staking work and what rewards can I expect on Ethereum? Thanks."
```

A random split drops copies of the same template on both sides. TF-IDF memorizes them and every metric comes back a clean 1.0000. That is leakage, not skill. The hidden holdout will be worded in ways the model never saw, so it will behave like a split where no template is shared, not like the random one.

So I report two numbers, both printed by `train.py`:

| Split | macro F1 | fraud recall | what it means |
|---|---:|---:|---|
| Random 80/20 + random 5-fold CV | 1.0000 | 1.0000 | leaky, treat as an upper bound |
| Grouped 5-fold CV (no shared template) | ~0.80 | ~0.72 | honest, what I expect on the holdout |

The grouped number holds up across seeds: macro F1 0.80 ± 0.01, fraud recall 0.72 ± 0.05 (it slips to 0.64 on a bad split). The grouping lives in [`leakage.py`](src/ticket_triage/leakage.py), which strips each message to its template skeleton so `StratifiedGroupKFold` can keep every skeleton inside one fold.

The fraud row is the one I actually care about. A recall of 0.72 means roughly one in four fraud reports worded in a new way land in the wrong queue, and it is the noisiest of the four metrics. In production that is the number I would put a threshold on and escalate against, not macro F1.

### Class imbalance handling

- I use `class_weight="balanced"` in `LinearSVC`, which raises the cost of getting a minority class wrong, fraud-report most of all.
- I would know imbalance was hurting if the gap between weighted F1 and macro F1 widened, or if grouped fraud recall dropped under threshold while accuracy stayed high.

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

- Whether to add character n-grams for typo and novel-phrasing tolerance. I could only judge this after fixing the leakage, since against a leaky 1.0000 every variant looks the same. Under grouped CV, adding `char_wb(3,5)` nudges macro F1 up (~0.80 to ~0.83) but doesn't help fraud recall and makes it bounce around more. Fraud recall is the number I care about most, so I stayed with the simpler word-only model: I give up a little tolerance for typos to get steadier fraud recall and fewer moving parts. It is the one place where fixing the eval actually flipped a decision I would otherwise have made the other way.

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
