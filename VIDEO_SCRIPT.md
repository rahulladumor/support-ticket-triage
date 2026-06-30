# Short video script - support-ticket triage lab

Target length: 5-7 minutes.

## 1. Opening

Hi, this is Rahul. I built a small classical ML service for routing support tickets into four queues: account-access, transaction-dispute, fraud-report, and general.

I kept the solution intentionally small because the dataset has around 400 labeled examples and the requirement is a reliable baseline, not a large ML platform.

## 2. Model choice

The model is TF-IDF word unigrams and bigrams with a LinearSVC classifier.

I chose this because the ticket messages are short, the route labels are intent-like, and lexical signals are strong. A classical model is cheap, deterministic, fast, easy to test, and easier to defend than an LLM for this scope.

I considered character n-grams for typo tolerance, but they did not improve validation results, so I kept the simpler model.

## 3. Metric reasoning

I used macro F1 as the primary metric because the classes are imbalanced. General has the most examples, while fraud-report has the fewest. Accuracy or weighted F1 could hide poor minority-class behavior.

Fraud-report is the highest-stakes route to miss, so I also track fraud recall separately. If fraud recall drops, I would rather escalate more tickets to review than silently miss possible fraud.

## 4. Imbalance handling

I used a stratified train/validation split so each class appears in both sets with similar proportions.

For model training, I used class_weight='balanced' in LinearSVC. That increases the penalty for mistakes on minority classes, mainly fraud-report.

I would know imbalance is hurting if weighted F1 remains high but macro F1 or fraud recall drops, or if the confusion matrix shows fraud-report being confused with transaction-dispute.

## 5. Code walkthrough

The important files are:

- src/ticket_triage/model.py: model definition.
- src/ticket_triage/train.py: train/validation evaluation and final model save.
- src/ticket_triage/predictor.py: simple predict(text) -> label interface.
- src/ticket_triage/predict_csv.py: holdout scoring command.
- tests/: input validation and prediction behavior tests.

The training script evaluates on a held-out validation split first, then refits on all labeled data before saving the model for hidden holdout scoring.

## 6. Trade-off I was unsure about

The decision I was most unsure about was whether to add character n-grams. They can help with misspellings and messy customer input, but on this dataset they added complexity without improving validation performance.

Given the time budget, I chose the simpler model and documented the trade-off.

## 7. Scaling or LLM discussion

At 10,000 requests per minute, this classical model is the right default because it is low-latency, cheap, and can run horizontally without external API dependency.

I would consider an LLM when tickets become longer, multilingual, ambiguous, or require reasoning over account context. In that case, I would not replace everything immediately. I would use a hybrid flow: classical classifier for most tickets, and LLM or human review for low-confidence and high-risk cases.

## 8. Closing

The main thing I optimized for here is a clean, testable, defensible ML baseline. With more time, I would add repeated cross-validation, error analysis, confidence-based escalation, and production monitoring around fraud recall and routing drift.
