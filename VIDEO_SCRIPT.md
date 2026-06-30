# Short video script - support-ticket triage lab

Target length: 5-7 minutes.

## 1. Opening

Hi, this is Rahul. I built a small classical ML service for routing support tickets into four queues: account-access, transaction-dispute, fraud-report, and general.

I kept the solution intentionally small because the dataset has around 400 labeled examples and the requirement is a reliable baseline, not a large ML platform.

## 2. Model choice

The model is TF-IDF word unigrams and bigrams with a LinearSVC classifier.

I chose this because the ticket messages are short, the route labels are intent-like, and lexical signals are strong. A classical model is cheap, deterministic, fast, easy to test, and easier to defend than an LLM for this scope.

I considered character n-grams for typo tolerance. I will come back to why I left them out, because that decision only became answerable after I fixed a leakage problem.

## 3. The leakage I caught (the most important part)

The first thing I did was look at the data, and the rows are templated. About 400 rows collapse to roughly 70 distinct sentence templates, five or six near-duplicates each, differing only by a coin name, a device, a dollar amount, or a polite prefix like "Hi" or "Urgent".

That matters because a random train/validation split puts the same template on both sides. TF-IDF just memorizes it, and every metric reads a perfect 1.0000. That is leakage, not skill, and the hidden holdout will use phrasings the model has never seen.

So I group rows by their template skeleton and use StratifiedGroupKFold, which keeps each template entirely inside one fold. The honest number drops to about 0.80 macro F1 and about 0.72 fraud recall. That grouped number is my actual prediction for the hidden holdout. I keep the leaky 1.0000 in the report only to show the gap.

## 4. Metric and imbalance handling

I used macro F1 as the primary metric because the classes are imbalanced. General has the most examples, fraud-report the fewest, so accuracy or weighted F1 could hide poor minority-class behavior.

Fraud-report is the highest-stakes route to miss, so I track fraud recall separately. Honestly it is about 0.72 and noisy, down to 0.64 on some splits, which means roughly one in four novel-phrasing fraud tickets would be misrouted. I would rather escalate those to human review than miss them.

For training I used class_weight='balanced' in LinearSVC, which raises the penalty on minority-class mistakes. I would know imbalance is hurting if the weighted-versus-macro F1 gap widens, or if grouped fraud recall falls below threshold while accuracy stays high.

## 5. Code walkthrough

The important files are:

- src/ticket_triage/model.py: model definition.
- src/ticket_triage/leakage.py: template-skeleton grouping that drives the honest evaluation.
- src/ticket_triage/train.py: leaky and grouped evaluation plus final model save.
- src/ticket_triage/predictor.py: simple predict(text) -> label interface.
- src/ticket_triage/predict_csv.py: holdout scoring command.
- tests/: input validation, prediction behavior, and template-grouping tests.

The training script reports both the leaky split and the grouped (honest) cross-validation, then refits on all labeled data before saving the model for hidden holdout scoring.

## 6. Trade-off I was unsure about

Whether to add character n-grams. They help with misspellings and messy input. But I could only judge them after fixing the leakage, because against the leaky 1.0000 every variant looks identical. Under grouped CV, adding char n-grams lifts macro F1 a little, from about 0.80 to 0.83, but it does not help fraud recall and makes it noisier. Since fraud recall is the metric I care most about, I kept the simpler word-only model. That is the clearest case where the honest evaluation changed my decision.

## 7. Scaling or LLM discussion

At 10,000 requests per minute, this classical model is the right default because it is low-latency, cheap, and can run horizontally without external API dependency.

I would consider an LLM when tickets become longer, multilingual, ambiguous, or require reasoning over account context. In that case, I would not replace everything immediately. I would use a hybrid flow: classical classifier for most tickets, and LLM or human review for low-confidence and high-risk cases.

## 8. Closing

The main thing I optimized for here is a clean, testable, defensible ML baseline. With more time, I would add repeated cross-validation, error analysis, confidence-based escalation, and production monitoring around fraud recall and routing drift.
