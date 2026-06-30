# Short video script - support-ticket triage lab

Target length: 5-7 minutes. These are notes I talk from, not a word-for-word read.

## 1. Opening

Hi, I'm Rahul. This is a small classical ML service that routes support tickets into four queues: account-access, transaction-dispute, fraud-report, and general.

I kept it deliberately small. There are only about 400 labeled examples, and the ask is a baseline I can trust and defend, not a big ML platform.

## 2. Model choice

It's TF-IDF, word unigrams and bigrams, into a LinearSVC.

The messages are short and the routes are basically intent labels, so the lexical signal is strong, which is exactly where a linear model over TF-IDF does well. It's also cheap, deterministic, and easy to test, and I can explain every part of it. An LLM would be heavier to run and harder to defend for something this size.

I did try character n-grams for typo tolerance. I'll come back to that, because it only became a real decision once I'd fixed a leakage problem.

## 3. The leakage I caught (the part I'd lead with)

So the first thing I did was actually read the data, and the rows are templated. Those ~400 rows really come from about 70 sentence templates, five or six near-duplicates each, with just a coin name, a device, a dollar amount, or a polite opener like "Hi" or "Urgent" swapped in.

That matters because if I split randomly, the same template ends up in train and in validation. TF-IDF memorizes it, and every metric comes back a perfect 1.0000. That's not skill, that's leakage, and your hidden holdout is going to be worded in ways the model has never seen.

So I group the rows by their template skeleton and use StratifiedGroupKFold, which keeps a whole template inside one fold. Do that and the honest number drops to about 0.80 macro F1 and about 0.72 fraud recall. That grouped number is what I actually expect on your holdout. I left the 1.0000 in the report on purpose, just to show the gap.

## 4. Metric and imbalance

Macro F1 is my primary metric, because the classes are imbalanced. General is the biggest, fraud-report the smallest, and accuracy or weighted F1 would let me look good while quietly missing the small classes.

Fraud-report is the most expensive one to get wrong, so I track its recall on its own. Honestly it's around 0.72 and noisy, down to 0.64 on a bad split, so roughly one in four fraud tickets in new wording would get misrouted. I'd rather push those to a human than miss them.

For training I set class_weight='balanced', which makes minority-class mistakes cost more. The way I'd catch imbalance hurting me is the gap between weighted and macro F1 opening up, or grouped fraud recall sliding under threshold while accuracy still looks fine.

## 5. Code walkthrough

Quick tour of the files:

- model.py, the model itself.
- leakage.py, the template-skeleton grouping the honest eval depends on.
- train.py, runs both the leaky and the grouped evaluation, then saves the model.
- predictor.py, the predict(text) -> label interface.
- predict_csv.py, the holdout scoring command.
- tests/, input validation, a prediction check, and the grouping tests.

train.py reports both splits, then refits on all the labeled data before it saves, so the shipped model has seen everything.

## 6. A call I wasn't sure about

The character n-grams. They'd help with misspellings and messy input, so it was tempting. But I couldn't judge them honestly until the leakage was fixed, since against a 1.0000 everything ties. Under grouped CV they bump macro F1 a little, 0.80 to about 0.83, but they don't help fraud recall and they make it jump around more. Fraud recall is the thing I care about, so I kept the simpler word-only model. That's the one spot where fixing the eval actually changed what I shipped.

## 7. Scale, and when I'd reach for an LLM

At 10,000 requests a minute I'd keep this model. It's fast, it's cheap, and it scales out with no API in the loop.

I'd bring in an LLM when the tickets get longer, multilingual, genuinely ambiguous, or need reasoning over someone's account history. Even then I wouldn't rip this out. I'd run a hybrid: the classifier handles the easy majority, and anything low-confidence or fraud-shaped goes to an LLM or a person.

## 8. Closing

What I went for here is a small, tested baseline I can stand behind, with an evaluation that tells the truth instead of a flattering 1.0000. With more time I'd add repeated CV with intervals, some error analysis on fraud versus dispute, a confidence threshold for escalation, and monitoring on fraud recall and drift once it's live.
