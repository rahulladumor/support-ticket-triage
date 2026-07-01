# Video walkthrough script (spoken, ~7 min)

Delivery notes:
- Talk to the camera like I am explaining to a colleague, not reading.
- Slow-medium pace. [/] means breathe and pause. Bold means land the word.
- Lift the energy on section 3, the leakage. That is the differentiator.
- Target ~7 min, stays under the 8 minute limit.

---

## 1. Open (~30s)

Hi, I am Rahul. [/] So this is a small classical ML service that routes support tickets into four queues... account-access, transaction-dispute, fraud-report, and general. [/] I kept it deliberately small, because see, the data is only around 400 labelled examples. And what they are asking for is a baseline I can actually **trust and defend**. Not a big ML platform. So that is the mindset I went in with.

## 2. Model choice (~45s)

So the model itself, it is TF-IDF, word unigrams and bigrams, going into a LinearSVC. [/] Why this one? The messages are short, and the routes are basically intent labels, so the lexical signal is very strong. And this is exactly where a linear model on top of TF-IDF does really well. It is cheap, it is deterministic, it is fast, and most important, I can explain **every single part** of it. An LLM here would be heavier to run and harder to defend for something this size. [/] I did try character n-grams also. But I will come to that... because that decision only made sense after I fixed one leakage problem.

## 3. The leakage I caught (~90s) [ENERGY UP]

So now the main part. The first thing I did, I actually sat and read the data. And I saw, the rows are templated. [/] These 400 rows, they are actually coming from only around **70 sentence templates**. Five, six near-duplicates each. Only the coin name is changing, or the device, or the dollar amount, or some polite opener like "Hi" or "Urgent". [/] Now why does this matter? See, if I split randomly, the same template goes into train, and also into validation. So TF-IDF just memorises it. And every metric comes back a perfect 1.0000. [/ lower voice] But that is not skill. That is leakage only. And the hidden holdout you will test me on, that will be worded in ways my model has never seen. So I cannot trust that 1.0. [/] So what I did is, I group the rows by their template skeleton, and I use StratifiedGroupKFold, so one full template stays inside one fold. And the moment I do that, the honest number drops... to around **0.80 macro F1, and around 0.72 fraud recall**. [/] That grouped number, that is my real prediction for your holdout. And I kept the 1.0 in the report also, just to show the gap. Because that gap is the whole story.

## 4. Metric choice (~45s)

Okay so, why macro F1 as the main metric? Because the classes are imbalanced. General is the biggest, fraud is the smallest. So accuracy or weighted F1, they will let me look good, while quietly missing the fraud tickets. Macro F1 treats all four routes equally. [/] But for fraud, I do one more thing. I track fraud **recall** separately. Because see, the cost is not symmetric. If I miss a fraud, the customer loses money, trust is gone. If I raise a false fraud alarm, worst case the ticket just goes to a human. So missing is much more costly than a false alarm. That is why I watch recall, not precision.

## 5. Class imbalance (~40s)

Now imbalance, how did I handle it. Three simple things. Stratified splits, so fraud is present in every fold. class_weight balanced in the SVC, so those 50 fraud rows are not drowned by 160 general ones. And I use fraud recall as a metric, so imbalance cannot hide behind accuracy. [/] One thing I deliberately did **not** do... SMOTE, or oversampling. Because on 400 templated rows, that will just copy templates and make my leakage worse. [/] And how will I know if imbalance is hurting me? Simple. If weighted F1 stays high but macro F1 falls. Or if fraud recall slips below my threshold. Or in the confusion matrix, if fraud is getting confused with transaction-dispute... because those two are very close, right? "Unauthorised withdrawal" versus "a withdrawal I never received."

## 6. The call I was unsure about (~40s)

The one decision I was really unsure about... the character n-grams. And honestly, I could only judge it **after** fixing the leakage. Because before that, everything is 1.0, so there is nothing to compare. [/] After grouping, when I add char n-grams, macro F1 goes up a little, 0.81 to 0.83. But fraud recall actually drops, and it becomes more noisy. So there is a clear trade-off. Char n-grams give me typo tolerance and a small macro gain, but they cost me stability on the metric I care about most. So I chose the simpler word model. [/] And I will be honest, this is a close call. The difference is within one standard deviation. On a bigger, messier dataset, I would definitely revisit it.

## 7. Scale, and when an LLM makes sense (~50s)

Last one. If this scales to 10,000 requests a minute, or if I swap for an LLM... when is each one right? [/] At 10,000 a minute, this classical model is clearly the right choice. Prediction is sub-millisecond, CPU only, no network, almost zero cost, and deterministic. If I put an LLM on that hot path, I am adding latency, cost running into thousands per day, rate limits, and an external dependency at my busiest point. [/] So when is the LLM the right call? When tickets become long, or multilingual, or they need reasoning over the account history. Or when the categories keep changing and I do not have labelled data. [/] But honestly, in real production, I would not pick one. I would do a hybrid. The classical model handles the easy majority, fast and cheap. And only the low-confidence ones, or the fraud-shaped ones, I send to an LLM or a human. Cost of the classical, judgement of the LLM, only where it is worth it.

## 8. Close (~20s)

So that is it. What I focused on is a small, tested baseline that I can fully stand behind... with an evaluation that tells the truth, not a nice-looking 1.0. [/] With more time, I would add repeated CV with confidence intervals, some error analysis on fraud versus dispute, a confidence threshold for escalation, and monitoring on fraud recall once it is live. [/] Thank you.
