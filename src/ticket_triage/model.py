"""Model factory for the support-ticket triage classifier."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE


def build_model() -> Pipeline:
    """Build the chosen classical ML baseline.

    Why this model:
    - TF-IDF word n-grams fit the short support-ticket domain well.
    - LinearSVC is strong for small-to-medium text classification datasets.
    - class_weight='balanced' reduces the chance that the minority fraud class
      is ignored by the decision boundary.

    Character n-grams were tested but did not improve validation performance on
    this dataset, so the simpler model is kept for readability and speed.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    C=0.8,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
