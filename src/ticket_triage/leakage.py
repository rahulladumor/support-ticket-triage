"""Template-aware grouping to expose train/validation leakage.

The training data is templated: ~400 rows reduce to ~71 distinct sentence
templates (about 5-6 near-duplicate rows each), differing only by a swapped
coin name, device, dollar amount, or a polite prefix/suffix. A random
train/validation split therefore puts the *same* template in both train and
validation, so TF-IDF memorizes it and every metric reads a perfect 1.0000.

That number is leakage, not skill. To estimate how the model generalizes to the
hidden holdout (which will use phrasings the model has never seen), we group
rows by their normalized template and split by group, so no template appears on
both sides. `template_skeleton` produces that group key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Polite openers and closers that carry no routing signal.
_PREFIXES = ["hello team", "quick question", "please help", "hello", "hi", "hey", "urgent"]
_SUFFIXES = [
    "this is time sensitive",
    "appreciate any help",
    "please advise",
    "let me know",
    "thank you",
    "thanks",
]
# Slot fillers that vary across near-duplicate rows without changing the route.
_COINS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "cardano", "ada",
    "polygon", "matic", "litecoin", "ltc", "dogecoin", "doge", "xrp", "usdc", "usdt",
]
_DEVICES = [
    "android app", "new iphone", "iphone", "desktop browser", "work computer",
    "laptop", "desktop", "tablet", "phone",
]
_SLOT_WORDS = sorted(_COINS + _DEVICES, key=len, reverse=True)


def template_skeleton(text: str) -> str:
    """Collapse a message to the template it was generated from.

    Two rows that differ only by coin, device, amount, or polite prefix/suffix
    map to the same skeleton, so they can be kept in the same CV fold.
    """
    t = text.lower().strip()

    for prefix in _PREFIXES:
        for sep in (": ", ", ", ". ", " "):
            if t.startswith(prefix + sep):
                t = t[len(prefix) + len(sep):]
                break

    stripped = True
    while stripped:
        stripped = False
        trimmed = t.rstrip(" .!,")
        for suffix in _SUFFIXES:
            if trimmed.endswith(suffix):
                t = trimmed[: -len(suffix)]
                stripped = True
                break

    t = re.sub(r"\$[\d,]+(?:\.\d+)?", " <AMT> ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", " <NUM> ", t)
    for word in _SLOT_WORDS:
        t = re.sub(r"\b" + re.escape(word) + r"\b", " <SLOT> ", t)

    t = re.sub(r"[^a-z<> ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def group_labels(texts: Iterable[str]) -> list[str]:
    """Return one template-skeleton group key per message, for grouped CV."""
    return [template_skeleton(t) for t in texts]
