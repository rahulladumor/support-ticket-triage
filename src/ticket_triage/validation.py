"""Input validation helpers."""

from __future__ import annotations

from collections.abc import Iterable


def validate_text(text: str) -> str:
    """Validate and normalize one inbound support message.

    The model expects a real text message. Empty strings are rejected because
    they usually indicate an integration bug upstream, not a valid customer
    ticket.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be empty")

    return normalized


def validate_texts(texts: Iterable[str]) -> list[str]:
    """Validate a batch of support messages."""
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not a single string")

    validated = [validate_text(text) for text in texts]
    if not validated:
        raise ValueError("texts must contain at least one message")
    return validated
