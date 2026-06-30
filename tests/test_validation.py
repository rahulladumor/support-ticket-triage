import pytest

from ticket_triage.validation import validate_text


def test_validate_text_rejects_empty_message():
    with pytest.raises(ValueError):
        validate_text("   ")


def test_validate_text_rejects_non_string():
    with pytest.raises(TypeError):
        validate_text(None)  # type: ignore[arg-type]
