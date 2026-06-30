import pytest

from ticket_triage.model import build_model
from ticket_triage.predictor import TicketTriageService


TRAIN_TEXTS = [
    "I cannot login and the OTP code never arrives",
    "My account is locked after password reset",
    "Someone withdrew BTC without my permission, I think I was hacked",
    "A phishing email stole my credentials and funds are gone",
    "My ETH withdrawal says completed but I never received the money",
    "The cardano trade total is wrong, please review the transaction",
    "How do staking rewards work on Ethereum",
    "Do you support SOL deposits in the app",
]

TRAIN_LABELS = [
    "account-access",
    "account-access",
    "fraud-report",
    "fraud-report",
    "transaction-dispute",
    "transaction-dispute",
    "general",
    "general",
]


def fitted_service():
    model = build_model()
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return TicketTriageService(
        model=model,
        labels=["account-access", "fraud-report", "general", "transaction-dispute"],
    )


def test_predict_returns_expected_label_for_fraud_language():
    service = fitted_service()
    prediction = service.predict("Unauthorized BTC withdrawal, my account may be hacked")
    assert prediction == "fraud-report"


def test_predict_rejects_empty_input():
    service = fitted_service()
    with pytest.raises(ValueError):
        service.predict("")
