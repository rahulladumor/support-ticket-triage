import joblib
import pandas as pd

from ticket_triage.config import LABELS
from ticket_triage.model import build_model
from ticket_triage.predict_csv import main as predict_csv_main


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


def test_predict_csv_writes_prediction_column(tmp_path, monkeypatch):
    model = build_model()
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)

    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "holdout.csv"
    output_path = tmp_path / "predictions.csv"

    joblib.dump({"model": model, "labels": LABELS}, model_path)
    input_path.write_text(
        "text\nUnauthorized BTC withdrawal from my wallet\nHow do staking rewards work?\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "predict_csv",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(model_path),
        ],
    )

    predict_csv_main()

    predictions = pd.read_csv(output_path)
    assert "predicted_label" in predictions.columns
    assert len(predictions) == 2
    assert set(predictions["predicted_label"]).issubset(set(LABELS))
