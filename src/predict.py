from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PROCESSED_TEST_PATH = "data/processed/test_clean.csv"
MODEL_PATH = "models/model.pkl"
SUBMISSION_PATH = "data/predictions/submission.csv"


def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run training first (dvc repro)."
        )

    model = joblib.load(MODEL_PATH)

    df_test = pd.read_csv(PROCESSED_TEST_PATH)

    if "PassengerId" not in df_test.columns:
        raise ValueError("Expected 'PassengerId' in processed test set for submission")

    passenger_id = df_test["PassengerId"]
    X_test = df_test.drop(columns=["PassengerId"])

    preds = model.predict(X_test)

    out = pd.DataFrame({"PassengerId": passenger_id, "Survived": preds})
    Path(SUBMISSION_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved predictions to {SUBMISSION_PATH}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
