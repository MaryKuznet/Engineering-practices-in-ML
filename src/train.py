from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from src.Different_models import make_model

PROCESSED_TRAIN_PATH = "data/processed/train_clean.csv"
MODEL_OUT_PATH = "models/model.pkl"
TRAIN_INFO_PATH = "reports/train_reports/train_info.json"

BEST_MODEL_NAME = "gb"
BEST_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "n_estimators": 400,
    "max_depth": 2,
}


def load_processed_train(
    path: str = PROCESSED_TRAIN_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in processed train set")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    return X, y


def main() -> None:
    X, y = load_processed_train(PROCESSED_TRAIN_PATH)

    model = make_model(BEST_MODEL_NAME, BEST_PARAMS)
    model.fit(X, y)

    Path(MODEL_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT_PATH)

    Path(TRAIN_INFO_PATH).parent.mkdir(parents=True, exist_ok=True)
    info = {
        "model_name": BEST_MODEL_NAME,
        "params": BEST_PARAMS,
        "n_train_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
    }
    with open(TRAIN_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {MODEL_OUT_PATH}")
    print(f"Saved train info to {TRAIN_INFO_PATH}")


if __name__ == "__main__":
    main()
