from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.core.config_store import ConfigStore

from src.config import Config
from src.models.Different_models import make_model

cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


def _load_processed_train(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in processed train set")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    return X, y


@hydra.main(version_base=None, config_path="../conf", config_name="config")  # type: ignore[misc]
def main(cfg: Config) -> None:
    X, y = _load_processed_train(cfg.data.train_path)

    # Собираем params строго по выбранной модели (без OmegaConf)
    model_name = str(cfg.model.name)
    if model_name == "gb":
        params = {
            "learning_rate": cfg.model.params["learning_rate"],
            "n_estimators": cfg.model.params["n_estimators"],
            "max_depth": cfg.model.params["max_depth"],
        }
    else:
        raise ValueError(f"Unsupported model config: {type(cfg.model)}")

    model = make_model(model_name, params)
    model.fit(X, y)

    Path(cfg.data.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.data.model_out)

    info = {
        "model_name": model_name,
        "params": params,
        "n_train_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
    }
    Path(cfg.data.train_info_out).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.data.train_info_out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {cfg.data.model_out}")
    print(f"Saved train info to {cfg.data.train_info_out}")


if __name__ == "__main__":
    main()
