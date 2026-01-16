from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.core.config_store import ConfigStore

from src.config import Config

cs = ConfigStore.instance()
cs.store(name="schema", node=Config)


@hydra.main(version_base=None, config_path="../conf", config_name="config")  # type: ignore[misc]
def main(cfg: Config) -> None:
    if not Path(cfg.data.model_out).exists():
        raise FileNotFoundError(
            f"Model not found: {cfg.data.model_out}. Run `dvc repro` first."
        )

    model = joblib.load(cfg.data.model_out)

    df_test = pd.read_csv(cfg.data.test_path)
    if "PassengerId" not in df_test.columns:
        raise ValueError("Expected 'PassengerId' in processed test set for submission")

    passenger_id = df_test["PassengerId"]
    X_test = df_test.drop(columns=["PassengerId"])

    preds = model.predict(X_test)

    out = pd.DataFrame({"PassengerId": passenger_id, "Survived": preds})

    Path(cfg.data.submission_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.data.submission_out, index=False)

    print(f"Saved predictions to {cfg.data.submission_out}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
