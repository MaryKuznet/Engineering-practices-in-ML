from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

# ====== Настройки проекта ======
TRAIN_PATH = "data/processed/train_clean.csv"
CLEARML_PROJECT = "titanic-dz3-tracking"
TASK_NAME = "gb_single_run"


@dataclass
class GBConfig:
    # модель
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 1.0
    random_state: int = 42

    # оценка
    cv_splits: int = 5


def load_train(path: str = TRAIN_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in train data")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    # на всякий случай
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    return X, y


def build_model(cfg: GBConfig) -> Any:
    # SimpleImputer нужен, если в данных есть NaN
    gb = GradientBoostingClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        random_state=cfg.random_state,
    )
    return make_pipeline(SimpleImputer(strategy="median"), gb)


def main() -> None:
    # 1) Создаём Task в ClearML (всё “авто” начнёт логироваться: git/commit, окружение, stdout)
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        tags=["titanic", "gb", "single-run"],
    )
    logger = task.get_logger()

    # 2) Конфиг + логирование параметров
    cfg = GBConfig()
    task.connect(asdict(cfg))  # параметры появятся в UI

    # 3) Данные
    X, y = load_train(TRAIN_PATH)

    # полезно залогировать список фич
    feature_cols_text = "\n".join(list(X.columns))
    task.upload_artifact(name="feature_columns.txt", artifact_object=feature_cols_text)

    # 4) CV-предсказания (чтобы посчитать метрики “как в честном CV”)
    model = build_model(cfg)
    cv = StratifiedKFold(
        n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state
    )

    # cross_val_predict даёт out-of-fold предикты для каждой строки
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    # вероятности нужны для roc_auc (если бинарная классификация)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))
    roc_auc = float(roc_auc_score(y, y_proba))

    # 5) Логирование метрик
    logger.report_scalar("accuracy", "cv_oof", acc, iteration=0)
    logger.report_scalar("f1", "cv_oof", f1, iteration=0)
    logger.report_scalar("roc_auc", "cv_oof", roc_auc, iteration=0)

    # 6) Confusion matrix как изображение (удобно для UI)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    fig_path = "artifacts/confusion_matrix.png"
    os.makedirs("artifacts", exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    task.upload_artifact(name="confusion_matrix.png", artifact_object=fig_path)

    # 7) Итоговое обучение на всех данных + сохранение модели
    model.fit(X, y)
    model_path = "artifacts/gb_model.joblib"
    joblib.dump(model, model_path)
    task.upload_artifact(name="model.joblib", artifact_object=model_path)

    # 8) Небольшая табличка результатов как артефакт
    results = pd.DataFrame(
        [
            {"metric": "accuracy", "value": acc},
            {"metric": "f1", "value": f1},
            {"metric": "roc_auc", "value": roc_auc},
        ]
    )
    results_path = "artifacts/metrics.csv"
    results.to_csv(results_path, index=False)
    task.upload_artifact(name="metrics.csv", artifact_object=results_path)

    print("Done. Logged GB single run to ClearML.")
    task.close()


if __name__ == "__main__":
    main()
