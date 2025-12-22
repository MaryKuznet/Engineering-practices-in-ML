from __future__ import annotations

from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from clearml import Task
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

TRAIN_PATH = "data/processed/train_clean.csv"

# ClearML: проект/названия лучше задавать константами
CLEARML_PROJECT = "titanic-dz3-tracking"
CLEARML_BASE_TASK = "grid-search"


def load_train(path: str = TRAIN_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' in processed train set")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])

    return X, y


def make_model(model_name: str, params: Dict[str, Any]) -> Any:
    if model_name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                max_iter=int(params.get("max_iter", 2000)),
                solver="lbfgs",
            ),
        )

    if model_name == "svc":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=float(params["C"]),
                kernel=str(params["kernel"]),
                gamma=str(params["gamma"]),
            ),
        )

    if model_name == "linear_svc":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                C=float(params["C"]),
                max_iter=int(params.get("max_iter", 5000)),
                random_state=42,
            ),
        )

    if model_name == "knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=int(params["n_neighbors"])),
        )

    if model_name == "gnb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GaussianNB(var_smoothing=float(params["var_smoothing"])),
        )

    if model_name == "perceptron":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Perceptron(
                alpha=float(params["alpha"]),
                max_iter=int(params["max_iter"]),
                random_state=42,
            ),
        )

    if model_name == "sgd":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(
                alpha=float(params["alpha"]),
                loss=str(params["loss"]),
                random_state=42,
            ),
        )

    if model_name == "dt":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=(
                    None if params["max_depth"] is None else int(params["max_depth"])
                ),
                min_samples_split=int(params["min_samples_split"]),
                random_state=42,
            ),
        )

    if model_name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=(
                    None if params["max_depth"] is None else int(params["max_depth"])
                ),
                random_state=42,
            ),
        )

    if model_name == "gb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                random_state=42,
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_name(model_name: str, params: Dict[str, Any]) -> str:
    parts = "_".join(f"{k}={v}" for k, v in params.items())
    return f"{model_name}__{parts}"


def tags_for(model_name: str) -> Dict[str, str]:
    return {
        "task": "titanic_survival",
        "model": model_name,
        "dataset": TRAIN_PATH,
    }


def train_eval_one(
    model_name: str,
    params: Dict[str, Any],
    X: Any,
    y: Any,
    cv_splits: int = 5,
    run_index: int = 0,
) -> Dict[str, Any]:
    # 1) Создаём отдельный ClearML Task для каждой комбинации параметров
    task = Task.init(
        project_name=CLEARML_PROJECT,
        task_name=run_name(model_name, params),
        task_type=Task.TaskTypes.training,
        tags=list(tags_for(model_name).values()),  # теги в ClearML — список строк
    )

    # 2) Логируем параметры (появятся в UI)
    task.connect(
        {
            "model_name": model_name,
            **params,
            "cv_splits": cv_splits,
            "train_path": TRAIN_PATH,
        }
    )

    logger = task.get_logger()

    model = make_model(model_name, params)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1"],
        return_train_score=False,
    )

    metrics = {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_f1_mean": float(scores["test_f1"].mean()),
    }

    # 3) Метрики в ClearML (пойдут в Scalars)
    logger.report_scalar(
        "cv_accuracy_mean", "cv", metrics["cv_accuracy_mean"], iteration=0
    )
    logger.report_scalar("cv_f1_mean", "cv", metrics["cv_f1_mean"], iteration=0)

    # 4) Fit на всех данных и сохраняем модель как артефакт
    model.fit(X, y)
    model_path = f"models/{run_name(model_name, params)}.joblib"
    import os

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    task.upload_artifact(name="model", artifact_object=model_path)

    # 5) То, что ты логировал как текст в MLflow
    feature_cols_text = "\n".join(list(X.columns))
    task.upload_artifact(name="feature_columns.txt", artifact_object=feature_cols_text)

    # Закрываем таск аккуратно (чтобы в UI было Completed)
    task.close()

    return {"params": {"model_name": model_name, **params}, "metrics": metrics}


def grids() -> List[Tuple[str, List[Dict[str, Any]]]]:
    return [
        (
            "logreg",
            [{"C": c, "max_iter": 2000} for c in [0.1, 0.3, 1.0, 3.0, 10.0]],
        ),  # 5
        (
            "rf",
            [
                {"n_estimators": n, "max_depth": d, "min_samples_split": 2}
                for n, d in [(200, 3), (300, 5), (500, 5), (800, None), (1000, None)]
            ],
        ),  # 5
        (
            "svc",
            [
                {"C": c, "kernel": k, "gamma": g}
                for c, k, g in [
                    (0.5, "rbf", "scale"),
                    (1.0, "rbf", "scale"),
                    (2.0, "rbf", "scale"),
                ]
            ],
        ),  # 3
        ("knn", [{"n_neighbors": k} for k in [3, 5, 7]]),  # 3
        (
            "gb",
            [
                {"learning_rate": lr, "n_estimators": n, "max_depth": d}
                for lr, n, d in [(0.05, 200, 2), (0.05, 400, 2)]
            ],
        ),  # 2
    ]


def main() -> None:
    # В ClearML tracking_uri не нужен — он берётся из clearml-init конфигурации
    X, y = load_train(TRAIN_PATH)

    total = 0
    run_index = 0
    for model_name, params_list in grids():
        for params in params_list:
            total += 1
            run_index += 1
            train_eval_one(model_name, params, X, y, cv_splits=5, run_index=run_index)

    print(f"Done. Logged {total} runs to ClearML project '{CLEARML_PROJECT}'.")


if __name__ == "__main__":
    main()
