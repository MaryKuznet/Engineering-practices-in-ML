from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv("data/raw/train.csv")

    X = df.drop(columns=["Survived"])
    X = X.select_dtypes(include=["number"])
    y = df["Survived"]
    return X, y


def train_and_log_model() -> None:
    mlflow.set_experiment("titanic-experiment")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    C = 0.1
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=C),
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.set_tag("task", "titanic_survival")
        mlflow.set_tag("model_family", "logistic_regression")
        mlflow.set_tag("dataset", "titanic_simple_v1")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imputer", "median")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("C", C)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="titanic-logreg",
        )

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_and_log_model()
