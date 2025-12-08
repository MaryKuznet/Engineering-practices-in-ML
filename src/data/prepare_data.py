from pathlib import Path

import pandas as pd


def clean_titanic(raw_path: str, output_path: str) -> None:
    df = pd.read_csv(raw_path)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "Survived" in df.columns and "Survived" not in numeric_cols:
        numeric_cols.append("Survived")

    df = df[numeric_cols]

    df = df.fillna(df.median(numeric_only=True))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset cleaned and saved to: {output_path}")


if __name__ == "__main__":
    clean_titanic(
        raw_path="data/raw/train.csv", output_path="data/processed/train_clean.csv"
    )
