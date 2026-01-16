from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DOCS_LATEST = ROOT / "docs" / "experiments" / "latest.md"

JSON_CANDIDATE = ROOT / "reports" / "train_reports" / "train_info.json"
CSV_CANDIDATE = ROOT / "reports" / "experiments.csv"


def load_results() -> pd.DataFrame:
    if CSV_CANDIDATE.exists():
        return pd.read_csv(CSV_CANDIDATE)

    if JSON_CANDIDATE.exists():
        data = json.loads(JSON_CANDIDATE.read_text(encoding="utf-8"))

        # Вариант 1: { "runs": [ ... ] }
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            runs = data["runs"]
            return pd.DataFrame(runs)

        # Вариант 2: [ { ... }, { ... } ] (список прогонов)
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty JSON list in {JSON_CANDIDATE}")
            return pd.DataFrame(data)

        # Вариант 3: один объект (как у тебя)
        if isinstance(data, dict):
            # превращаем в одну строку таблицы
            row = {
                "model": data.get("model_name"),
                "params": json.dumps(data.get("params", {}), ensure_ascii=False),
                "n_train_rows": data.get("n_train_rows"),
                "n_features": data.get("n_features"),
                "feature_columns": ", ".join(data.get("feature_columns", [])),
            }
            # если метрики появятся — тоже подхватим
            metrics = data.get("metrics", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    row[k] = v

            return pd.DataFrame([row])

        raise ValueError(f"Unexpected JSON format in {JSON_CANDIDATE}")

    raise FileNotFoundError(
        "No experiment results found. Expected one of:\n"
        f"- {CSV_CANDIDATE}\n"
        f"- {JSON_CANDIDATE}\n"
    )


def build_plot(df: pd.DataFrame) -> Path | None:
    # подстрой под свои метрики: accuracy/f1/roc_auc etc.
    metric_cols = [c for c in ["accuracy", "f1", "roc_auc"] if c in df.columns]
    if not metric_cols:
        return None

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "metrics_bar.png"

    # барчарт: каждая метрика по экспериментам
    df_plot = df.copy()
    label_col = "model" if "model" in df_plot.columns else None
    if label_col is None:
        df_plot["run"] = [f"run_{i}" for i in range(len(df_plot))]
        label_col = "run"

    df_plot = df_plot[[label_col] + metric_cols].set_index(label_col)

    ax = df_plot.plot(kind="bar", rot=30)
    ax.set_ylabel("score")
    ax.set_title("Experiment metrics comparison")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def to_md_table(df: pd.DataFrame) -> Any:
    # аккуратно выбираем колонки
    preferred = [
        c for c in ["model", "params", "accuracy", "f1", "roc_auc"] if c in df.columns
    ]
    df_show = df[preferred] if preferred else df
    return df_show.to_markdown(index=False)


def write_report(df: pd.DataFrame, fig_path: Path | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    table_md = to_md_table(df)

    fig_md = ""
    if fig_path is not None:
        # ссылка для README/Markdown
        fig_md = f"\n\n## Plots\n\n![]({fig_path.as_posix()})\n"

    full_md = f"""# Experiment report

This file is generated automatically by `pixi run report`.

## Comparison table

{table_md}
{fig_md}
"""

    (REPORTS_DIR / "experiments.md").write_text(full_md, encoding="utf-8")

    # Дублируем “последний отчёт” в docs, чтобы он попал на Git Pages
    DOCS_LATEST.parent.mkdir(parents=True, exist_ok=True)
    DOCS_LATEST.write_text(full_md, encoding="utf-8")


def main() -> None:
    df = load_results()
    fig_path = build_plot(df)
    write_report(df, fig_path)


if __name__ == "__main__":
    main()
