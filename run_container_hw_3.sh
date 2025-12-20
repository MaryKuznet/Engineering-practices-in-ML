#!/usr/bin/env bash
set -e

pixi run dvc pull

# стартуем mlflow UI в фоне
pixi run mlflow server \
  --backend-store-uri sqlite:///./mlruns/mlflow.db \
  --host 0.0.0.0 \
  --port 5000

# обучение
pixi run python -m src.models.Different_models

# держим контейнер живым, чтобы UI не умер
tail -f /dev/null
