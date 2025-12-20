#!/usr/bin/env bash
set -e

pixi run dvc pull

# стартуем mlflow UI в фоне
pixi run mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns &

# обучение
pixi run python src/models/LogReg.py

# держим контейнер живым, чтобы UI не умер
tail -f /dev/null
