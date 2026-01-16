#!/usr/bin/env bash
set -e

pixi run dvc pull

# стартуем mlflow UI в фоне
pixi run mlflow server \
  --backend-store-uri sqlite:////app/mlflow.db \
  --default-artifact-root file:/app/mlruns \
  --host 0.0.0.0 \
  --port 5000 &


MLFLOW_PID=$!

# ждём пока MLflow поднимется
for i in {1..50}; do
  if curl -sSf http://127.0.0.1:5000/ >/dev/null; then
    echo "[entrypoint] MLflow is up"
    break
  fi
  sleep 0.2
done

# чтобы обучение логировало в tracking server (а не в локальную sqlite)
#export MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

# обучение
pixi run python -m src.models.Different_models

# держим контейнер живым, чтобы UI не умер
#tail -f /dev/null
wait $MLFLOW_PID
