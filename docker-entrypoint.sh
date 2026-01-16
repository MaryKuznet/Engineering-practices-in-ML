#!/usr/bin/env bash
set -e

# Если смонтировали локальное DVC-хранилище — переписываем remote
if [ -d "/dvc_storage" ]; then
  echo "[entrypoint] Using /dvc_storage as DVC remote"
  pixi run dvc remote modify localstore url /dvc_storage || true
  pixi run dvc remote default localstore || true
fi

exec "$@"
