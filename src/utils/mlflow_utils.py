from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, ParamSpec, TypeVar

import mlflow
import mlflow.sklearn

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    tracking_uri: str | None = None
    tags: Optional[Dict[str, str]] = None


def setup_mlflow(cfg: RunConfig) -> None:
    """
    Central place to configure MLflow storage.
    For local reproducibility we typically use file store (default ./mlruns).
    """
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


@contextmanager
def mlflow_run(
    run_name: str | None = None, tags: Dict[str, str] | None = None
) -> Iterator[mlflow.ActiveRun]:
    """Context manager wrapper around mlflow.start_run() + tag setting."""
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: Dict[str, Any]) -> None:
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: Dict[str, float]) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def track_experiment(
    *,
    run_name_fn: Callable[..., str] | None = None,
    tags_fn: Callable[..., Dict[str, str]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            run_name = run_name_fn(*args, **kwargs) if run_name_fn else func.__name__
            tags = tags_fn(*args, **kwargs) if tags_fn else None

            with mlflow_run(run_name=run_name, tags=tags):
                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    params = result.get("params")
                    metrics = result.get("metrics")
                    if isinstance(params, dict):
                        log_params(params)
                    if isinstance(metrics, dict):
                        log_metrics(metrics)
                return result

        return wrapper

    return decorator


def log_sklearn_model(model: Any, name: str = "model") -> None:
    mlflow.sklearn.log_model(model, name=name, registered_model_name=f"titanic-{name}")
