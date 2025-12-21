from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union


class ModelName(str, Enum):
    gb = "gb"
    rf = "rf"


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    model_out: str
    train_info_out: str
    submission_out: str


@dataclass
class GBConfig:
    name: ModelName
    params: Dict[str, Any]


ModelConfig = Union[GBConfig]


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
