from abc import ABC, abstractmethod
from typing import Any


class BaseEstimator(ABC):
    @abstractmethod
    def __init__(self, model_path) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def validate(self, config) -> dict[Any, float]:
        raise NotImplementedError
