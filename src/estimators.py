from pathlib import Path
from typing import Any, Literal

import torch
from cv2.typing import MatLike
from loguru import logger
from ultralytics import YOLO

from src.base.BaseEstimator import BaseEstimator


class Detector(BaseEstimator):
    def __init__(self, model_path: str | Path) -> None:
        try:
            self.model = YOLO(model_path)
            self.device: Literal["cuda", "cpu"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        except FileNotFoundError as e:
            logger.error(f"Path {model_path} for model not exists")
            raise e

    def predict(self, frame: MatLike) -> list[Any]:
        self.model.eval()
        return self.model(frame)

    def validate(self, config, **kwargs) -> dict[Any, float]:
        return self.model.val(data=config, **kwargs)
