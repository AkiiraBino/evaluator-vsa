from pathlib import Path
from typing import Iterator

import mlflow
from loguru import logger
from minio.datatypes import Object
from ultralytics import settings as settings_yolo

from settings.config import settings
from settings.db import MinioSession


def init_service():
    logger.info("Begin init service")
    mlflow.set_tracking_uri(str(settings.mlflow_tracking_uri))
    settings_yolo.update({"datasets_dir": "data/", "mlflow": True})


    if not settings.models_path.exists():
        settings.models_path.mkdir()

    if not settings.test_set_path.exists():
        settings.test_set_path.mkdir()

    with MinioSession() as session:
        session.fget_object(
            settings.bucket_models,
            settings.model_name,
            str(settings.detectors_path),
        )

        objects: Iterator[Object] = session.list_objects(
            settings.bucket_test_set, recursive=True
        )

        for obj in objects:
            name = str(obj.object_name)
            path = str(settings.test_set_path.joinpath(name))
            session.fget_object(settings.bucket_test_set, name, path)

    logger.info("Init service successfuly")


def get_file_content(path: str | Path) -> list[list[float | int]]:
    with open(path) as f:
        contents = f.read().split("\n")
        contents = [
            [
                float(c) if i != 0 else int(c)
                for i, c in enumerate(content.split(" "))
            ]
            for content in contents[:-1]
            if content != [""]
        ]

    return contents


def write_file_content(path: str | Path, content) -> None:
    with open(path, "w") as file:
        for inner_list in content:
            line = " ".join(map(str, inner_list))
            file.write(f"{line}\n")
