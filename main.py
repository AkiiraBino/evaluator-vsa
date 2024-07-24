
from shutil import rmtree

import mlflow
import torch
import yaml
from loguru import logger

from settings.config import settings
from src.estimators import Detector
from src.utils import get_file_content, init_service, write_file_content

if __name__ == "__main__":
    init_service()

    with open(settings.config_path) as f:
        config = yaml.safe_load(stream=f)
    detector = Detector(settings.detectors_path)
    model2config = {}

    for key_model, value_model in detector.model.names.items(): # type: ignore
        for key_config, value_config in config["names"].items():
            if value_model == value_config:
                model2config[key_config] = key_model

    logger.info(f"Join model and config id successfuly {model2config}")

    for path in settings.test_set_path.joinpath("val/labels").glob("*"):
        contents = get_file_content(path)
        for content in contents:
            content[0] = model2config[content[0]]
        write_file_content(path, contents)

    logger.info("Replace classes successfuly")

    mlflow.pytorch.autolog()


    with mlflow.start_run(
        experiment_id=settings.experiment_id,
        run_name=settings.model_name
    ), torch.no_grad():
        result = detector.validate("data/example.yaml", batch=1)

        mlflow.log_metrics(
            {
                key.replace("(", " ").replace(")", " "): item
                for key, item in result.results_dict.items() # type: ignore
            }
        )

    rmtree(settings.test_set_path)
