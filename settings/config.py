from pathlib import Path

import mlflow
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, DirectoryPath, Field, FilePath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(".env", override=True)


class Settings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    root_dir: DirectoryPath = Path.cwd()

    minio_url: str
    minio_access_key: str = Field(validation_alias="ACCESS_KEY")
    minio_secret_key: str = Field(validation_alias="SECRET_KEY")

    mlflow_tracking_uri: AnyHttpUrl

    bucket_models: str
    bucket_test_set: str

    model_name: str
    models_dir: str
    test_set_dir: str
    experiment_name: str

    @computed_field()
    @property
    def models_path(self) -> DirectoryPath:
        return self.root_dir.joinpath(self.models_dir)

    @computed_field()
    @property
    def detectors_path(self) -> FilePath:
        return self.models_path.joinpath(self.model_name)

    @computed_field()
    @property
    def test_set_path(self) -> DirectoryPath:
        return self.root_dir.joinpath(self.test_set_dir)

    @computed_field()
    @property
    def experiment_id(self) -> str:
        if not mlflow.get_experiment_by_name(self.experiment_name):
            return mlflow.create_experiment(self.experiment_name)

        return mlflow.get_experiment_by_name(
            self.experiment_name
        ).experiment_id  # type: ignore

    @computed_field()
    @property
    def config_path(self) -> FilePath:
        return self.test_set_path.joinpath("coco.yaml")


settings = Settings()  # type: ignore
