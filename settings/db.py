import minio

from settings.config import settings


class MinioSession:
    def __init__(self) -> None:
        self.endpoint = settings.minio_url
        self.access_key = settings.minio_access_key
        self.secret_key = settings.minio_secret_key

    def __enter__(self) -> minio.Minio:
        self.session = minio.Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False
        )
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del self.session
