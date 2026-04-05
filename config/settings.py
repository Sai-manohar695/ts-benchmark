from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent


class Settings(BaseSettings):
    gcp_project_id: str = Field(..., env="GCP_PROJECT_ID")
    bq_location: str = Field(default="US", env="BQ_LOCATION")
    bq_cache_dir: str = Field(default="data/cache", env="BQ_CACHE_DIR")
    bq_use_cache: bool = Field(default=True, env="BQ_USE_CACHE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataset_config(dataset_name: str) -> dict:
    cfg = load_yaml("datasets.yaml")
    if dataset_name not in cfg["datasets"]:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(cfg['datasets'].keys())}"
        )
    return cfg["datasets"][dataset_name]