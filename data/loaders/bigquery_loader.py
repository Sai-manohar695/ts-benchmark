import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from config.settings import get_dataset_config, get_settings

logger = logging.getLogger(__name__)


class BigQueryLoader:
    """
    Pulls time series data from BigQuery and caches it locally as Parquet.
    On repeat runs it reads from cache instead of hitting BQ again.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = bigquery.Client(project=self.settings.gcp_project_id)
        self.cache_dir = Path(self.settings.bq_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, start_date: str, end_date: str) -> pd.Series:
        cache_path = self.cache_dir / f"{dataset_name}_{start_date}_{end_date}.parquet"

        if self.settings.bq_use_cache and cache_path.exists():
            logger.info("Loading %s from cache", dataset_name)
            df = pd.read_parquet(cache_path)
            return self._to_series(df)

        logger.info("Querying BigQuery for %s", dataset_name)
        df = self._query(dataset_name, start_date, end_date)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached %s to %s", dataset_name, cache_path)
        return self._to_series(df)

    def _query(self, dataset_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        cfg = get_dataset_config(dataset_name)
        freq_trunc = "WEEK" if cfg["frequency"] == "W" else "DAY"

        query = f"""
            SELECT
                DATE_TRUNC({cfg['date_column']}, {freq_trunc}) AS ds,
                {cfg['aggregation']}({cfg['target_column']}) AS y
            FROM `{cfg['bq_table']}`
            WHERE DATE({cfg['date_column']}) BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY ds
            ORDER BY ds
        """

        df = self.client.query(query).to_dataframe()
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    def _to_series(self, df: pd.DataFrame) -> pd.Series:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df["y"] = df["y"].astype(float)
        series = df.set_index("ds")["y"]
        series.index = pd.DatetimeIndex(series.index, freq="infer")
        return series