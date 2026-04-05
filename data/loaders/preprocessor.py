import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Cleans and prepares a raw time series for modelling.

    Steps applied in order:
        1. Fill missing dates so the index has no gaps
        2. Interpolate missing values (linear)
        3. Remove outliers beyond 3 standard deviations
        4. Clip negative values to zero (counts and revenue can't be negative)
    """

    def __init__(self, frequency: str = "D"):
        self.frequency = frequency

    def run(self, series: pd.Series) -> pd.Series:
        series = self._fill_gaps(series)
        series = self._interpolate(series)
        series = self._remove_outliers(series)
        series = self._clip_negatives(series)
        logger.info(
            "Preprocessed series: %d periods, %d nulls remaining",
            len(series),
            series.isnull().sum(),
        )
        return series

    def _fill_gaps(self, series: pd.Series) -> pd.Series:
        full_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=self.frequency,
        )
        return series.reindex(full_index)

    def _interpolate(self, series: pd.Series) -> pd.Series:
        return series.interpolate(method="linear").ffill().bfill()

    def _remove_outliers(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        outliers = (series < lower) | (series > upper)
        if outliers.sum() > 0:
            logger.info("Replacing %d outliers with interpolated values", outliers.sum())
            series[outliers] = None
            series = series.interpolate(method="linear").ffill().bfill()
        return series

    def _clip_negatives(self, series: pd.Series) -> pd.Series:
        return series.clip(lower=0)