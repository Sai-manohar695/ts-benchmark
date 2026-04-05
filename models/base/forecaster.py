"""
Base interface that every forecasting model implements.
The validation engine only calls fit() and predict() — 
so swapping models in and out requires zero changes upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """
    Returned by every model's predict() call.

    forecast  — point estimates, shape (horizon,)
    lower     — lower 95% confidence bound, may be None
    upper     — upper 95% confidence bound, may be None
    model_name — e.g. "arima", "prophet", "lstm", "tft"
    meta      — model-specific extras (AIC, attention weights, etc.)
    """

    forecast: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    model_name: str
    meta: dict = field(default_factory=dict)

    def to_dataframe(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame({"forecast": self.forecast}, index=index)
        if self.lower is not None:
            df["lower"] = self.lower
        if self.upper is not None:
            df["upper"] = self.upper
        return df


class BaseForecaster(ABC):

    def __init__(self, config: dict) -> None:
        self.config = config
        self._is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        ...

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"Call fit() before predict() on {self.name}."
            )