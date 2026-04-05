import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.base import BaseForecaster, ForecastResult

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):

    name = "arima"

    def __init__(self, config: dict, seasonal_period: int = 7) -> None:
        super().__init__(config)
        self.seasonal_period = seasonal_period
        self._model_fit = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, self.seasonal_period)

        if self.config.get("auto_select", True):
            best_order, best_seasonal = self._grid_search(series)

        logger.info(
            "Fitting SARIMA%s x %s on %d observations",
            best_order, best_seasonal, len(series)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=best_order,
                seasonal_order=best_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model_fit = model.fit(
                disp=False,
                maxiter=self.config.get("max_iterations", 50),
            )

        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastResult:
        self._check_fitted()
        forecast_obj = self._model_fit.get_forecast(steps=horizon)
        mean = forecast_obj.predicted_mean.values
        ci = forecast_obj.conf_int(alpha=0.05)
        return ForecastResult(
            forecast=mean,
            lower=ci.iloc[:, 0].values,
            upper=ci.iloc[:, 1].values,
            model_name=self.name,
            meta={
                "aic": self._model_fit.aic,
                "order": self._model_fit.model.order,
                "seasonal_order": self._model_fit.model.seasonal_order,
            },
        )

    def _grid_search(self, series: pd.Series) -> tuple:
        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_seasonal = (0, 0, 0, 0)

        p_range = range(0, self.config["order_search"]["p_range"][1] + 1)
        d_range = range(0, self.config["order_search"]["d_range"][1] + 1)
        q_range = range(0, self.config["order_search"]["q_range"][1] + 1)

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(
                                series,
                                order=(p, d, q),
                                seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            )
                            result = model.fit(disp=False, maxiter=50)
                            if result.aic < best_aic:
                                best_aic = result.aic
                                best_order = (p, d, q)
                    except Exception:
                        continue

        logger.info("Best ARIMA order: %s AIC: %.2f", best_order, best_aic)
        return best_order, best_seasonal