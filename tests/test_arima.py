import logging

from data.loaders.bigquery_loader import BigQueryLoader
from data.loaders.preprocessor import Preprocessor
from models.classical.arima import ARIMAForecaster

logging.basicConfig(level=logging.INFO)

# load and preprocess
loader = BigQueryLoader()
series = loader.load("iowa_liquor_sales", "2020-01-01", "2022-12-31")
preprocessor = Preprocessor(frequency="W")
clean = preprocessor.run(series)

# config matches what we'll put in models.yaml
config = {
    "auto_select": True,
    "order_search": {
        "p_range": [0, 2],
        "d_range": [0, 1],
        "q_range": [0, 2],
    },
    "max_iterations": 50,
}

# fit on first 80% of data
split = int(len(clean) * 0.8)
train = clean.iloc[:split]
test = clean.iloc[split:]

print(f"\nTrain size: {len(train)} weeks")
print(f"Test size:  {len(test)} weeks")

model = ARIMAForecaster(config=config, seasonal_period=52)
model.fit(train)

result = model.predict(horizon=len(test))

print(f"\nModel:    {result.model_name}")
print(f"Order:    {result.meta['order']}")
print(f"AIC:      {result.meta['aic']:.2f}")
print(f"\nForecast (first 5):\n{result.forecast[:5]}")
print(f"\nActual   (first 5):\n{test.values[:5]}")