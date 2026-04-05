import logging

from data.loaders.bigquery_loader import BigQueryLoader
from data.loaders.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO)

loader = BigQueryLoader()

datasets = [
    ("iowa_liquor_sales", "2020-01-01", "2022-12-31", "W"),
    ("chicago_taxi_trips", "2020-01-01", "2022-12-31", "D"),
    ("nyc_citi_bike", "2015-01-01", "2018-05-31", "D"),
]

for name, start, end, freq in datasets:
    print(f"\n--- {name} ---")
    series = loader.load(name, start, end)
    preprocessor = Preprocessor(frequency=freq)
    clean = preprocessor.run(series)
    print(f"Raw shape:    {series.shape}")
    print(f"Clean shape:  {clean.shape}")
    print(f"Nulls:        {clean.isnull().sum()}")
    print(f"Min value:    {clean.min():.2f}")
    print(f"Max value:    {clean.max():.2f}")
    print(f"Sample:\n{clean.head(3)}")