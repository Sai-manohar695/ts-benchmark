import logging

from data.loaders.bigquery_loader import BigQueryLoader

logging.basicConfig(level=logging.INFO)

loader = BigQueryLoader()

datasets = [
    ("iowa_liquor_sales", "2020-01-01", "2022-12-31"),
    ("chicago_taxi_trips", "2020-01-01", "2022-12-31"),
    ("nyc_citi_bike", "2015-01-01", "2018-05-31"),
]

for name, start, end in datasets:
    print(f"\n--- {name} ---")
    series = loader.load(name, start, end)
    print(f"Shape:      {series.shape}")
    print(f"Date range: {series.index.min()} to {series.index.max()}")
    print(f"Nulls:      {series.isnull().sum()}")
    print(f"Sample:\n{series.head(3)}")