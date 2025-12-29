from src.data import fetch_historical_prices
from src.features import compute_basic_metrics

df = fetch_historical_prices("BTC-USD")
metrics = compute_basic_metrics(df)

print(metrics)