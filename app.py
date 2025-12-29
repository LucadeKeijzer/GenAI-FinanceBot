from src.pipeline import run_asset_pipeline

WATCHLIST = ["BTC-USD", "ETH-USD", "SPY"]

results = []
for symbol in WATCHLIST:
    r = run_asset_pipeline(symbol, period="4y", forecast_days=30)
    results.append(r)
    print(symbol, r.metrics["trend"], r.metrics["last_price"])
    print(r.forecast.head(2))
    print("---")