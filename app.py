from src.data import fetch_historical_prices

df = fetch_historical_prices("BTC-USD")
print(df.head())
print(df.tail())