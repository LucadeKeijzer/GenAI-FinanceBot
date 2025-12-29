from src.data import fetch_historical_prices
from src.forecast import generate_price_forecast

df = fetch_historical_prices("BTC-USD")
forecast = generate_price_forecast(df, forecast_days=30)

print(forecast.head())
print(forecast.tail())