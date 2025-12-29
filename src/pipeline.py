from dataclasses import dataclass
import pandas as pd

from src.data import fetch_historical_prices
from src.features import compute_basic_metrics
from src.forecast import generate_price_forecast


@dataclass
class AssetResult:
    symbol: str
    prices: pd.DataFrame
    metrics: dict
    forecast: pd.Series


def run_asset_pipeline(symbol: str, period: str = "4y", forecast_days: int = 90) -> AssetResult:
    prices = fetch_historical_prices(symbol, period=period)
    metrics = compute_basic_metrics(prices)
    forecast = generate_price_forecast(prices, forecast_days=forecast_days)

    return AssetResult(
        symbol=symbol,
        prices=prices,
        metrics=metrics,
        forecast=forecast,
    )