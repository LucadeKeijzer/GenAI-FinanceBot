import pandas as pd
import numpy as np


def _to_scalar(x):
    """Convert pandas scalar / 1-element Series to a Python float."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def compute_basic_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"]

    ma_50 = _to_scalar(close.rolling(window=50).mean().iloc[-1])
    ma_200 = _to_scalar(close.rolling(window=200).mean().iloc[-1])
    last_price = _to_scalar(close.iloc[-1])

    if last_price > ma_50 > ma_200:
        trend = "bullish"
    elif last_price < ma_50 < ma_200:
        trend = "bearish"
    else:
        trend = "neutral"

    daily_returns = close.pct_change().dropna()

    volatility = _to_scalar(daily_returns.std() * np.sqrt(365))

    rolling_max = close.cummax()
    drawdown = (close / rolling_max) - 1.0
    max_drawdown = _to_scalar(drawdown.min())

    return {
        "last_price": last_price,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "trend": trend,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
    }