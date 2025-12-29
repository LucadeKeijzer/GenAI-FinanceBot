import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def generate_price_forecast(
    df: pd.DataFrame,
    forecast_days: int = 90
) -> pd.Series:
    """
    Generates a simple baseline price forecast using exponential smoothing.

    Args:
        df (pd.DataFrame): Historical price data with a 'Close' column
        forecast_days (int): Number of days to forecast

    Returns:
        pd.Series: Forecasted prices indexed by date
    """
    close = df["Close"]

    model = ExponentialSmoothing(
        close,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )

    fitted_model = model.fit()
    forecast = fitted_model.forecast(forecast_days)

    return forecast