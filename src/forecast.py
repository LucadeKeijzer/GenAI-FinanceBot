import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def generate_price_forecast(df: pd.DataFrame, forecast_days: int = 90) -> pd.Series:
    """
    Generates a simple baseline price forecast using exponential smoothing.

    Args:
        df (pd.DataFrame): Historical price data with a 'Close' column
        forecast_days (int): Number of days to forecast

    Returns:
        pd.Series: Forecasted prices indexed by date
    """
    close = df["Close"].copy()

    # Ensure datetime index
    close.index = pd.to_datetime(close.index)

    # Try to attach a frequency (helps statsmodels keep a date index)
    inferred = pd.infer_freq(close.index)
    if inferred is not None:
        close = close.asfreq(inferred)

    model = ExponentialSmoothing(
        close,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )
    fitted_model = model.fit()
    forecast = fitted_model.forecast(forecast_days)

    # If statsmodels still returns an integer index, rebuild a date index
    if not isinstance(forecast.index, pd.DatetimeIndex):
        start = close.index[-1] + pd.Timedelta(days=1)
        forecast.index = pd.date_range(start=start, periods=forecast_days, freq="D")

    return forecast