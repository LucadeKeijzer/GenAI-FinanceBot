import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def generate_price_forecast(df: pd.DataFrame, forecast_days: int = 90) -> pd.Series:
    close = df["Close"].copy()

    # Ensure datetime index
    close.index = pd.to_datetime(close.index)

    # Attach a frequency (helps statsmodels keep a date index)
    inferred = pd.infer_freq(close.index)
    if inferred is not None:
        close = close.asfreq(inferred)
    else:
        # Market data often has no regular frequency (weekends/holidays).
        # Force daily frequency so statsmodels has a supported index.
        close = close.asfreq("D").ffill()

    model = ExponentialSmoothing(
        close,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )
    fitted_model = model.fit()
    forecast = fitted_model.forecast(forecast_days)

    # Safety: ensure date index on output
    if not isinstance(forecast.index, pd.DatetimeIndex):
        start = close.index[-1] + pd.Timedelta(days=1)
        forecast.index = pd.date_range(start=start, periods=forecast_days, freq="D")

    return forecast