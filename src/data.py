import yfinance as yf
import pandas as pd


def fetch_historical_prices(symbol: str, period: str = "4y") -> pd.DataFrame:
    """
    Fetch historical daily price data for a given asset symbol.

    Args:
        symbol (str): Asset ticker (e.g. 'BTC-USD')
        period (str): Lookback period supported by yfinance (e.g. '1y', '4y')

    Returns:
        pd.DataFrame: DataFrame containing historical prices
    """
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df = df.dropna()

    return df