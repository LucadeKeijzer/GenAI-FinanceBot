import streamlit as st
import matplotlib.pyplot as plt

from src.pipeline import run_asset_pipeline

WATCHLIST = ["BTC-USD", "ETH-USD", "SPY"]
PERIOD = "4y"
FORECAST_DAYS = 90


@st.cache_data(show_spinner=False)
def compute_all_assets(watchlist, period, forecast_days):
    results = {}
    for symbol in watchlist:
        results[symbol] = run_asset_pipeline(symbol, period=period, forecast_days=forecast_days)
    return results


def plot_price_and_forecast(prices, forecast, symbol: str):
    fig, ax = plt.subplots()

    ax.plot(prices.index, prices["Close"], label="Close")
    ax.plot(forecast.index, forecast.values, label="Forecast")

    ax.set_title(f"{symbol} — Price + Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    return fig


def main():
    st.set_page_config(page_title="FinanceBot v0.1", layout="wide")
    st.title("FinanceBot v0.1 — Multi-Asset Analysis (No GenAI yet)")

    with st.spinner("Analyzing assets..."):
        results = compute_all_assets(WATCHLIST, PERIOD, FORECAST_DAYS)

    symbol = st.selectbox("Select an asset", WATCHLIST, index=0)
    r = results[symbol]

    left, right = st.columns([2, 1])

    with left:
        fig = plot_price_and_forecast(r.prices, r.forecast, symbol)
        st.pyplot(fig)

    with right:
        st.subheader("Metrics")
        st.json(r.metrics)

    st.caption("Educational demo. Not financial advice.")


if __name__ == "__main__":
    main()