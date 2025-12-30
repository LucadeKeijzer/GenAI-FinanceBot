import streamlit as st
import matplotlib.pyplot as plt

from src.pipeline import run_asset_pipeline
from src.llm import build_evidence_packet, call_ollama_json

WATCHLIST = ["BTC-USD", "ETH-USD", "SPY"]
PERIOD = "4y"
FORECAST_DAYS = 90
OLLAMA_MODEL = "llama3.2:1b"


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
    st.title("FinanceBot v0.1 — Evidence-Based Analysis + GenAI Recommendation")

    with st.spinner("Analyzing assets..."):
        results = compute_all_assets(WATCHLIST, PERIOD, FORECAST_DAYS)

    with st.spinner("Generating GenAI recommendation..."):
        evidence = build_evidence_packet(results)
        genai_output, raw_llm = call_ollama_json(evidence, model=OLLAMA_MODEL)

    st.subheader("📌 GenAI Investment Recommendation (Educational)")

    rec = genai_output.get("recommended_symbol")
    ranking = genai_output.get("ranking", [])
    reasoning = genai_output.get("reasoning_bullets", [])
    risks = genai_output.get("risks", [])
    disclaimer = genai_output.get("disclaimer", "")

    if rec:
        st.markdown(f"### ✅ Suggested Focus: **{rec}**")

    if ranking:
        st.write("**Asset ranking (best → worst):**")
        st.write(" → ".join(ranking))

    if reasoning:
        st.write("**Why this ranking was generated:**")
        for bullet in reasoning:
            st.write(f"- {bullet}")

    if risks:
        st.write("**Risks & limitations:**")
        for risk in risks:
            st.write(f"- {risk}")

    if disclaimer:
        st.caption(disclaimer)

    st.divider()

    symbol = st.selectbox("Explore an asset", WATCHLIST, index=0)
    r = results[symbol]

    left, right = st.columns([2, 1])

    with left:
        fig = plot_price_and_forecast(r.prices, r.forecast, symbol)
        st.pyplot(fig)

    with right:
        st.subheader("Metrics")
        st.json(r.metrics)

    with st.expander("🔍 Raw GenAI output (debug)"):
        st.text(raw_llm)


if __name__ == "__main__":
    main()