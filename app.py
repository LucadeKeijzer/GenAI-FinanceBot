import streamlit as st
import matplotlib.pyplot as plt

from src.pipeline import run_asset_pipeline
from src.llm import build_evidence_packet, run_ranker_llm, run_explainer_llm

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
    st.set_page_config(page_title="FinanceBot v0.2", layout="wide")
    st.title("FinanceBot v0.2 — Evidence + GenAI Ranker + GenAI Explainer (Educational)")

    with st.spinner("Analyzing assets..."):
        results = compute_all_assets(WATCHLIST, PERIOD, FORECAST_DAYS)

    # Build evidence once (used by both ranker and explainer)
    evidence = build_evidence_packet(results)

    with st.spinner("Generating GenAI ranking (Ranker)..."):
        ranker_output, raw_ranker = run_ranker_llm(evidence, model=OLLAMA_MODEL)

    final_ranking = ranker_output.get("ranking", [])
    rec = ranker_output.get("recommended_symbol")
    ranker_notes = ranker_output.get("ranker_notes", [])

    with st.spinner("Generating explanation (Explainer)..."):
        explainer_output, raw_explainer = run_explainer_llm(
            final_ranking=final_ranking,
            evidence=evidence,
            recommended_symbol=rec,
            ranker_notes=ranker_notes,
            model=OLLAMA_MODEL
        )

    st.subheader("📌 GenAI Investment Recommendation (Educational)")

    # ---- Ranker display ----
    if rec:
        st.markdown(f"### ✅ Suggested Focus: **{rec}**")

    if final_ranking:
        st.write("**Asset ranking (best → worst):**")
        st.write(" → ".join(final_ranking))

    if ranker_notes:
        st.write("**Ranker notes (decision justification):**")
        for bullet in ranker_notes:
            st.write(f"- {bullet}")

    st.divider()

    # ---- Explainer display ----
    headline = explainer_output.get("headline", "")
    explanation = explainer_output.get("explanation", [])
    tradeoffs = explainer_output.get("key_tradeoffs", [])
    risks = explainer_output.get("risks", [])
    disclaimer = explainer_output.get("disclaimer", "")

    if headline:
        st.markdown(f"### 🧠 Explanation\n**{headline}**")
    else:
        st.markdown("### 🧠 Explanation")

    if explanation:
        for para in explanation:
            st.write(f"- {para}")

    if tradeoffs:
        st.write("**Key tradeoffs:**")
        for t in tradeoffs:
            st.write(f"- {t}")

    if risks:
        st.write("**Risks & limitations:**")
        for r in risks:
            st.write(f"- {r}")

    if disclaimer:
        st.caption(disclaimer)

    st.divider()

    # ---- Evidence explorer ----
    symbol = st.selectbox("Explore an asset", WATCHLIST, index=0)
    r = results[symbol]

    left, right = st.columns([2, 1])

    with left:
        fig = plot_price_and_forecast(r.prices, r.forecast, symbol)
        st.pyplot(fig)

    with right:
        st.subheader("Metrics")
        st.json(r.metrics)

    # ---- Debug ----
    with st.expander("🔍 Raw Ranker output (debug)"):
        st.text(raw_ranker)

    with st.expander("🔍 Raw Explainer output (debug)"):
        st.text(raw_explainer)


if __name__ == "__main__":
    main()