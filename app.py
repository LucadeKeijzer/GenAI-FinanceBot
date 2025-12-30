import streamlit as st
import matplotlib.pyplot as plt

from src.pipeline import run_asset_pipeline
from src.llm import build_evidence_packet, run_ranker_llm, run_explainer_llm
from src.settings import load_user_settings, save_user_settings, UserSettings
from src.wallet import load_wallet  # v0.3 Step 2

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


def plot_price_and_forecast(asset_result):
    """
    Minimal meaningful chart:
    - historical close
    - forecast line (simple)
    """
    prices = asset_result.prices
    forecast = asset_result.forecast

    fig, ax = plt.subplots()
    ax.plot(prices.index, prices["Close"], label="Historical Close")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title(asset_result.symbol)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


def render_first_run_setup() -> None:
    st.subheader("🛠️ First-time setup")

    experience_help = (
        "Beginner: plain language, minimal jargon.\n"
        "Intermediate: some investing terms, light metrics.\n"
        "Advanced: more detail, may reference metrics/assumptions."
    )

    budget_help = (
        "Used to frame position sizing language.\n"
        "Educational only (not financial advice)."
    )

    detail_help = (
        "Simple: no unexplained jargon, fewer numbers.\n"
        "Advanced: may include more metrics and technical detail."
    )

    language_help = "English only for now. Dutch can be added later."

    with st.form("setup_form"):
        experience_level = st.selectbox(
            "Experience level",
            ["Beginner", "Intermediate", "Advanced"],
            index=0,
            help=experience_help,
        )

        budget_range = st.selectbox(
            "Budget range",
            ["€0–100", "€100–1000", "€1000+"],
            index=1,
            help=budget_help,
        )

        detail_level = st.selectbox(
            "Detail level",
            ["Simple", "Advanced"],
            index=0,
            help=detail_help,
        )

        language = st.selectbox(
            "Language",
            ["English"],
            index=0,
            help=language_help,
        )

        submitted = st.form_submit_button("Save settings")

    if submitted:
        settings = UserSettings(
            experience_level=experience_level,
            budget_range=budget_range,
            detail_level=detail_level,
            language=language,
        )
        save_user_settings(settings)
        st.success("Settings saved! Rerunning...")
        st.rerun()

    # IMPORTANT: stop execution so v0.2 pipeline does not run until settings exist.
    st.stop()


def main():
    st.set_page_config(page_title="FinanceBot v0.3", layout="wide")
    st.title("FinanceBot v0.3 — Personalized Evidence + GenAI Ranker + GenAI Explainer (Educational)")

    # -----------------------------
    # v0.3 Step 1: Load User Settings (or first-run setup)
    # -----------------------------
    settings = load_user_settings()
    if settings is None:
        render_first_run_setup()

    # -----------------------------
    # v0.3 Step 2: Load Wallet (API -> CSV fallback)
    # MUST happen before we render sidebar that references wallet
    # -----------------------------
    wallet = load_wallet()

    # -----------------------------
    # Sidebar (read-only settings + wallet display)
    # -----------------------------
    with st.sidebar:
        st.subheader("⚙️ User Settings")
        st.write(f"Experience: **{settings.experience_level}**")
        st.write(f"Budget: **{settings.budget_range}**")
        st.write(f"Detail: **{settings.detail_level}**")
        st.write(f"Language: **{settings.language}**")

        st.divider()

        st.subheader("👛 Wallet")
        st.caption(f"Source: {wallet.source}")
        if wallet.positions:
            for p in wallet.positions:
                st.write(f"- {p['symbol']}: {p['quantity']}")
        else:
            st.write("No holdings found.")

    # -----------------------------
    # v0.2 pipeline continues unchanged below this line
    # -----------------------------
    with st.spinner("Analyzing assets..."):
        results = compute_all_assets(WATCHLIST, PERIOD, FORECAST_DAYS)

    evidence = build_evidence_packet(results)

    with st.spinner("Generating GenAI ranking (Ranker)..."):
        ranker_output, raw_ranker = run_ranker_llm(evidence, model=OLLAMA_MODEL)

    final_ranking = ranker_output.get("ranking", [])
    recommended_symbol = ranker_output.get("recommended_symbol", "")
    ranker_notes = ranker_output.get("ranker_notes", [])

    with st.spinner("Generating explanation (Explainer)..."):
        explainer_output, raw_explainer = run_explainer_llm(
            final_ranking=final_ranking,
            evidence=evidence,
            model=OLLAMA_MODEL,
            recommended_symbol=recommended_symbol,
            ranker_notes=ranker_notes,
        )

    # ---- UI ----
    st.subheader("✅ Recommendation")
    if recommended_symbol:
        st.markdown(f"**Recommended asset:** `{recommended_symbol}`")
    else:
        st.warning("No recommendation returned.")

    st.subheader("📊 Ranking")
    if final_ranking:
        st.write(final_ranking)
    else:
        st.warning("No ranking returned.")

    st.subheader("🧠 Explanation")
    headline = explainer_output.get("headline", "")
    explanation = explainer_output.get("explanation", [])
    risks = explainer_output.get("risks", [])
    disclaimer = explainer_output.get("disclaimer", "")

    if headline:
        st.markdown(f"**{headline}**")

    if explanation:
        for line in explanation:
            st.write(f"- {line}")

    if risks:
        st.markdown("**Risks / Limitations**")
        for r in risks:
            st.write(f"- {r}")

    if disclaimer:
        st.info(disclaimer)

    st.subheader("📈 Evidence (Charts + Metrics)")
    symbols = list(results.keys())
    left, right = st.columns(2)

    with left:
        st.subheader("Charts")
        for sym in symbols:
            plot_price_and_forecast(results[sym])

    with right:
        st.subheader("Metrics")
        for sym in symbols:
            st.markdown(f"**{sym}**")
            st.json(results[sym].metrics)

    # ---- Debug ----
    with st.expander("🔍 Raw Ranker output (debug)"):
        st.text(raw_ranker)

    with st.expander("🔍 Raw Explainer output (debug)"):
        st.text(raw_explainer)


if __name__ == "__main__":
    main()