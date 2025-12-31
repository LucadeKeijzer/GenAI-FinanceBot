import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.pipeline import run_asset_pipeline
from src.wallet import load_wallet, wallet_symbols
from src.settings import load_user_settings
from src.llm import build_evidence_packet, run_ranker_llm, run_explainer_llm

# -----------------------------
# App config
# -----------------------------
OLLAMA_MODEL = "llama3.2:1b"
FORECAST_DAYS = 90

st.set_page_config(page_title="FinanceBot v0.3", layout="wide")


# -----------------------------
# Deterministic cache (adds real UX value)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_run_asset_pipeline(symbol: str, period: str, forecast_days: int):
    # Cache per (symbol, period, forecast_days)
    return run_asset_pipeline(symbol=symbol, period=period, forecast_days=forecast_days)


# -----------------------------
# Helper: context label for hover
# (simple + deterministic; no metric names, no jargon)
# -----------------------------
def _context_label(vol: float, dd: float) -> str:
    try:
        vol = float(vol)
    except Exception:
        vol = 0.0
    try:
        dd = float(dd)
    except Exception:
        dd = 0.0

    # dd is typically negative
    if dd <= -0.50:
        return "Deep drawdown risk"
    if dd <= -0.20:
        return "Meaningful drawdown risk"
    if vol >= 0.70:
        return "Very volatile period"
    if vol >= 0.35:
        return "Volatile period"
    return "More stable behavior"


# -----------------------------
# Helper: build the one chart
# -----------------------------
def build_interactive_chart(results: dict, symbols: list[str]) -> go.Figure:
    normalize = len(symbols) > 1
    fig = go.Figure()

    for sym in symbols:
        r = results.get(sym)
        if r is None:
            continue

        # Ensure 1D series
        prices = r.prices["Close"].squeeze().dropna()
        if getattr(prices, "ndim", 1) != 1 or prices.empty:
            continue

        df = pd.DataFrame(
            {"date": prices.index, "price": prices.values}
        )

        if normalize:
            df["value"] = (df["price"] / df["price"].iloc[0]) * 100.0
            y_title = "Normalized value (start = 100)"
        else:
            df["value"] = df["price"]
            y_title = "Price"

        vol = r.metrics.get("volatility", 0.0)
        dd = r.metrics.get("max_drawdown", 0.0)
        context = _context_label(vol, dd)

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                mode="lines",
                name=sym,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    + f"{sym}: "
                    + "%{y:.2f}<br>"
                    + f"{context}"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title="Assets",
    )
    return fig


# -----------------------------
# Helper: short natural-language insight
# (only shown if it adds value; no internal metric names)
# -----------------------------
def metric_insight(evidence: dict, selected_symbols: list[str], detail_level: str) -> str | None:
    if len(selected_symbols) < 2:
        return None

    # Compare first two selected to keep it concise
    a, b = selected_symbols[0], selected_symbols[1]
    assets = {x.get("symbol"): x for x in evidence.get("assets", [])}
    aa = assets.get(a)
    bb = assets.get(b)
    if not aa or not bb:
        return None

    vola = aa.get("volatility")
    volb = bb.get("volatility")
    dda = aa.get("max_drawdown")
    ddb = bb.get("max_drawdown")
    fca = aa.get("forecast_change_pct")
    fcb = bb.get("forecast_change_pct")

    # Robust casts
    try:
        vola = float(vola)
    except Exception:
        vola = None
    try:
        volb = float(volb)
    except Exception:
        volb = None
    try:
        dda = float(dda)
    except Exception:
        dda = None
    try:
        ddb = float(ddb)
    except Exception:
        ddb = None
    try:
        fca = float(fca)
    except Exception:
        fca = None
    try:
        fcb = float(fcb)
    except Exception:
        fcb = None

    parts = []

    # Stability comparison
    if vola is not None and volb is not None:
        if vola < volb:
            parts.append(f"{a} looks more stable than {b} over this period.")
        elif volb < vola:
            parts.append(f"{b} looks more stable than {a} over this period.")

    # Drawdown comparison
    if dda is not None and ddb is not None:
        # drawdown is negative; closer to 0 is "smaller drawdowns"
        if dda > ddb:
            parts.append(f"{a} had smaller drawdowns than {b}.")
        elif ddb > dda:
            parts.append(f"{b} had smaller drawdowns than {a}.")

    # Advanced: allow ONE rounded % if it helps
    if detail_level == "Advanced" and (fca is not None) and (fcb is not None):
        if abs(fca - fcb) >= 0.02:
            stronger = a if fca > fcb else b
            pct = round(max(fca, fcb) * 100)  # 0.027 -> 3%
            parts.append(f"{stronger} also shows a stronger upside signal, around {pct}% over the forecast horizon.")

    if not parts:
        return None

    # Keep it short
    return " ".join(parts[:3])


# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("📈 FinanceBot v0.3")

    # Load persisted settings (UserSettings dataclass)
    settings = load_user_settings()

    # Load wallet (Wallet dataclass)
    wallet = load_wallet()

    with st.sidebar:
        st.header("Wallet")
        st.caption(f"Source: {wallet.source}")
        if wallet.positions:
            for p in wallet.positions:
                sym = p.get("symbol", "")
                qty = p.get("quantity", 0)
                st.write(f"- {sym}: {qty}")
        else:
            st.caption("No holdings found.")

        if st.button("🔄 Refresh"):
            st.rerun()

    # Period control near the chart (not hidden in sidebar)
    st.subheader("📊 Evidence view")
    period = st.radio(
        "Analysis period",
        ["1y", "3y", "5y", "max"],
        horizontal=True,
        help="Controls how much historical data is used for analysis and ranking."
    )

    # Candidate universe = wallet + starter list (locked design)
    starter = ["BTC-USD", "ETH-USD", "SPY", "SOL-USD"]
    candidates = sorted(set(starter + wallet_symbols(wallet)))

    # Compute results (cached per symbol/period)
    results = {}
    failed = []
    with st.spinner("Loading market data..."):
        for symbol in candidates:
            try:
                results[symbol] = cached_run_asset_pipeline(symbol, period, FORECAST_DAYS)
            except Exception:
                failed.append(symbol)

    if failed:
        st.warning(
            "Some symbols failed to load and were excluded from ranking: "
            + ", ".join(failed)
        )

    available_symbols = sorted(results.keys())
    if not available_symbols:
        st.error("No symbols could be loaded. Check your data source / internet connection and try again.")
        return

    # Convert settings dataclass -> JSON-safe dict for evidence
    settings_json = {
        "experience_level": settings.experience_level,
        "budget_range": settings.budget_range,
        "detail_level": settings.detail_level,
        "language": settings.language,
    }

    # Wallet JSON-safe dict (positions already list-of-dicts)
    wallet_json = {
        "source": wallet.source,
        "positions": wallet.positions,
    }

    # Build evidence using your existing helper (prevents None forecast values)
    evidence = build_evidence_packet(results, user_settings=settings_json, wallet=wallet_json)

    # Ranker should NOT see detail_level (communication-only)
    ranker_evidence = dict(evidence)
    ranker_user_settings = dict(ranker_evidence.get("user_settings", {}))
    ranker_user_settings.pop("detail_level", None)
    ranker_evidence["user_settings"] = ranker_user_settings

    # Ranker + Explainer
    ranker_output, raw_ranker = run_ranker_llm(ranker_evidence, model=OLLAMA_MODEL)
    final_ranking = ranker_output.get("ranking", [])
    recommended_symbol = ranker_output.get("recommended_symbol", "")

    explainer_output, raw_explainer = run_explainer_llm(
        final_ranking=final_ranking,
        evidence=evidence,
        recommended_symbol=recommended_symbol,
        ranker_notes=ranker_output.get("ranker_notes"),
        model=OLLAMA_MODEL,
    )

    # Recommendation + Ranking (restored)
    st.subheader("✅ Recommendation")
    actions = ranker_output.get("actions", {})
    confidence = ranker_output.get("confidence", "medium")

    if recommended_symbol:
        rec_action = actions.get(recommended_symbol, "consider")
        st.markdown(f"**Recommended asset:** `{recommended_symbol}`")
        st.markdown(f"**Suggested action:** `{rec_action}`  |  **Confidence:** `{confidence}`")
    else:
        st.warning("No recommendation produced (recommended_symbol missing).")

    st.subheader("📊 Ranking")
    if final_ranking:
        for sym in final_ranking:
            st.write(f"- {sym} → **{actions.get(sym, 'consider')}**")
    else:
        st.warning("Ranking is empty. Check raw ranker output in debug.")

    # Explainer
    st.subheader("🧠 Explanation")
    st.caption(f"Explanation detail level: {settings.detail_level}")
    st.markdown(f"**{explainer_output.get('headline', '')}**")
    for b in explainer_output.get("explanation", []):
        st.markdown(f"- {b}")

    if final_ranking and len(final_ranking) >= 2:
        st.caption(f"Compared against: {final_ranking[1]} (next best in this run)")

    # -----------------------------
    # One chart + selection controls
    # -----------------------------
    # Default selection: top 2 from ranking that are actually available
    ranked_available = [s for s in final_ranking if s in available_symbols]
    default_symbols = ranked_available[:2] if len(ranked_available) >= 2 else available_symbols[:2]

    selected_symbols = st.multiselect(
        "Select assets to display",
        options=available_symbols,
        default=default_symbols,
        max_selections=4,
        help="Select one asset for a historical view, or multiple assets for a normalized comparison."
    )

    # Chart title AFTER selection (prevents UnboundLocalError)
    if len(selected_symbols) <= 1:
        st.subheader("📈 Price history")
    else:
        st.subheader("📊 Asset comparison (normalized)")

    if selected_symbols:
        fig = build_interactive_chart(results, selected_symbols)
        st.plotly_chart(fig, use_container_width=True)

        insight = metric_insight(evidence, selected_symbols, settings.detail_level)
        if insight:
            st.caption(insight)
    else:
        st.caption("Select at least one asset to show the chart.")

    # Debug
    with st.expander("🔍 Raw Ranker output (debug)"):
        st.text(raw_ranker)

    with st.expander("🔍 Raw Explainer output (debug)"):
        st.text(raw_explainer)


if __name__ == "__main__":
    main()