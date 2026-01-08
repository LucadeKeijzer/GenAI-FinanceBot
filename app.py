import hashlib
import json
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metrics import RunMetrics, Timer, append_jsonl
from src.llm import build_evidence_packet, run_explainer_llm, run_ranker_llm
from src.pipeline import run_asset_pipeline
from src.settings import UserSettings, load_user_settings, save_user_settings
from src.wallet import load_wallet, wallet_symbols


# -----------------------------
# App config
# -----------------------------
OLLAMA_MODEL = "llama3.2:1b"
FORECAST_DAYS = 90

st.set_page_config(page_title="FinanceBot", layout="wide")

# -----------------------------
# Deterministic cache (adds real UX value)
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_run_asset_pipeline(symbol: str, period: str, forecast_days: int):
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

    y_title = "Price"
    if normalize:
        y_title = "Normalized value (start = 100)"

    for sym in symbols:
        r = results.get(sym)
        if r is None:
            continue

        prices = r.prices["Close"].squeeze().dropna()
        if getattr(prices, "ndim", 1) != 1 or prices.empty:
            continue

        df = pd.DataFrame({"date": prices.index, "price": prices.values})

        if normalize:
            df["value"] = (df["price"] / df["price"].iloc[0]) * 100.0
        else:
            df["value"] = df["price"]

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

    a, b = selected_symbols[0], selected_symbols[1]
    assets = {x.get("symbol"): x for x in evidence.get("assets", [])}
    aa = assets.get(a)
    bb = assets.get(b)
    if not aa or not bb:
        return None

    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    vola = _f(aa.get("volatility"))
    volb = _f(bb.get("volatility"))
    dda = _f(aa.get("max_drawdown"))
    ddb = _f(bb.get("max_drawdown"))
    fca = _f(aa.get("forecast_change_pct"))
    fcb = _f(bb.get("forecast_change_pct"))

    parts = []

    if vola is not None and volb is not None:
        if vola < volb:
            parts.append(f"{a} looks more stable than {b} over this period.")
        elif volb < vola:
            parts.append(f"{b} looks more stable than {a} over this period.")

    if dda is not None and ddb is not None:
        # drawdown is negative; closer to 0 is "smaller drawdowns"
        if dda > ddb:
            parts.append(f"{a} had smaller drawdowns than {b}.")
        elif ddb > dda:
            parts.append(f"{b} had smaller drawdowns than {a}.")

    if detail_level == "Advanced" and (fca is not None) and (fcb is not None):
        if abs(fca - fcb) >= 0.02:
            stronger = a if fca > fcb else b
            pct = round(max(fca, fcb) * 100)
            parts.append(
                f"{stronger} also shows a stronger upside signal, around {pct}% over the forecast horizon."
            )

    if not parts:
        return None

    return " ".join(parts[:3])


# -----------------------------
# Main app
# -----------------------------
def main():
    col1, col2 = st.columns([1, 6])

    with col1:
        st.image("assets/logo.png", width=80)

    with col2:
        st.title("FinanceBot")
        st.caption("Your financial assistant")

    # Load persisted settings (UserSettings dataclass)
    settings = load_user_settings()

    # --- Metrics init ---
    if "metrics_runs" not in st.session_state:
        st.session_state["metrics_runs"] = []

    m = RunMetrics()
    run_start = time.perf_counter()
    m.set("app_version", "v0.3")

    # Determine run reason (best-effort)
    prev_exp = st.session_state.get("_prev_experience_level")
    current_exp = settings.experience_level

    run_reason = "initial"
    if prev_exp is not None and prev_exp != current_exp:
        run_reason = "experience_change"

    m.set("run_reason", run_reason)
    m.set("experience_level", current_exp)
    st.session_state["_prev_experience_level"] = current_exp

    # Sidebar: settings
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=False):
            with st.form("settings_editor"):
                exp = st.selectbox(
                    "Experience level",
                    ["Beginner", "Intermediate", "Advanced"],
                    index=["Beginner", "Intermediate", "Advanced"].index(settings.experience_level),
                    help="Controls how technical the explanations feel. Beginner avoids jargon."
                )

                budget = st.selectbox(
                    "Budget range",
                    ["€0–100", "€100–1000", "€1000+"],
                    index=["€0–100", "€100–1000", "€1000+"].index(settings.budget_range),
                    help="Used as context for recommendations. FinanceBot does not execute trades."
                )

                submitted = st.form_submit_button("Save settings")

            if submitted:
                new_settings = UserSettings(
                    experience_level=exp,
                    budget_range=budget,
                    language="English",
                )
                save_user_settings(new_settings)
                st.success("Settings saved.")
                st.rerun()

    # Wallet
    with Timer(m, "wallet_load_seconds"):
        wallet = load_wallet()
    m.set("wallet_source", wallet.source)
    m.set("wallet_positions_count", len(wallet.positions or []))

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
    m.set("candidate_symbols_count", len(candidates))
    m.set("period", period)

    # Compute results (cached per symbol/period)
    results = {}
    failed = []

    with st.spinner("Loading market data..."):
        with Timer(m, "asset_pipeline_total_seconds"):
            for symbol in candidates:
                try:
                    results[symbol] = cached_run_asset_pipeline(symbol, period, FORECAST_DAYS)
                except Exception:
                    failed.append(symbol)

    m.set("symbols_loaded_count", len(results))
    m.set("symbols_failed_count", len(failed))

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
        "language": settings.language,
    }

    # Wallet JSON-safe dict
    wallet_json = {
        "source": wallet.source,
        "positions": wallet.positions,
    }

    # Build evidence packet
    with Timer(m, "evidence_build_seconds"):
        evidence = build_evidence_packet(results, user_settings=settings_json, wallet=wallet_json)

    # prompt-size driver: evidence chars
    evidence_chars = len(json.dumps(evidence, ensure_ascii=True, sort_keys=True))
    m.set("evidence_chars", evidence_chars)

    # -----------------------------
    # Ranker (decision-only, cached)
    # -----------------------------
    ranker_evidence = dict(evidence)
    ranker_evidence.pop("user_settings", None)  # ranker must not depend on communication settings

    ranker_key = hashlib.sha256(
        json.dumps(ranker_evidence, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if "ranker_cache" not in st.session_state:
        st.session_state["ranker_cache"] = {}

    m.set("ranker_cache_hit", False)

    with Timer(m, "ranker_total_seconds"):
        if ranker_key in st.session_state["ranker_cache"]:
            m.set("ranker_cache_hit", True)
            ranker_output, raw_ranker = st.session_state["ranker_cache"][ranker_key]
        else:
            ranker_output, raw_ranker = run_ranker_llm(ranker_evidence, model=OLLAMA_MODEL)
            st.session_state["ranker_cache"][ranker_key] = (ranker_output, raw_ranker)

    m.set("ranker_ran", not bool(m.data.get("ranker_cache_hit")))
    m.set("ranker_prompt_chars", int(ranker_output.get("_prompt_chars", 0)) if isinstance(ranker_output, dict) else 0)
    m.set("ranker_response_chars", len(raw_ranker or ""))

    final_ranking = ranker_output.get("ranking", [])
    recommended_symbol = ranker_output.get("recommended_symbol", "")

    m.set(
        "ranker_ran_on_experience_change",
        bool(m.data.get("ranker_ran")) and m.data.get("run_reason") == "experience_change"
    )
    m.set(
        "ranker_should_have_been_cached",
        m.data.get("run_reason") == "experience_change"
    )

    # -----------------------------
    # Explainer (cached by ranking + experience)
    # -----------------------------
    user_settings = evidence.get("user_settings", {}) or {}
    experience_level = str(user_settings.get("experience_level", "Beginner")).strip()

    explainer_key = hashlib.sha256(
        json.dumps({
            "ranking": final_ranking,
            "recommended": recommended_symbol,
            "experience_level": experience_level,
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if "explainer_cache" not in st.session_state:
        st.session_state["explainer_cache"] = {}

    m.set("explainer_cache_hit", False)

    with Timer(m, "explainer_total_seconds"):
        if explainer_key in st.session_state["explainer_cache"]:
            m.set("explainer_cache_hit", True)
            explainer_output, raw_explainer = st.session_state["explainer_cache"][explainer_key]
        else:
            explainer_output, raw_explainer = run_explainer_llm(
                final_ranking=final_ranking,
                evidence=evidence,  # includes user_settings on purpose
                recommended_symbol=recommended_symbol,
                ranker_notes=ranker_output.get("ranker_notes"),
                model=OLLAMA_MODEL,
            )
            st.session_state["explainer_cache"][explainer_key] = (explainer_output, raw_explainer)

    m.set("explainer_ran", not bool(m.data.get("explainer_cache_hit")))
    m.set("explainer_prompt_chars", int(explainer_output.get("_prompt_chars", 0)) if isinstance(explainer_output, dict) else 0)
    m.set("explainer_response_chars", len(raw_explainer or ""))

    m.set(
        "explainer_ran_on_experience_change",
        bool(m.data.get("explainer_ran")) and m.data.get("run_reason") == "experience_change"
    )

    # -----------------------------
    # Recommendation + Ranking UI
    # -----------------------------
    st.subheader("✅ Recommendation")
    actions = ranker_output.get("actions", {})
    confidence = ranker_output.get("confidence", "medium")

    with st.expander("How to interpret action and confidence"):
        st.markdown(
            """
**Suggested action** reflects what the analysis implies *given your current wallet and the evidence*:
- **Buy / Increase**: The asset is attractive relative to others and you currently hold little or none.
- **Hold**: The asset ranks well but does not clearly justify increasing exposure.
- **Reduce / Sell**: The asset ranks lower or carries higher risk relative to alternatives you hold.

**Confidence** reflects how strong and consistent the supporting signals are:
- **High**: Multiple indicators point in the same direction with limited tradeoffs.
- **Medium**: Signals are mixed or involve meaningful tradeoffs.
- **Low**: Evidence is weak, conflicting, or highly sensitive to assumptions.

This tool is educational only and does not provide financial advice.
"""
        )

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

    # Explainer UI
    st.subheader("🧠 Explanation")
    st.markdown(f"**{explainer_output.get('headline', '')}**")
    for b in explainer_output.get("explanation", []):
        st.markdown(f"- {b}")

    if final_ranking and len(final_ranking) >= 2:
        st.caption(f"Compared against: {final_ranking[1]} (next best in this run)")

    # -----------------------------
    # One chart + selection controls
    # -----------------------------
    ranked_available = [s for s in final_ranking if s in available_symbols]
    default_symbols = ranked_available[:2] if len(ranked_available) >= 2 else available_symbols[:2]

    selected_symbols = st.multiselect(
        "Select assets to display",
        options=available_symbols,
        default=default_symbols,
        max_selections=4,
        help="Select one asset for a historical view, or multiple assets for a normalized comparison."
    )

    if len(selected_symbols) <= 1:
        st.subheader("📈 Price history")
    else:
        st.subheader("📊 Asset comparison (normalized)")

    if selected_symbols:
        fig = build_interactive_chart(results, selected_symbols)
        st.plotly_chart(fig, width="stretch")

        insight = metric_insight(evidence, selected_symbols, settings.detail_level)
        if insight:
            st.caption(insight)
    else:
        st.caption("Select at least one asset to show the chart.")

    # -----------------------------
    # Finalize & persist metrics
    # -----------------------------
    m.set("report_total_seconds", time.perf_counter() - run_start)
    m.set("ts_unix", time.time())

    append_jsonl("logs/metrics.jsonl" ,m.data)

if __name__ == "__main__":
    main()