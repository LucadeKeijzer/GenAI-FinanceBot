import sys
import time
from pathlib import Path

# Ensure src import works when running from /scripts
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_asset_pipeline
from src.llm import build_evidence_packet, run_ranker_llm, run_explainer_llm

WATCHLIST = ["BTC-USD", "ETH-USD", "SPY"]

# Keep this small for fast iteration
PERIOD = "1y"
FORECAST_DAYS = 14
MODEL = "llama3.2:1b"


def log(msg: str):
    print(msg, flush=True)


def timed(label: str, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    log(f"{label} ✅ ({dt:.1f}s)")
    return out


def main():
    log("TEST_LLM: script started")

    # 1) Run deterministic pipeline
    log("TEST_LLM: running asset pipelines...")
    results = {}

    def run_pipelines():
        for symbol in WATCHLIST:
            log(f" - pipeline: {symbol} ...")
            results[symbol] = run_asset_pipeline(symbol, period=PERIOD, forecast_days=FORECAST_DAYS)
        return results

    timed("TEST_LLM: pipelines complete", run_pipelines)

    # 2) Build evidence packet
    log("TEST_LLM: building evidence packet...")
    evidence = timed("TEST_LLM: evidence built", lambda: build_evidence_packet(results))
    log(f"TEST_LLM: evidence symbols = {[a['symbol'] for a in evidence.get('assets', [])]}")

    # 3) Call Ranker
    log("TEST_LLM: calling ranker...")
    # If your run_ranker_llm doesn't accept timeout yet, just remove timeout_s from this call.
    ranker_parsed, ranker_raw = timed(
        "TEST_LLM: ranker returned",
        lambda: run_ranker_llm(evidence, model=MODEL)
    )

    log("\nRANKER PARSED JSON:")
    log(str(ranker_parsed))
    log("\nRANKER RAW OUTPUT:")
    log(ranker_raw[:1200] + ("..." if len(ranker_raw) > 1200 else ""))

    final_ranking = ranker_parsed.get("ranking", [])
    rec = ranker_parsed.get("recommended_symbol")
    notes = ranker_parsed.get("ranker_notes", [])

    # 4) Call Explainer
    log("TEST_LLM: calling explainer...")
    explainer_parsed, explainer_raw = timed(
        "TEST_LLM: explainer returned",
        lambda: run_explainer_llm(
            final_ranking=final_ranking,
            evidence=evidence,
            recommended_symbol=rec,
            ranker_notes=notes,
            model=MODEL
        )
    )

    log("\nEXPLAINER PARSED JSON:")
    log(str(explainer_parsed))
    log("\nEXPLAINER RAW OUTPUT:")
    log(explainer_raw[:1200] + ("..." if len(explainer_raw) > 1200 else ""))

    log("TEST_LLM: done")


if __name__ == "__main__":
    main()