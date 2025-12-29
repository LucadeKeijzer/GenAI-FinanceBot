import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_asset_pipeline
from src.llm import build_evidence_packet, call_ollama_json

WATCHLIST = ["BTC-USD", "ETH-USD", "SPY"]

def main():
    results = {}
    for symbol in WATCHLIST:
        results[symbol] = run_asset_pipeline(symbol, period="2y", forecast_days=14)

    evidence = build_evidence_packet(results)
    parsed, raw = call_ollama_json(evidence, model="llama3.2:1b")

    print("PARSED JSON:")
    print(parsed)
    print("\nRAW OUTPUT:")
    print(raw)

if __name__ == "__main__":
    main()