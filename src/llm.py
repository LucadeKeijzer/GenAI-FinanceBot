import json
from typing import Dict, Any, Tuple

import requests


def _forecast_change_pct(last_price: float, forecast_end: float) -> float:
    if last_price == 0:
        return 0.0
    return (forecast_end / last_price) - 1.0


def build_evidence_packet(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, comparable evidence packet from AssetResult objects.
    Keeps the payload small for light local models.
    """
    assets = []
    for symbol, r in results.items():
        last_price = float(r.metrics["last_price"])
        forecast_end = float(r.forecast.iloc[-1])

        assets.append({
            "symbol": symbol,
            "trend": r.metrics["trend"],
            "volatility": round(float(r.metrics["volatility"]), 4),
            "max_drawdown": round(float(r.metrics["max_drawdown"]), 4),
            "forecast_change_pct": round(_forecast_change_pct(last_price, forecast_end), 4),
        })

    return {
        "task": "Compare assets for long-term investing (educational).",
        "assets": assets
    }


def _make_prompt(evidence: Dict[str, Any]) -> str:
    symbols = [a["symbol"] for a in evidence["assets"]]

    return f"""
You are an educational long-term investing assistant. You are NOT a financial advisor.
Use ONLY the DATA below. Do NOT add any external facts (no historical averages, no market claims).
Do NOT mention returns that are not in DATA.

Return ONLY valid JSON. No markdown. No extra text.

Allowed symbols (use ONLY these exact strings):
{json.dumps(symbols)}

JSON schema (follow exactly):
{{
  "recommended_symbol": "one of the allowed symbols",
  "ranking": ["all allowed symbols, each exactly once, best to worst"],
  "reasoning_bullets": ["3-5 short bullets that reference ONLY trend/volatility/max_drawdown/forecast_change_pct"],
  "risks": ["2-4 bullets"],
  "disclaimer": "Educational only, not financial advice."
}}

DATA:
{json.dumps(evidence, indent=2)}
""".strip()


def call_ollama_json(
    evidence: Dict[str, Any],
    model: str = "llama3.2:1b",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: int = 60
) -> Tuple[Dict[str, Any], str]:
    """
    Call Ollama and return (parsed_json, raw_text). If parsing fails or Ollama errors,
    returns a safe fallback object and the raw response text.
    """
    prompt = _make_prompt(evidence)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 300
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)

        if r.status_code != 200:
            fallback = {
                "recommended_symbol": None,
                "ranking": [],
                "reasoning_bullets": [f"(Ollama error {r.status_code})"],
                "risks": ["Could not generate response from local model."],
                "disclaimer": "Educational only, not financial advice."
            }
            return fallback, r.text

        raw = r.json().get("response", "").strip()

    except Exception as e:
        fallback = {
            "recommended_symbol": None,
            "ranking": [],
            "reasoning_bullets": ["(Request to Ollama failed.)"],
            "risks": [str(e)],
            "disclaimer": "Educational only, not financial advice."
        }
        return fallback, str(e)

    # Try strict JSON parse
    try:
        parsed = json.loads(raw)
        return parsed, raw
    except Exception:
        # Fallback: try to extract JSON block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed, raw
            except Exception:
                pass
        # Last-resort: if it looks like JSON but is missing a closing brace, try to close it
        if raw.strip().startswith("{") and not raw.strip().endswith("}"):
            try:
                parsed = json.loads(raw.strip() + "}")
                return parsed, raw
            except Exception:
                pass

    fallback = {
        "recommended_symbol": None,
        "ranking": [],
        "reasoning_bullets": ["(Model output could not be parsed as JSON.)"],
        "risks": ["LLM output parsing failed; try rerunning or changing the model."],
        "disclaimer": "Educational only, not financial advice."
    }
    return fallback, raw