import json
from typing import Dict, Any, Tuple, List

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


def fallback_rank_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    assets = evidence.get("assets", [])

    def score(a: Dict[str, Any]) -> float:
        fc = float(a.get("forecast_change_pct", 0.0))
        vol = float(a.get("volatility", 0.0))
        mdd = float(a.get("max_drawdown", 0.0))
        # Higher forecast_change_pct better; lower volatility better; less-negative drawdown better.
        return fc - 0.25 * vol + 0.15 * mdd

    ranked = sorted(assets, key=score, reverse=True)
    return [a["symbol"] for a in ranked if "symbol" in a]


def normalize_llm_output(parsed: Dict[str, Any], allowed_symbols: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    rec = parsed.get("recommended_symbol")
    ranking = parsed.get("ranking", [])

    # If ranking isn't a proper list of allowed symbols, fall back to evidence ranking
    if not isinstance(ranking, list) or not ranking or any(s not in allowed_symbols for s in ranking):
        ranking = fallback_rank_from_evidence(evidence)

    # Ensure ranking contains all valid symbols exactly once
    ranking = [s for s in ranking if s in allowed_symbols]
    for s in allowed_symbols:
        if s not in ranking:
            ranking.append(s)

    # If rec invalid, default to rank #1
    if rec not in allowed_symbols:
        rec = ranking[0] if ranking else (allowed_symbols[0] if allowed_symbols else None)

    # Force rec to be ranked #1
    if ranking and rec and ranking[0] != rec:
        ranking = [rec] + [s for s in ranking if s != rec]

    parsed["recommended_symbol"] = rec
    parsed["ranking"] = ranking

    # Ensure bullets are strings (not dicts/objects), limit length
    rb = parsed.get("reasoning_bullets", [])
    if not isinstance(rb, list):
        rb = []
    parsed["reasoning_bullets"] = [str(x).strip() for x in rb if str(x).strip()][:5]

    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    parsed["risks"] = [str(x).strip() for x in risks if str(x).strip()][:4]

    # De-duplicate risks vs reasoning
    rb_set = set(parsed["reasoning_bullets"])
    parsed["risks"] = [r for r in parsed["risks"] if r not in rb_set]

    # Ensure at least 2 risks
    if len(parsed["risks"]) < 2:
        parsed["risks"] = [
            "Forecasts are based on historical patterns and may fail if market regimes change.",
            "High volatility can cause large drawdowns even if long-term trends appear positive."
        ]

    if "disclaimer" not in parsed or not isinstance(parsed["disclaimer"], str):
        parsed["disclaimer"] = "Educational only, not financial advice."

    return parsed


def _make_prompt(evidence: Dict[str, Any]) -> str:
    symbols = [a["symbol"] for a in evidence["assets"]]

    example = {
        "recommended_symbol": symbols[0] if symbols else "SPY",
        "ranking": symbols,
        "reasoning_bullets": [
            "SPY ranks above BTC-USD because volatility is lower while forecast_change_pct is comparable.",
            "BTC-USD ranks above ETH-USD due to a higher forecast_change_pct despite higher volatility."
        ],
        "risks": [
            "Forecasts are based on historical patterns and can fail if market conditions change.",
            "High volatility can lead to large drawdowns even in long-term trends."
        ],
        "disclaimer": "Educational only, not financial advice."
    }

    return f"""
You are an educational long-term investing assistant. You are NOT a financial advisor.
Use ONLY the DATA below. Do NOT add external facts or returns not present in DATA.

Return ONLY valid JSON (no markdown, no extra text).
All items in ranking/reasoning_bullets/risks MUST be STRINGS (no objects, no placeholders).

Allowed symbols:
{json.dumps(symbols)}

Required JSON schema:
{{
  "recommended_symbol": "one of allowed symbols",
  "ranking": ["all allowed symbols exactly once, best to worst"],
  "reasoning_bullets": ["3-5 SHORT string bullets comparing assets using ONLY: trend, volatility, max_drawdown, forecast_change_pct"],
  "risks": ["2-4 SHORT string bullets of general limitations (do NOT repeat reasoning_bullets)"],
  "disclaimer": "Educational only, not financial advice."
}}

Example of correct format (follow structure only):
{json.dumps(example, indent=2)}

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
    allowed = [a["symbol"] for a in evidence.get("assets", [])]

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
        parsed = normalize_llm_output(parsed, allowed, evidence)
        return parsed, raw
    except Exception:
        pass

    # Fallback: try to extract JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            parsed = normalize_llm_output(parsed, allowed, evidence)
            return parsed, raw
        except Exception:
            pass

    # Last-resort: close likely-truncated JSON
    if raw.strip().startswith("{") and not raw.strip().endswith("}"):
        try:
            parsed = json.loads(raw.strip() + "\n}")
            parsed = normalize_llm_output(parsed, allowed, evidence)
            return parsed, raw
        except Exception:
            pass

    fallback = {
        "recommended_symbol": None,
        "ranking": fallback_rank_from_evidence(evidence),
        "reasoning_bullets": ["(Model output could not be parsed as JSON.)"],
        "risks": ["LLM output parsing failed; try rerunning or changing the model."],
        "disclaimer": "Educational only, not financial advice."
    }
    fallback = normalize_llm_output(fallback, allowed, evidence)
    return fallback, raw