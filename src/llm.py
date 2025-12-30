import json
from typing import Dict, Any, Tuple, List, Optional

import requests


def _sanitize_json_text(raw: str) -> str:
    """
    Local LLMs sometimes emit smart quotes or odd unicode quotes that break JSON.
    Convert common variants to standard ASCII quotes before json.loads().
    """
    if not isinstance(raw, str):
        raw = str(raw)

    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
        "…": "...",
    }
    for k, v in replacements.items():
        raw = raw.replace(k, v)

    return raw


def _forecast_change_pct(last_price: float, forecast_end: float) -> float:
    if last_price == 0:
        return 0.0
    return (forecast_end / last_price) - 1.0


def build_evidence_packet(
    results: Dict[str, Any],
    user_settings: Optional[Dict[str, Any]] = None,
    wallet: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
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

    evidence = {
        "task": "Compare assets for long-term investing (educational).",
        "assets": assets
    }

    # v0.3 Step 4: inject context (optional)
    if user_settings is not None:
        evidence["user_settings"] = user_settings

    if wallet is not None:
        evidence["wallet"] = wallet

    return evidence

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


# ----------------------------
# v0.2 Ranker (decision only)
# ----------------------------

def _make_ranker_prompt(evidence: Dict[str, Any]) -> str:
    symbols = [a["symbol"] for a in evidence.get("assets", [])]

    wallet_positions = (evidence.get("wallet") or {}).get("positions", [])
    held_symbols = []
    for p in wallet_positions:
        try:
            sym = str(p.get("symbol", "")).strip()
            qty = float(p.get("quantity", 0))
        except Exception:
            continue
        if sym and qty > 0:
            held_symbols.append(sym)

    allowed_actions = ["consider", "add", "hold", "reduce", "sell", "avoid"]

    example_actions = {s: ("hold" if s in held_symbols else "consider") for s in symbols}
    if symbols:
        # Example: top pick usually "add" (or "hold" if already held)
        example_actions[symbols[0]] = "hold" if symbols[0] in held_symbols else "add"

    example = {
        "recommended_symbol": symbols[0] if symbols else "SPY",
        "ranking": symbols,
        "actions": example_actions,
        "confidence": "medium",
        "ranker_notes": [
            "Top pick has the strongest forecast_change_pct in this dataset.",
            "Second is more stable (lower volatility) but shows less upside signal here.",
            "Third has worse drawdown/volatility tradeoffs in this window."
        ]
    }

    return f"""
You are an educational long-term investing assistant. You are NOT a financial advisor.
Use ONLY the DATA below. Do NOT add external facts.

Return ONLY valid JSON (no markdown, no extra text).
Use only standard JSON double quotes " (ASCII). Do NOT use smart quotes like “ ” or „.
All items in ranking and ranker_notes MUST be STRINGS.

Allowed symbols:
{json.dumps(symbols)}

Allowed actions:
{json.dumps(allowed_actions)}

Wallet held symbols (quantity > 0):
{json.dumps(held_symbols)}

Action rule:
- "sell" and "reduce" are ONLY allowed for symbols that are held in the wallet.
- For symbols not held, use: consider / add / hold / avoid (NOT sell/reduce).

Required JSON schema:
{{
  "recommended_symbol": "one of allowed symbols",
  "ranking": ["all allowed symbols exactly once, best to worst"],
  "actions": {{"SYMBOL": "one allowed action" (must include ALL symbols)}},
  "confidence": "low|medium|high",
  "ranker_notes": ["2-4 SHORT string bullets that justify ranking/actions using ONLY: trend, volatility, max_drawdown, forecast_change_pct, plus wallet context"]
}}

Example of correct format (follow structure only):
{json.dumps(example, indent=2)}

DATA:
{json.dumps(evidence, indent=2)}
""".strip()

def normalize_ranker_output(parsed: Dict[str, Any], allowed_symbols: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    rec = parsed.get("recommended_symbol")
    ranking = parsed.get("ranking", [])

    # --- validate ranking (existing v0.2 logic) ---
    valid = isinstance(ranking, list) and ranking and all(isinstance(s, str) for s in ranking)
    valid = valid and all(s in allowed_symbols for s in ranking)
    valid = valid and len(set(ranking)) == len(ranking)
    valid = valid and set(ranking) == set(allowed_symbols)
    valid = valid and isinstance(rec, str) and rec in allowed_symbols and ranking and ranking[0] == rec

    used_fallback = False
    if not valid:
        used_fallback = True
        ranking = fallback_rank_from_evidence(evidence)
        ranking = [s for s in ranking if s in allowed_symbols]
        for s in allowed_symbols:
            if s not in ranking:
                ranking.append(s)
        rec = ranking[0] if ranking else (allowed_symbols[0] if allowed_symbols else None)

    # --- wallet held symbols (for action guardrail) ---
    wallet_positions = (evidence.get("wallet") or {}).get("positions", [])
    held = set()
    for p in wallet_positions:
        try:
            sym = str(p.get("symbol", "")).strip()
            qty = float(p.get("quantity", 0))
        except Exception:
            continue
        if sym and qty > 0:
            held.add(sym)

    allowed_actions = {"consider", "add", "hold", "reduce", "sell", "avoid"}

    # --- actions normalization (new in v0.3 Step 5) ---
    actions_in = parsed.get("actions", {})
    actions: Dict[str, str] = {}

    if isinstance(actions_in, dict):
        for sym in allowed_symbols:
            act = actions_in.get(sym)
            if isinstance(act, str):
                act = act.strip().lower()
                actions[sym] = act

    # fill defaults for missing/invalid actions
    for sym in allowed_symbols:
        if sym not in actions or actions[sym] not in allowed_actions:
            actions[sym] = "hold" if sym in held else "consider"

    # hard guardrail: sell/reduce only if held
    for sym in allowed_symbols:
        if actions[sym] in {"sell", "reduce"} and sym not in held:
            actions[sym] = "avoid"

    # --- confidence normalization ---
    conf = parsed.get("confidence", "medium")
    if not isinstance(conf, str):
        conf = "medium"
    conf = conf.strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "medium"

    # --- notes normalization (existing) ---
    if used_fallback:
        notes = [
            "The model output was structurally invalid, so a deterministic fallback ranking was used.",
            "Fallback ranking is based on forecast_change_pct, volatility, and max_drawdown from the evidence.",
            "Actions were normalized using wallet holdings (sell/reduce only allowed when held)."
        ]
    else:
        notes = parsed.get("ranker_notes", [])
        if not isinstance(notes, list):
            notes = []
        notes = [str(x).strip() for x in notes if str(x).strip()][:4]
        if len(notes) < 2:
            notes = [
                "Ranking/actions are based on trend, volatility, max_drawdown, and forecast_change_pct in the evidence.",
                "Sell/reduce actions are only valid for assets held in the wallet (guardrail enforced)."
            ]

    return {
        "recommended_symbol": rec,
        "ranking": ranking,
        "actions": actions,
        "confidence": conf,
        "ranker_notes": notes
    }

def call_ollama_ranker_json(
    evidence: Dict[str, Any],
    model: str = "llama3.2:1b",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: int = 60
) -> Tuple[Dict[str, Any], str]:
    prompt = _make_ranker_prompt(evidence)
    allowed = [a["symbol"] for a in evidence.get("assets", [])]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 240
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            fallback = {
                "recommended_symbol": None,
                "ranking": fallback_rank_from_evidence(evidence),
                "ranker_notes": [f"(Ollama error {r.status_code})"]
            }
            return normalize_ranker_output(fallback, allowed, evidence), r.text

        raw = r.json().get("response", "").strip()
        raw_clean = _sanitize_json_text(raw)

    except Exception as e:
        fallback = {
            "recommended_symbol": None,
            "ranking": fallback_rank_from_evidence(evidence),
            "ranker_notes": ["(Request to Ollama failed.)"]
        }
        return normalize_ranker_output(fallback, allowed, evidence), str(e)

    try:
        parsed = json.loads(raw_clean)
        return normalize_ranker_output(parsed, allowed, evidence), raw
    except Exception:
        pass

    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw_clean[start:end + 1])
            return normalize_ranker_output(parsed, allowed, evidence), raw
        except Exception:
            pass

    if raw_clean.strip().startswith("{") and not raw_clean.strip().endswith("}"):
        try:
            parsed = json.loads(raw_clean.strip() + "\n}")
            return normalize_ranker_output(parsed, allowed, evidence), raw
        except Exception:
            pass

    fallback = {
        "recommended_symbol": None,
        "ranking": fallback_rank_from_evidence(evidence),
        "ranker_notes": ["(Model output could not be parsed as JSON.)"]
    }
    return normalize_ranker_output(fallback, allowed, evidence), raw


# ----------------------------
# v0.2 Explainer (explain only)
# ----------------------------

def _make_explainer_prompt(
    evidence: Dict[str, Any],
    final_ranking: List[str],
    recommended_symbol: str,
    ranker_notes: Optional[List[str]] = None,
) -> str:
    symbols = [a["symbol"] for a in evidence.get("assets", [])]
    ranker_notes = ranker_notes or []

    example = {
        "headline": "Based on the evidence, SPY is the top long-term candidate in this set.",
        "explanation": [
            "SPY ranks first mainly because its forecast_change_pct is strongest in this dataset while volatility is relatively lower.",
            "BTC-USD ranks second due to stronger upside signals but higher volatility and drawdown risk in this window.",
            "ETH-USD ranks third given the tradeoffs shown in the evidence packet."
        ],
        "key_tradeoffs": [
            "Upside signal (forecast_change_pct) vs stability (volatility).",
            "Max drawdown highlights potential pain during market stress.",
            "Trend labels depend on the chosen time window."
        ],
        "risks": [
            "Forecasts are based on historical patterns and may fail in new market regimes.",
            "High volatility can cause drawdowns that take years to recover from.",
            "A limited historical window may overweight recent conditions."
        ],
        "disclaimer": "Educational only, not financial advice."
    }

    return f"""
You are an educational long-term investing assistant. You are NOT a financial advisor.
Use ONLY the DATA below. Do NOT add external facts.

You MUST explain the FINAL ranking provided. You are NOT allowed to change the ranking.
Do NOT output a new ranking. Do NOT recommend a different symbol.

Return ONLY valid JSON (no markdown, no extra text).
Use only standard JSON double quotes " (ASCII). Do NOT use smart quotes like “ ” or „.
All list items MUST be STRINGS.

Allowed symbols:
{json.dumps(symbols)}

FINAL RANKING (must be explained exactly):
{json.dumps(final_ranking)}

RECOMMENDED SYMBOL (must match rank #1):
{json.dumps(recommended_symbol)}

Optional ranker notes (may help keep consistency):
{json.dumps(ranker_notes)}

Required JSON schema:
{{
  "headline": "1 sentence",
  "explanation": ["2-4 short bullets/paragraphs explaining WHY the final ranking makes sense using ONLY: trend, volatility, max_drawdown, forecast_change_pct"],
  "key_tradeoffs": ["3 short bullets (comparisons/tradeoffs)"],
  "risks": ["3-5 short bullets (general limitations, do NOT repeat explanation)"],
  "disclaimer": "Educational only, not financial advice."
}}

Example of correct format (follow structure only):
{json.dumps(example, indent=2)}

DATA:
{json.dumps(evidence, indent=2)}
""".strip()


def normalize_explainer_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    def _strip_wrapping_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        return s

    headline = parsed.get("headline", "")
    if not isinstance(headline, str):
        headline = str(headline)
    headline = _strip_wrapping_quotes(headline)

    explanation = parsed.get("explanation", [])
    if not isinstance(explanation, list):
        explanation = []
    explanation = [_strip_wrapping_quotes(str(x).strip()) for x in explanation if str(x).strip()][:4]

    tradeoffs = parsed.get("key_tradeoffs", [])
    if not isinstance(tradeoffs, list):
        tradeoffs = []
    tradeoffs = [_strip_wrapping_quotes(str(x).strip()) for x in tradeoffs if str(x).strip()][:3]

    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    risks = [_strip_wrapping_quotes(str(x).strip()) for x in risks if str(x).strip()][:5]

    used = set(explanation) | set(tradeoffs)
    risks = [r for r in risks if r not in used]

    if len(risks) < 3:
        risks = [
            "Forecasts are based on historical patterns and may fail if market regimes change.",
            "Volatile assets can experience large drawdowns that take years to recover from.",
            "This comparison depends on the chosen time window and limited metrics."
        ]

    disclaimer = parsed.get("disclaimer", "")
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        disclaimer = "Educational only, not financial advice."

    if not headline.strip():
        headline = "Based on the evidence, the top-ranked asset is the recommended focus in this set."

    return {
        "headline": headline.strip(),
        "explanation": explanation,
        "key_tradeoffs": tradeoffs,
        "risks": risks,
        "disclaimer": disclaimer.strip()
    }


def call_ollama_explainer_json(
    evidence: Dict[str, Any],
    final_ranking: List[str],
    recommended_symbol: str,
    ranker_notes: Optional[List[str]] = None,
    model: str = "llama3.2:1b",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: int = 60
) -> Tuple[Dict[str, Any], str]:
    prompt = _make_explainer_prompt(evidence, final_ranking, recommended_symbol, ranker_notes=ranker_notes)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_ctx": 1024,
            "num_predict": 380
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            fallback = normalize_explainer_output({
                "headline": f"(Ollama error {r.status_code})",
                "explanation": ["Could not generate explanation from local model."],
                "key_tradeoffs": [],
                "risks": ["Local model call failed; try rerunning or changing the model."],
                "disclaimer": "Educational only, not financial advice."
            })
            return fallback, r.text

        raw = r.json().get("response", "").strip()
        raw_clean = _sanitize_json_text(raw)

    except Exception as e:
        fallback = normalize_explainer_output({
            "headline": "(Request to Ollama failed.)",
            "explanation": ["Could not generate explanation due to an error."],
            "key_tradeoffs": [],
            "risks": [str(e)],
            "disclaimer": "Educational only, not financial advice."
        })
        return fallback, str(e)

    try:
        parsed = json.loads(raw_clean)
        return normalize_explainer_output(parsed), raw
    except Exception:
        pass

    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw_clean[start:end + 1])
            return normalize_explainer_output(parsed), raw
        except Exception:
            pass

    if raw_clean.strip().startswith("{") and not raw_clean.strip().endswith("}"):
        try:
            parsed = json.loads(raw_clean.strip() + "\n}")
            return normalize_explainer_output(parsed), raw
        except Exception:
            pass

    fallback = normalize_explainer_output({
        "headline": "(Model output could not be parsed as JSON.)",
        "explanation": ["LLM output parsing failed; try rerunning or changing the model."],
        "key_tradeoffs": [],
        "risks": [],
        "disclaimer": "Educational only, not financial advice."
    })
    return fallback, raw


# ----------------------------
# Public v0.2 wrappers
# ----------------------------

def run_ranker_llm(
    evidence: Dict[str, Any],
    model: str = "llama3.2:1b",
) -> Tuple[Dict[str, Any], str]:
    return call_ollama_ranker_json(evidence, model=model)


def run_explainer_llm(
    final_ranking: List[str],
    evidence: Dict[str, Any],
    model: str = "llama3.2:1b",
    recommended_symbol: Optional[str] = None,
    ranker_notes: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    if recommended_symbol is None:
        recommended_symbol = final_ranking[0] if final_ranking else ""
    return call_ollama_explainer_json(
        evidence=evidence,
        final_ranking=final_ranking,
        recommended_symbol=recommended_symbol,
        ranker_notes=ranker_notes,
        model=model
    )


# ----------------------------
# Legacy v0.1 one-shot call (kept temporarily)
# ----------------------------

def normalize_llm_output(parsed: Dict[str, Any], allowed_symbols: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    rec = parsed.get("recommended_symbol")
    ranking = parsed.get("ranking", [])

    if not isinstance(ranking, list) or not ranking or any(s not in allowed_symbols for s in ranking):
        ranking = fallback_rank_from_evidence(evidence)

    ranking = [s for s in ranking if s in allowed_symbols]
    for s in allowed_symbols:
        if s not in ranking:
            ranking.append(s)

    if rec not in allowed_symbols:
        rec = ranking[0] if ranking else (allowed_symbols[0] if allowed_symbols else None)

    if ranking and rec and ranking[0] != rec:
        ranking = [rec] + [s for s in ranking if s != rec]

    parsed["recommended_symbol"] = rec
    parsed["ranking"] = ranking

    rb = parsed.get("reasoning_bullets", [])
    if not isinstance(rb, list):
        rb = []
    parsed["reasoning_bullets"] = [str(x).strip() for x in rb if str(x).strip()][:5]

    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    parsed["risks"] = [str(x).strip() for x in risks if str(x).strip()][:4]

    rb_set = set(parsed["reasoning_bullets"])
    parsed["risks"] = [r for r in parsed["risks"] if r not in rb_set]

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
        raw_clean = _sanitize_json_text(raw)

    except Exception as e:
        fallback = {
            "recommended_symbol": None,
            "ranking": [],
            "reasoning_bullets": ["(Request to Ollama failed.)"],
            "risks": [str(e)],
            "disclaimer": "Educational only, not financial advice."
        }
        return fallback, str(e)

    try:
        parsed = json.loads(raw_clean)
        parsed = normalize_llm_output(parsed, allowed, evidence)
        return parsed, raw
    except Exception:
        pass

    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw_clean[start:end + 1])
            parsed = normalize_llm_output(parsed, allowed, evidence)
            return parsed, raw
        except Exception:
            pass

    if raw_clean.strip().startswith("{") and not raw_clean.strip().endswith("}"):
        try:
            parsed = json.loads(raw_clean.strip() + "\n}")
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