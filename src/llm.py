import json
from typing import Dict, Any, Tuple, List, Optional

import requests


# ----------------------------
# Small utilities
# ----------------------------

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


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and parse the first valid JSON object from a string.
    Handles cases where the LLM returns multiple JSON objects back-to-back.
    """
    decoder = json.JSONDecoder()

    s = (text or "").strip()
    start = s.find("{")
    while start != -1:
        try:
            obj, _end = decoder.raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = s.find("{", start + 1)

    raise ValueError("No valid JSON object found in LLM output")


def _robust_json_parse(text: str) -> Dict[str, Any]:
    """
    Try multiple deterministic strategies to parse a JSON object from text.
    Used for ranker output (which should be ONE JSON object).
    """
    s = (text or "").strip()

    # 1) direct
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) slice first {...last}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) try adding closing brace (common truncation)
    if s.startswith("{") and not s.endswith("}"):
        try:
            obj = json.loads(s + "\n}")
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError("Could not parse JSON object from model output")


def _with_prompt_chars(out: Dict[str, Any], prompt_chars: int) -> Dict[str, Any]:
    """
    Attach prompt character count for metrics, without breaking if types are odd.
    """
    try:
        out["_prompt_chars"] = int(prompt_chars)
    except Exception:
        out["_prompt_chars"] = 0
    return out


def _forecast_change_pct(last_price: float, forecast_end: float) -> float:
    if last_price == 0:
        return 0.0
    return (forecast_end / last_price) - 1.0


# ----------------------------
# Evidence & deterministic fallback
# ----------------------------

def build_evidence_packet(
    results: Dict[str, Any],
    user_settings: Optional[Dict[str, Any]] = None,
    wallet: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a compact evidence packet from AssetResult objects.
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

    evidence: Dict[str, Any] = {
        "task": "Compare assets for long-term investing (educational).",
        "assets": assets
    }

    # Inject context (optional)
    if user_settings is not None:
        evidence["user_settings"] = user_settings
    if wallet is not None:
        evidence["wallet"] = wallet

    return evidence


def fallback_rank_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    assets = evidence.get("assets", [])

    def score(a: Dict[str, Any]) -> float:
        fc = float(a.get("forecast_change_pct") or 0.0)
        vol = float(a.get("volatility") or 0.0)
        mdd = float(a.get("max_drawdown") or 0.0)
        # Higher forecast_change_pct better; lower volatility better; less-negative drawdown better.
        return fc - 0.25 * vol + 0.15 * mdd

    ranked = sorted(assets, key=score, reverse=True)
    return [a["symbol"] for a in ranked if "symbol" in a]


# ----------------------------
# Ranker (decision only)
# ----------------------------

def _make_ranker_prompt(evidence: Dict[str, Any]) -> str:
    symbols = [a["symbol"] for a in evidence.get("assets", [])]

    wallet_positions = (evidence.get("wallet") or {}).get("positions", [])
    held_symbols: List[str] = []
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


def normalize_ranker_output(
    parsed: Dict[str, Any],
    allowed_symbols: List[str],
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    rec = parsed.get("recommended_symbol")
    ranking = parsed.get("ranking", [])

    # Validate ranking
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

    # Wallet held symbols
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

    # Actions normalization
    actions_in = parsed.get("actions", {})
    actions: Dict[str, str] = {}

    if isinstance(actions_in, dict):
        for sym in allowed_symbols:
            act = actions_in.get(sym)
            if isinstance(act, str):
                actions[sym] = act.strip().lower()

    # Fill defaults for missing/invalid actions
    for sym in allowed_symbols:
        if sym not in actions or actions[sym] not in allowed_actions:
            actions[sym] = "hold" if sym in held else "consider"

    # Hard guardrail: sell/reduce only if held
    for sym in allowed_symbols:
        if actions[sym] in {"sell", "reduce"} and sym not in held:
            actions[sym] = "avoid"

    # Soft consistency: if forecast meaningfully negative, don't consider/add a not-held asset
    asset_by_symbol = {a["symbol"]: a for a in evidence.get("assets", [])}
    for sym in allowed_symbols:
        a = asset_by_symbol.get(sym, {})
        fc = a.get("forecast_change_pct", None)
        try:
            fc = float(fc)
        except Exception:
            continue

        not_held = sym not in held
        if not_held and fc < -0.02 and actions.get(sym) in {"consider", "add"}:
            actions[sym] = "avoid"

    # Confidence normalization
    conf = parsed.get("confidence", "medium")
    if not isinstance(conf, str):
        conf = "medium"
    conf = conf.strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "medium"

    # Notes normalization
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
    model: str = "llama3.2:3b",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: int = 60
) -> Tuple[Dict[str, Any], str]:
    prompt = _make_ranker_prompt(evidence)
    prompt_chars = len(prompt)
    allowed = [a["symbol"] for a in evidence.get("assets", [])]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 350
        }
    }

    raw = ""
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)

        if r.status_code != 200:
            fallback = {
                "recommended_symbol": None,
                "ranking": fallback_rank_from_evidence(evidence),
                "ranker_notes": [f"(Ollama error {r.status_code})"]
            }
            out = normalize_ranker_output(fallback, allowed, evidence)
            return _with_prompt_chars(out, prompt_chars), r.text

        raw = (r.json().get("response", "") or "").strip()
        raw_clean = _sanitize_json_text(raw)

    except Exception as e:
        fallback = {
            "recommended_symbol": None,
            "ranking": fallback_rank_from_evidence(evidence),
            "ranker_notes": ["(Request to Ollama failed.)"]
        }
        out = normalize_ranker_output(fallback, allowed, evidence)
        return _with_prompt_chars(out, prompt_chars), str(e)

    try:
        parsed = _robust_json_parse(raw_clean)
        out = normalize_ranker_output(parsed, allowed, evidence)
        return _with_prompt_chars(out, prompt_chars), raw
    except Exception:
        fallback = {
            "recommended_symbol": None,
            "ranking": fallback_rank_from_evidence(evidence),
            "ranker_notes": ["(Model output could not be parsed as JSON.)"]
        }
        out = normalize_ranker_output(fallback, allowed, evidence)
        return _with_prompt_chars(out, prompt_chars), raw


# ----------------------------
# Explainer (explain only)
# ----------------------------

def _make_explainer_prompt(
    evidence: Dict[str, Any],
    final_ranking: List[str],
    recommended_symbol: str,
    ranker_notes: Optional[List[str]] = None,
) -> str:
    ranker_notes = ranker_notes or []
    user_settings = evidence.get("user_settings", {}) or {}

    # Explanation depth and vocabulary are controlled by experience_level.
    experience_level = str(user_settings.get("experience_level", "Beginner")).strip() or "Beginner"
    exp_norm = experience_level.lower()

    if exp_norm == "beginner":
        n_bullets = 2
        vocab_rules = (
            "- Use plain, everyday language only.\n"
            "- Do NOT use financial jargon: volatility, drawdown, momentum, valuation, diversification.\n"
            "- Do NOT use numbers or percentages.\n"
            "- Focus on intuitive reasons only.\n"
        )
        numeric_rules = "- Do NOT include any numbers.\n"

    elif exp_norm == "intermediate":
        n_bullets = 3
        vocab_rules = (
            "- Use mostly plain language.\n"
            "- You MAY use ONE basic finance term (volatility OR drawdown), explained intuitively.\n"
            "- Avoid heavy jargon.\n"
            "- Do NOT repeat beginner phrasing; add one extra nuance (comparison or limitation).\n"
        )
        numeric_rules = (
            "- You MAY include at most ONE approximate numeric reference.\n"
            "- Numbers must be rounded and explained in words (e.g. 'around 15% higher').\n"
        )

    else:  # Advanced
        n_bullets = 4
        vocab_rules = (
            "- You SHOULD use analytical terms such as volatility, drawdown, and trend.\n"
            "- Keep explanations precise and educational.\n"
        )
        numeric_rules = (
            "- You SHOULD include 1–2 approximate numeric references where helpful.\n"
            "- Numbers must be rounded and explained in words (e.g. 'roughly half the volatility').\n"
        )

    example_expl = [
        f"{recommended_symbol} shows stronger expected upside than alternatives in this window.",
        "Its price movements appear more stable compared to other assets.",
        "Downside risk looks smaller based on historical worst drops.",
        "Trend signals appear more supportive than peers over this period."
    ][:n_bullets]

    example = {
        "headline": f"{recommended_symbol} is the top-ranked long-term pick in this comparison.",
        "explanation": example_expl,
        "key_tradeoffs": [
            "Expected upside versus stability.",
            "Potential growth versus downside risk."
        ],
        "risks": [
            "Historical patterns may not repeat in future markets.",
            "Market conditions can change unexpectedly."
        ],
        "disclaimer": "Educational only, not financial advice."
    }

    return f"""
DATA (JSON):
{json.dumps(evidence, ensure_ascii=True)}

ROLE:
You are an educational long-term investing assistant (NOT financial advice).
Use ONLY the DATA. Do NOT add external facts.

NON-NEGOTIABLE:
- The Ranker already made the FINAL decision.
- Explain WHY the top-ranked asset is ranked highest.
- Treat "{recommended_symbol}" as the final top pick.
- Do NOT re-rank assets.
- Output ONE valid JSON object only (no markdown, no extra text).

USER SETTINGS:
experience_level={experience_level}

HEADLINE RULES:
- The headline MUST name the recommended symbol.
- The headline may remain the same across experience levels.
- Do NOT write phrases like "top pick" or "rank #1".

EXPLANATION RULES (MANDATORY):
- "explanation" MUST be a JSON array of strings.
- Provide EXACTLY {n_bullets} explanation bullets.
- Each bullet MUST focus on a DIFFERENT evidence dimension:
  1) expected upside
  2) stability / price swings
  3) downside risk / worst drops
  4) trend or momentum (if applicable)
- Do NOT restate the same idea using different wording.
- Each bullet must be ONE clear sentence.
- Do NOT include literal "[" or "]" characters.

METRIC TRANSLATION (MANDATORY):
- NEVER print raw metric keys or labels anywhere (including inside sentences).
- Forbidden strings (must not appear): forecast_change_pct, max_drawdown, trend_strength.
- Use natural language terms instead:
  - forecast_change_pct -> expected upside signal (stronger/weaker)
  - volatility -> price swings / stability (more/less stable)
  - max_drawdown -> worst drop / downside risk (bigger/smaller drawdowns)
  - When discussing downside risk, compare worst drops using words like "smaller/larger worst drops" (do NOT say "half the time").
  - trend_strength -> trend / momentum (more/less supportive)
- If you would otherwise write a metric key, rewrite it using the natural language term.

VOCABULARY RULES:
{vocab_rules}

NUMERIC RULES:
{numeric_rules}

FINAL RANKING (already decided):
{json.dumps(final_ranking, ensure_ascii=True)}

RECOMMENDED SYMBOL:
{json.dumps(recommended_symbol, ensure_ascii=True)}

OPTIONAL RANKER NOTES:
{json.dumps(ranker_notes, ensure_ascii=True)}

REQUIRED JSON SCHEMA:
{{
  "headline": "string",
  "explanation": ["string", "..."],
  "key_tradeoffs": ["string", "..."],
  "risks": ["string", "..."],
  "disclaimer": "Educational only, not financial advice."
}}

Example structure (follow format, not wording):
{json.dumps(example, ensure_ascii=True)}
""".strip()


def normalize_explainer_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    def _strip_wrapping_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        return s

    parsed = dict(parsed or {})

    headline = parsed.get("headline", "")
    if not isinstance(headline, str):
        headline = str(headline)
    headline = _strip_wrapping_quotes(headline)
    if not headline.strip():
        headline = "Based on the evidence, the top-ranked asset is the recommended focus in this set."

    explanation = parsed.get("explanation", [])
    if isinstance(explanation, str):
        explanation = [explanation]
    elif not isinstance(explanation, list):
        explanation = []

    explanation = [
        _strip_wrapping_quotes(str(x).strip())
        for x in explanation
        if str(x).strip()
    ][:4]

    # If model collapses multiple bullets into one string, split deterministically.
    if len(explanation) == 1:
        one = explanation[0].strip()
        if one.startswith("[") and one.endswith("]"):
            one = one[1:-1].strip()

        if ". " in one:
            parts = [p.strip() for p in one.split(". ") if p.strip()]
            rebuilt = []
            for p in parts:
                if not p.endswith("."):
                    p = p + "."
                rebuilt.append(_strip_wrapping_quotes(p))
            explanation = rebuilt[:4]

    tradeoffs = parsed.get("key_tradeoffs", [])
    if isinstance(tradeoffs, str):
        tradeoffs = [tradeoffs]
    elif not isinstance(tradeoffs, list):
        tradeoffs = []
    tradeoffs = [
        _strip_wrapping_quotes(str(x).strip())
        for x in tradeoffs
        if str(x).strip()
    ][:3]

    risks = parsed.get("risks", [])
    if isinstance(risks, str):
        risks = [risks]
    elif not isinstance(risks, list):
        risks = []
    risks = [
        _strip_wrapping_quotes(str(x).strip())
        for x in risks
        if str(x).strip()
    ][:5]

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
    model: str = "llama3.2:3b",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: int = 180
) -> Tuple[Dict[str, Any], str]:
    prompt = _make_explainer_prompt(
        evidence,
        final_ranking,
        recommended_symbol,
        ranker_notes=ranker_notes
    )
    prompt_chars = len(prompt)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 350
        }
    }

    raw = ""
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
            return _with_prompt_chars(fallback, prompt_chars), r.text

        raw = (r.json().get("response", "") or "").strip()
        raw_clean = _sanitize_json_text(raw)

    except Exception as e:
        fallback = normalize_explainer_output({
            "headline": "(Request to Ollama failed.)",
            "explanation": ["Could not generate explanation due to an error."],
            "key_tradeoffs": [],
            "risks": [str(e)],
            "disclaimer": "Educational only, not financial advice."
        })
        return _with_prompt_chars(fallback, prompt_chars), str(e)

    try:
        parsed = _extract_first_json_object(raw_clean)
        normalized = normalize_explainer_output(parsed)
        return _with_prompt_chars(normalized, prompt_chars), raw
    except Exception:
        fallback = normalize_explainer_output({
            "headline": "[Fallback] Explanation could not be generated reliably",
            "explanation": ["LLM output parsing failed; try rerunning or changing the model."],
            "key_tradeoffs": [],
            "risks": [],
            "disclaimer": "Educational only, not financial advice."
        })
        return _with_prompt_chars(fallback, prompt_chars), raw


# ----------------------------
# Public wrappers
# ----------------------------

def run_ranker_llm(
    evidence: Dict[str, Any],
    model: str = "llama3.2:3b",
) -> Tuple[Dict[str, Any], str]:
    return call_ollama_ranker_json(evidence, model=model)


def run_explainer_llm(
    final_ranking: List[str],
    evidence: Dict[str, Any],
    model: str = "llama3.2:3b",
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