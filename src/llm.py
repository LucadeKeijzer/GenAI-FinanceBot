import json
import requests
import re

from typing import Dict, Any, Tuple, List, Optional

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

import re

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _remove_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)

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
    symbols = [a["symbol"] for a in evidence.get("assets", []) if a.get("symbol")]

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

    # DATA compact (no pretty indent)
    data_json = json.dumps(evidence, ensure_ascii=True)

    # Keep the schema explicit but short (no example)
    schema = {
        "recommended_symbol": "one of allowed symbols",
        "ranking": ["all allowed symbols exactly once, best to worst"],
        "actions": {"SYMBOL": "one allowed action (must include ALL symbols)"},
        "confidence": "low|medium|high",
        "ranker_notes": ["2-4 short strings using ONLY the provided data"]
    }

    return (
        "You are an educational long-term investing assistant (NOT financial advice).\n"
        "Use ONLY the DATA. Do NOT add external facts.\n"
        "Return ONE valid JSON object only (no markdown, no extra text).\n"
        "All items in ranking and ranker_notes MUST be strings.\n\n"
        f"Allowed symbols: {json.dumps(symbols, ensure_ascii=True)}\n"
        f"Allowed actions: {json.dumps(allowed_actions, ensure_ascii=True)}\n"
        f"Held symbols (qty>0): {json.dumps(held_symbols, ensure_ascii=True)}\n\n"
        "Action rule:\n"
        '- "sell" and "reduce" are ONLY allowed for symbols that are held.\n'
        '- For symbols not held, use consider/add/hold/avoid (NOT sell/reduce).\n\n'
        f"Required JSON schema: {json.dumps(schema, ensure_ascii=True)}\n\n"
        f"DATA: {data_json}"
    )

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
    model: str = "llama3.2:1b",
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
            "num_predict": 160
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

    experience_level = str(user_settings.get("experience_level", "Beginner")).strip() or "Beginner"
    exp_norm = experience_level.lower()

    # Experience-driven explanation depth & numeric expectations
    if exp_norm == "beginner":
        n_bullets = 2
        vocab_rules = (
            "Plain everyday language. No finance jargon (volatility, drawdown, momentum, valuation, diversification)."
        )
        numeric_rules = "No numbers or percentages."
    elif exp_norm == "intermediate":
        n_bullets = 3
        vocab_rules = (
            "Mostly plain language. You may use ONE basic term (volatility OR drawdown) and explain it simply."
        )
        numeric_rules = "At most ONE rounded number, explained in words."
    else:  # Advanced
        n_bullets = 4
        vocab_rules = "You may use analytical terms (volatility, drawdown, trend). Keep it educational and precise."
        numeric_rules = "Include 1–2 rounded numbers where helpful, explained in words."

    evidence_json = json.dumps(evidence, ensure_ascii=True)
    ranking_json = json.dumps(final_ranking, ensure_ascii=True)
    rec_json = json.dumps(recommended_symbol, ensure_ascii=True)
    notes_json = json.dumps(ranker_notes, ensure_ascii=True)

    # Tiny example (structure only) to reduce JSON parse failures
    example = {
        "headline": "<headline mentioning symbol>",
        "explanation": ["<bullet 1>", "<bullet 2>", "<bullet 3>", "<bullet 4>"][:n_bullets],
        "key_tradeoffs": ["One short tradeoff."],
        "risks": ["One short risk.", "One short risk."],
        "disclaimer": "Educational only, not financial advice."
    }
    example_json = json.dumps(example, ensure_ascii=True)

    return (
        "TASK: Explain why the recommended asset is ranked highest, using ONLY the DATA.\n"
        "You are educational only (NOT financial advice).\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- Return EXACTLY ONE valid JSON object.\n"
        "- No markdown, no ``` fences, no extra text before/after JSON.\n"
        "- Use standard JSON double quotes only.\n"
        "- Do not include trailing commas.\n\n"
        f"experience_level={experience_level}\n"
        f"recommended_symbol={rec_json}\n\n"
        "DO NOT RE-RANK:\n"
        "- The ranker already decided the final ranking.\n"
        "- Treat recommended_symbol as final top pick.\n\n"
        "EXPLANATION RULES:\n"
        f"- explanation must be a JSON array with EXACTLY {n_bullets} strings.\n"
        "- Each bullet must be ONE sentence.\n"
        "- Each bullet must focus on a different dimension:\n"
        "  1) expected upside\n"
        "  2) stability / price swings\n"
        "  3) downside risk / worst drops\n"
        "  4) trend / momentum (only if bullet count allows)\n\n"
        "METRIC TRANSLATION (STRICT):\n"
        "- NEVER output raw metric keys anywhere.\n"
        "- Forbidden strings: forecast_change_pct, max_drawdown, trend_strength.\n"
        "- Use natural language:\n"
        "  expected upside signal, stability/price swings, downside risk/worst drop, trend/momentum.\n\n"
        "IMPORTANT:\n"
        "- Do NOT copy any text from the EXAMPLE.\n"
        "- The EXAMPLE is structure-only.\n"
        "- The literal phrase 'One sentence' MUST NOT appear.\n\n"
        f"VOCAB: {vocab_rules}\n"
        f"NUMBERS: {numeric_rules}\n\n"
        f"FINAL_RANKING: {ranking_json}\n"
        f"RANKER_NOTES: {notes_json}\n\n"
        "REQUIRED JSON KEYS:\n"
        'headline (string), explanation (array of strings), key_tradeoffs (array), risks (array), disclaimer (string)\n\n'
        f"EXAMPLE (structure only): {example_json}\n\n"
        f"DATA: {evidence_json}"
    )

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
    model: str = "llama3.2:1b",
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

    exp = str((evidence.get("user_settings") or {})
        .get("experience_level", "Beginner")
        .strip()
        .lower()
    )  
    num_predict = 450 if exp == "advanced" else 350
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": num_predict
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
        raw_clean = _strip_code_fences(raw_clean)
        raw_clean = _remove_trailing_commas(raw_clean)

    except Exception as e:
        fallback = normalize_explainer_output({
            "headline": "(Request to Ollama failed.)",
            "explanation": ["Could not generate explanation due to an error."],
            "key_tradeoffs": [],
            "risks": [str(e)],
            "disclaimer": "Educational only, not financial advice."
        })
        return _with_prompt_chars(fallback, prompt_chars), str(e)

    # 1) best attempt: extract first JSON object
    try:
        parsed = _extract_first_json_object(raw_clean)
        normalized = normalize_explainer_output(parsed)
        normalized["_prompt_chars"] = prompt_chars
        return normalized, raw
    except Exception:
        pass

    # 2) slice from first { to last }
    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw_clean[start:end + 1])
            normalized = normalize_explainer_output(parsed)
            normalized["_prompt_chars"] = prompt_chars
            return normalized, raw
        except Exception:
            pass

    # 3) if it starts with { but missing closing }, try to close it
    if raw_clean.strip().startswith("{") and not raw_clean.strip().endswith("}"):
        try:
            parsed = json.loads(raw_clean.strip() + "\n}")
            normalized = normalize_explainer_output(parsed)
            normalized["_prompt_chars"] = prompt_chars
            return normalized, raw
        except Exception:
            pass

# ----------------------------
# Public wrappers
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