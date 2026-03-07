"""
LLM Orchestrator — reasoning layer that synthesises signals from all agents.
Uses a local rule-based engine by default (no API key / no model download needed).
Optional: Ollama (local Llama 3) or OpenAI-compatible API if configured.
Outputs a structured JSON decision with a natural-language explanation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based reasoner (default — zero dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_decision(context: dict) -> dict:
    """
    Weighted voting across all agent signals.
    context keys: technical, sentiment, rl, regime, risk_ok, confidence_scores
    Returns: {direction, confidence, action, explanation}
    """
    votes: dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

    # Technical agent (weight 2.0)
    ta = context.get("technical", {})
    ta_sig = ta.get("signal", 0)
    ta_conf = ta.get("confidence", 0.5)
    if ta_sig == 1:
        votes["buy"] += 2.0 * ta_conf
    elif ta_sig == -1:
        votes["sell"] += 2.0 * ta_conf
    else:
        votes["hold"] += 2.0

    # Sentiment agent (weight 1.5)
    sent_score = context.get("sentiment_score", 0.0)
    sent_threshold = context.get("sentiment_threshold", 0.3)
    if sent_score > sent_threshold:
        votes["buy"] += 1.5 * abs(sent_score)
    elif sent_score < -sent_threshold:
        votes["sell"] += 1.5 * abs(sent_score)
    else:
        votes["hold"] += 0.5

    # RL agent (weight 2.5 — highest weight)
    rl = context.get("rl", {})
    rl_action = rl.get("action", 0)   # 0=hold, 1=buy, 2=sell
    rl_conf = rl.get("confidence", 0.5)
    if rl_action == 1:
        votes["buy"] += 2.5 * rl_conf
    elif rl_action == 2:
        votes["sell"] += 2.5 * rl_conf
    else:
        votes["hold"] += 1.0

    # Correlation agent (veto or reduce)
    corr = context.get("correlation", {})
    corr_allowed = corr.get("allowed", True)
    if not corr_allowed:
        votes["buy"] = 0.0
        votes["sell"] = 0.0
        votes["hold"] = 10.0

    # Risk check (hard veto)
    if not context.get("risk_ok", True):
        votes["buy"] = 0.0
        votes["sell"] = 0.0
        votes["hold"] = 10.0

    # Regime modulation
    regime = context.get("regime", "unknown")
    if regime in ("volatile", "quiet"):
        votes["buy"] *= 0.5
        votes["sell"] *= 0.5

    # Final decision
    best = max(votes, key=votes.__getitem__)
    total = sum(votes.values()) + 1e-8
    confidence = votes[best] / total

    direction = {"buy": 1, "sell": -1, "hold": 0}[best]
    action = best

    # Build explanation
    reasons = []
    if ta_sig != 0:
        reasons.extend(ta.get("reasons", [])[:3])
    if abs(sent_score) > sent_threshold:
        reasons.append(f"Sentiment: {sent_score:+.2f} ({'bullish' if sent_score > 0 else 'bearish'})")
    if rl_action in (1, 2):
        reasons.append(f"RL agent: {'BUY' if rl_action == 1 else 'SELL'} (conf={rl_conf:.0%})")
    if regime not in ("unknown",):
        reasons.append(f"Regime: {regime}")

    explanation = (
        f"Decision: {action.upper()} | Confidence: {confidence:.0%}\n"
        + " | ".join(reasons[:4]) if reasons else f"Decision: {action.upper()}"
    )

    return {
        "direction": direction,
        "action": action,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "vote_breakdown": {k: round(v, 3) for k, v in votes.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional Ollama backend (local Llama 3 — no API key)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_decision(context: dict, model: str = "llama3", host: str = "http://localhost:11434") -> dict | None:
    """
    Use a local Ollama model to reason about the trade context.
    Falls back to rule-based if Ollama is not running.
    """
    try:
        import requests
        prompt = _build_prompt(context)
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        raw = resp.json().get("response", "")
        # Parse JSON from LLM response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Normalise
            action = str(result.get("action", "hold")).lower()
            confidence = float(result.get("confidence", 0.5))
            direction = {"buy": 1, "sell": -1, "hold": 0}.get(action, 0)
            return {
                "direction": direction,
                "action": action,
                "confidence": round(min(confidence, 0.99), 3),
                "explanation": result.get("explanation", "LLM decision"),
                "vote_breakdown": {},
            }
    except Exception as e:
        logger.debug("Ollama unavailable: %s", e)
    return None


def _build_prompt(context: dict) -> str:
    ta = context.get("technical", {})
    return f"""You are a professional Forex trading AI. Analyse this signal context and output a JSON decision.

TECHNICAL: signal={ta.get('signal',0)}, confidence={ta.get('confidence',0):.0%}, reasons={ta.get('reasons',[])}
SENTIMENT: score={context.get('sentiment_score',0):.2f}
RL_AGENT: action={context.get('rl',{}).get('action',0)}, confidence={context.get('rl',{}).get('confidence',0):.0%}
REGIME: {context.get('regime','unknown')}
RISK_OK: {context.get('risk_ok',True)}
PAIR: {context.get('pair','EURUSD')}

Respond ONLY with valid JSON:
{{"action": "buy|sell|hold", "confidence": 0.0-1.0, "explanation": "brief reason"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator class
# ─────────────────────────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Synthesises signals from Technical, Sentiment, RL, and Correlation agents
    into a final trade decision.

    Backend priority:
      1. Ollama (local Llama 3) — if running
      2. Rule-based engine — always available, zero dependencies
    """

    def __init__(
        self,
        backend: str = "auto",   # "auto" | "rules" | "ollama"
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        sentiment_threshold: float = 0.3,
    ):
        self.backend = backend
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.sentiment_threshold = sentiment_threshold
        self._ollama_available: bool | None = None

    def _check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import requests
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def decide(
        self,
        pair: str,
        technical_signal: dict,
        sentiment_score: float,
        rl_signal: dict,
        correlation_result: dict,
        risk_ok: bool,
        regime: str = "unknown",
    ) -> dict:
        """
        Produce final trade decision from all agent inputs.

        Returns:
            {
              "pair":        str,
              "direction":   +1 | -1 | 0,
              "action":      "buy" | "sell" | "hold",
              "confidence":  float,
              "explanation": str,
            }
        """
        context = {
            "pair": pair,
            "technical": technical_signal,
            "sentiment_score": sentiment_score,
            "sentiment_threshold": self.sentiment_threshold,
            "rl": rl_signal,
            "correlation": correlation_result,
            "risk_ok": risk_ok,
            "regime": regime,
        }

        result = None

        if self.backend in ("auto", "ollama") and self._check_ollama():
            result = _ollama_decision(context, self.ollama_model, self.ollama_host)
            if result:
                logger.debug("LLM decision via Ollama: %s", result["action"])

        if result is None:
            result = _rule_based_decision(context)
            logger.debug("LLM decision via rules: %s", result["action"])

        result["pair"] = pair
        return result

    def batch_decide(self, signals_by_pair: dict[str, dict]) -> list[dict]:
        """Run decide() for multiple pairs simultaneously."""
        results = []
        for pair, ctx in signals_by_pair.items():
            try:
                r = self.decide(
                    pair=pair,
                    technical_signal=ctx.get("technical", {}),
                    sentiment_score=ctx.get("sentiment_score", 0.0),
                    rl_signal=ctx.get("rl", {}),
                    correlation_result=ctx.get("correlation", {"allowed": True}),
                    risk_ok=ctx.get("risk_ok", True),
                    regime=ctx.get("regime", "unknown"),
                )
                results.append(r)
            except Exception as e:
                logger.error("LLM decide error for %s: %s", pair, e)
        return results
