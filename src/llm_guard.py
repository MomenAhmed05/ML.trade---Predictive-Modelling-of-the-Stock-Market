"""
llm_guard.py  -  LLM AI decision layer for the trading pipeline

Wraps Qwen 3.6 Plus (via OpenRouter, free tier) to provide an advisory
signal-review layer that can be activated with --glm on the pipeline
or the demo runner.

Three hooks are exposed:

  Hook 1 - review_signals()
      Batch-review all non-zero signals for a timestep before they reach
      the PortfolioEngine / AlpacaBridge.  Returns a boolean mask.

  Hook 2 - review_trade()
      Per-trade last-chance veto with optional size adjustment.
      Called inside the order loop just before execution.

  Hook 3 - review_open_positions()
      Scans open positions each bar and recommends early exits.

All hooks are fail-open: if the LLM is unreachable or returns unparseable
output, the original signal passes through unchanged.

Usage:
    # Pipeline (offline backtest):
    python pipeline.py --glm

    # Demo runner (live paper trading):
    export OPENROUTER_API_KEY="sk-or-..."
    python demo_runner.py
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# -- System prompt shared across all hooks -------------------------------------
_SYSTEM_PROMPT = """\
You are a quantitative risk officer reviewing trading signals for a long/short \
US equity strategy. The strategy uses an LSTM + XGBoost ensemble trained on \
hourly bars with regime-adaptive sizing (BULL / BEAR / CRISIS). Your role is \
to apply qualitative judgment on top of the model's quantitative signals. Be \
concise. Always respond ONLY with valid JSON - no markdown, no explanation \
outside the JSON object.\
"""

# Default API key from environment (can also be passed to constructor)
_DEFAULT_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


class GLMGuard:
    """
    Lightweight wrapper around an OpenAI-compatible LLM endpoint for
    trading decision support.

    Parameters
    ----------
    api_key : str
        OpenRouter API key.
    base_url : str
        OpenAI-compatible base URL.
    model : str
        Model identifier on the endpoint.
    timeout : float
        Max seconds to wait for a response before failing open.
    temperature : float
        Sampling temperature (lower = more deterministic decisions).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen3.6-plus-preview:free",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ):
        api_key = api_key or _DEFAULT_API_KEY
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Pass api_key= or set OPENROUTER_API_KEY env var."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required.  pip install openai")

        self._client      = OpenAI(base_url=base_url, api_key=api_key)
        self._model       = model
        self._timeout     = timeout
        self._temperature = temperature

    # --------------------------------------------------------------------------
    # Hook 1 - batch signal review
    # --------------------------------------------------------------------------

    def review_signals(
        self,
        signal_row,
        prob_row,
        tickers: List[str],
        regime: str,
        extra_context: str = "",
        enrichment: Optional[Dict] = None,
    ) -> List[bool]:
        """
        Review all non-zero signals for one timestep in a single LLM call.

        Returns a boolean list parallel to `tickers`:  True = approved,
        False = vetoed.  On any failure returns all-True (fail-open).

        Parameters
        ----------
        enrichment : dict, optional
            Rich context from the pipeline containing per-ticker model
            disagreement, sentiment, technicals, headlines, regime confidence,
            and portfolio signal balance.
        """
        candidates = []
        for i, ticker in enumerate(tickers):
            sig = int(signal_row[i])
            if sig == 0:
                continue
            prob       = float(prob_row[i])
            confidence = prob if sig > 0 else (1.0 - prob)
            entry = {
                "ticker":     ticker,
                "direction":  "LONG" if sig > 0 else "SHORT",
                "confidence": round(confidence, 3),
            }
            # Attach enrichment data if available
            if enrichment and "per_ticker" in enrichment:
                td = enrichment["per_ticker"].get(ticker, {})
                if td.get("model_agreement"):
                    ma = td["model_agreement"]
                    entry["model_agreement"] = (
                        f"{ma.get('votes', '?')}/3 "
                        f"(LSTM: {ma.get('lstm', '?')}, "
                        f"XGB: {ma.get('xgb', '?')}, "
                        f"RF: {ma.get('rf', '?')})"
                    )
                if td.get("sentiment") and td["sentiment"].get("score", 0) != 0:
                    entry["sentiment"] = td["sentiment"]
                if td.get("technicals"):
                    entry["technicals"] = td["technicals"]
                if td.get("headlines"):
                    entry["headlines"] = td["headlines"][:3]  # cap at 3 for prompt length
            candidates.append(entry)

        if not candidates:
            return [False] * len(tickers)

        # Build prompt header with regime and portfolio context
        prompt = f"Current regime: {regime}"
        if enrichment and "regime_confidence" in enrichment:
            rc = enrichment["regime_confidence"]
            prompt += (
                f" (confidence: BULL={rc.get('BULL', 0):.0%}, "
                f"BEAR={rc.get('BEAR', 0):.0%}, "
                f"CRISIS={rc.get('CRISIS', 0):.0%})"
            )
        prompt += "\n"

        if enrichment and "signal_balance" in enrichment:
            sb = enrichment["signal_balance"]
            prompt += f"Signal balance this bar: {sb['long']} LONG, {sb['short']} SHORT\n"

        if extra_context:
            prompt += f"Additional context: {extra_context}\n"

        prompt += (
            f"\nThe model generated {len(candidates)} signal(s) this bar:\n"
            f"{json.dumps(candidates, indent=2)}\n\n"
            "For each ticker, decide whether to APPROVE or VETO.\n"
            "Consider:\n"
            "  - Model consensus: do all 3 models agree, or is one dissenting strongly?\n"
            "  - Headline alignment: does the news support or contradict the signal direction?\n"
            "  - Technical confirmation: do RSI, Bollinger Band position, and MACD histogram\n"
            "    confirm or diverge from the signal?\n"
            "  - Regime confidence: is the regime label reliable or is it a borderline call?\n"
            "  - Portfolio balance: are we over-concentrated in one direction?\n\n"
            "Respond with ONLY this JSON:\n"
            '{"decisions": [{"ticker": "AAPL", "approved": true, "reason": "..."}]}'
        )

        response = self._call(prompt)
        if response is None:
            return [int(signal_row[i]) != 0 for i in range(len(tickers))]

        try:
            data      = json.loads(response)
            decisions = {d["ticker"]: d["approved"] for d in data.get("decisions", [])}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"LLM review_signals: parse error ({exc}) - failing open")
            return [int(signal_row[i]) != 0 for i in range(len(tickers))]

        result = []
        for i, ticker in enumerate(tickers):
            sig = int(signal_row[i])
            if sig == 0:
                result.append(False)
            else:
                approved = decisions.get(ticker, True)
                if not approved:
                    logger.info(f"  LLM vetoed {ticker}")
                result.append(bool(approved))

        return result

    # --------------------------------------------------------------------------
    # Hook 2 - per-trade guard
    # --------------------------------------------------------------------------

    def review_trade(
        self,
        ticker: str,
        direction: int,
        confidence: float,
        regime: str,
        notional: float,
        current_price: float,
        open_positions: Optional[Dict] = None,
        enrichment: Optional[Dict] = None,
    ) -> Dict:
        """
        Per-trade veto + optional size adjustment.

        Returns {"approved": bool, "size_multiplier": float, "reason": str}.
        On failure returns approved=True, size_multiplier=1.0 (fail-open).
        """
        side_label = "LONG" if direction > 0 else "SHORT"

        prompt = f"Trade pending:\n"
        prompt += f"  Ticker: {ticker}  Direction: {side_label}  Confidence: {confidence:.3f}\n"
        prompt += f"  Regime: {regime}"
        if enrichment and "regime_confidence" in enrichment:
            rc = enrichment["regime_confidence"]
            prompt += (
                f" (BULL={rc.get('BULL', 0):.0%}, "
                f"BEAR={rc.get('BEAR', 0):.0%}, "
                f"CRISIS={rc.get('CRISIS', 0):.0%})"
            )
        prompt += f"\n  Notional: ${notional:,.0f}  Price: ${current_price:.4f}\n"

        if enrichment and "portfolio" in enrichment:
            pf = enrichment["portfolio"]
            prompt += (
                f"  Portfolio: equity=${pf['equity']:,.0f}  "
                f"drawdown={pf['drawdown_pct']:.1f}%  "
                f"open={pf['open_positions']} ({pf['long_count']}L/{pf['short_count']}S)\n"
            )

        if enrichment and "model_agreement" in enrichment:
            ma = enrichment["model_agreement"]
            prompt += (
                f"  Model agreement: LSTM={ma.get('lstm','?')}  "
                f"XGB={ma.get('xgb','?')}  RF={ma.get('rf','?')}\n"
            )

        if enrichment and "sentiment" in enrichment:
            st = enrichment["sentiment"]
            prompt += (
                f"  Sentiment: score={st.get('score', 0):.3f}  "
                f"magnitude={st.get('magnitude', 0):.3f}\n"
            )

        if enrichment and "technicals" in enrichment and enrichment["technicals"]:
            tc = enrichment["technicals"]
            prompt += (
                f"  Technicals: RSI={tc.get('RSI_14', '?')}  "
                f"BB%B={tc.get('BB_PctB', '?')}  "
                f"MACD_hist={tc.get('MACD_hist', '?')}  "
                f"vol_pctl={tc.get('vol_percentile', '?')}\n"
            )

        if enrichment and "headlines" in enrichment and enrichment["headlines"]:
            prompt += "  Recent headlines:\n"
            for h in enrichment["headlines"][:3]:
                prompt += f"    - {h}\n"

        prompt += (
            "\nApprove this trade? If yes, optionally adjust size_multiplier (0.25-1.0).\n"
            "Consider: model consensus, headline alignment, technical confirmation,\n"
            "regime confidence, portfolio balance, and current drawdown.\n\n"
            "Respond with ONLY this JSON:\n"
            '{"approved": true, "size_multiplier": 1.0, "reason": "..."}'
        )

        fallback = {"approved": True, "size_multiplier": 1.0, "reason": "llm_unavailable"}
        response = self._call(prompt)
        if response is None:
            return fallback

        try:
            data = json.loads(response)
            return {
                "approved":        bool(data.get("approved", True)),
                "size_multiplier": float(max(0.1, min(1.0, data.get("size_multiplier", 1.0)))),
                "reason":          str(data.get("reason", "")),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return fallback

    # --------------------------------------------------------------------------
    # Hook 3 - open position review
    # --------------------------------------------------------------------------

    def review_open_positions(
        self,
        open_positions: Dict,
        regime: str,
        extra_context: str = "",
    ) -> List[str]:
        """
        Review open positions and return tickers to close early.
        Empty list = hold everything.  On failure returns [] (fail-open).
        """
        if not open_positions:
            return []

        summary = []
        for ticker, info in open_positions.items():
            entry = info.get("entry_price", 0)
            stop  = info.get("stop_price", 0)
            summary.append({
                "ticker":     ticker,
                "side":       info.get("side", "?"),
                "bars_held":  info.get("bars_held", 0),
                "entry":      round(entry, 4),
                "stop":       round(stop, 4),
            })

        prompt = f"Current regime: {regime}\n"
        if extra_context:
            prompt += f"Context: {extra_context}\n"
        prompt += (
            f"\nOpen positions:\n{json.dumps(summary, indent=2)}\n\n"
            "Should any be closed early? Consider regime, time held, risk.\n\n"
            "Respond with ONLY this JSON:\n"
            '{"close_early": ["TSLA"], "reasons": {"TSLA": "..."}}'
        )

        response = self._call(prompt)
        if response is None:
            return []

        try:
            data = json.loads(response)
            return [str(t) for t in data.get("close_early", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    # --------------------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------------------

    def _call(self, user_prompt: str) -> Optional[str]:
        """
        Non-streaming LLM call.  Returns full response text, or None on error.
        """
        try:
            start = time.time()
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=1024,
                timeout=self._timeout,
                extra_body={"reasoning": {"enabled": True}},
            )
            elapsed = time.time() - start
            text    = completion.choices[0].message.content or ""
            logger.debug(f"LLM responded in {elapsed:.1f}s ({len(text)} chars)")
            return text.strip() or None
        except Exception as exc:
            logger.warning(f"LLM call failed: {exc}")
            return None
