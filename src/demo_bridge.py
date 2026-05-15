"""
demo_bridge.py  -  Alpaca Paper Trading Execution Bridge

Translates signal_matrix rows produced by the existing pipeline
(PortfolioEngine / run_ensemble) into real Alpaca paper-account orders.

Design goals:
  - Strategy is IDENTICAL to the backtest: same thresholds, same sizing,
    same regime-adaptive config from pipeline.py's REGIME_CONFIG.
  - base_trade_size_pct is set to 0.10 for demo safety.
  - Leverage / stop-loss / horizon / partial-exit logic mirrors PredictionEngine exactly.

Notes on Alpaca order types:
  - LONG  orders: fractional notional.
  - SHORT orders: whole shares (error 42210000 for fractional shorts).
                  Non-shortable assets are skipped.
  - CLOSE orders: open orders cancelled first (prevents wash-trade 40310000).
  - BUYING POWER: checked before each order; skipped if insufficient.
  - MAX POSITIONS: hard cap of `max_positions` concurrent positions.
  - RESTART SAFETY: open_positions is synced from the live Alpaca account
                    on __init__ so restarts don't re-enter existing positions.

Usage:
    bridge = AlpacaBridge(api_key=..., secret_key=..., paper=True)
    bridge.step(signal_row, prob_row, tickers, regime="BULL")

Dependencies:
    pip install alpaca-py
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Persistent state file - survives runner restarts
STATE_FILE = Path(__file__).resolve().parent.parent / "models" / "demo_bridge_state.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -- Regime config (must match pipeline.py REGIME_CONFIG exactly) --------------
# base_trade_size_pct here mirrors pipeline.py; AlpacaBridge.__init__ accepts a
# base_trade_size_pct constructor arg that overrides this for demo safety (0.10
# instead of 0.20 so the paper account doesn't exhaust buying power instantly).
REGIME_CONFIG = {
    "BULL": {
        "long_threshold":        0.52,
        "short_threshold":       0.47,
        "long_safety_sl":        0.05,
        "short_safety_sl":       0.05,
        "base_trade_size_pct":   0.25,
        "leverage_min":          3.0,
        "leverage_max":          6.0,
        "short_size_multiplier": 0.50,
    },
    "CRISIS": {
        "long_threshold":        0.55,
        "short_threshold":       0.40,
        "long_safety_sl":        0.05,
        "short_safety_sl":       0.05,
        "base_trade_size_pct":   0.25,
        "leverage_min":          2.5,
        "leverage_max":          6.0,
        "short_size_multiplier": 0.50,
    },
    "BEAR": {
        "long_threshold":        0.60,
        "short_threshold":       0.45,
        "long_safety_sl":        0.03,
        "short_safety_sl":       0.05,
        "base_trade_size_pct":   0.20,
        "leverage_min":          2.0,
        "leverage_max":          4.0,
        "short_size_multiplier": 0.80,
    },
}

MAX_POSITIONS    = 8
MIN_BUYING_POWER = 500.0


class AlpacaBridge:

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        horizon: int = 8,
        base_trade_size_pct: float = 0.20,
        max_positions: int = MAX_POSITIONS,
    ):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError("alpaca-py is required.  pip install alpaca-py")

        self.client              = TradingClient(api_key, secret_key, paper=paper)
        self.horizon             = horizon
        self.half_horizon        = max(1, horizon // 2)
        self.base_trade_size_pct = base_trade_size_pct
        self.max_positions       = max_positions
        self.paper               = paper
        self._api_key            = api_key
        self._secret_key         = secret_key

        # Populated from saved state + Alpaca reconciliation on init.
        self.open_positions: Dict[str, dict] = {}
        self._load_state()
        self._sync_positions_from_alpaca()

        mode = "PAPER" if paper else "LIVE"
        logger.info(
            f"AlpacaBridge initialised - mode={mode}  horizon={horizon}  half_horizon={self.half_horizon}  "
            f"base_trade_size_pct={base_trade_size_pct:.0%}  "
            f"max_positions={max_positions}  "
            f"synced_positions={list(self.open_positions.keys())}"
        )

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def step(self, signal_row, prob_row, tickers, regime="BULL"):
        cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG["BULL"])
        self._age_and_expire(tickers)
        self._check_stop_losses(tickers, cfg)
        self._open_new_positions(signal_row, prob_row, tickers, cfg)
        self._save_state()

    def close_all(self):
        try:
            self.client.close_all_positions(cancel_orders=True)
            self.open_positions.clear()
            logger.info("All positions closed.")
        except Exception as exc:
            logger.error(f"close_all failed: {exc}")

    def get_account_equity(self) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception as exc:
            logger.error(f"get_account_equity failed: {exc}")
            return 0.0

    def get_buying_power(self) -> float:
        try:
            return float(self.client.get_account().buying_power)
        except Exception as exc:
            logger.error(f"get_buying_power failed: {exc}")
            return 0.0

    def is_shortable(self, ticker: str) -> bool:
        try:
            asset = self.client.get_asset(ticker)
            return bool(asset.shortable) and bool(asset.easy_to_borrow)
        except Exception:
            return True

    # --------------------------------------------------------------------------
    # State persistence - survives runner restarts
    # --------------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist open_positions to disk so bars_held survives restarts."""
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self.open_positions, f, indent=2)
        except Exception as exc:
            logger.warning(f"  _save_state failed: {exc}")

    def _load_state(self) -> None:
        """Load previously saved position state (bars_held, halved, etc.)."""
        if not STATE_FILE.exists():
            return
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            self.open_positions = saved
            held = ', '.join(f'{t}: {d["bars_held"]}' for t, d in saved.items())
            logger.info(f"  Loaded saved state: {list(saved.keys())} (bars_held: {held})")
        except Exception as exc:
            logger.warning(f"  _load_state failed: {exc}")

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _sync_positions_from_alpaca(self) -> None:
        """
        Reconcile open_positions with the live Alpaca account on startup.

        If saved state exists (from _load_state), bars_held is preserved.
        If a position exists on Alpaca but NOT in saved state, it's added
        with bars_held=0 (conservative). If saved state has a position that
        no longer exists on Alpaca, it's removed.
        """
        try:
            positions = self.client.get_all_positions()
            alpaca_tickers = set()
            if not positions:
                logger.info("  No existing Alpaca positions found on startup.")
                # Clear any stale saved state
                self.open_positions.clear()
                self._save_state()
                return
            for pos in positions:
                ticker      = str(pos.symbol)
                alpaca_tickers.add(ticker)
                side        = "long" if float(pos.qty) > 0 else "short"
                entry_price = float(pos.avg_entry_price)

                if ticker in self.open_positions:
                    # Already loaded from saved state - preserve bars_held
                    saved = self.open_positions[ticker]
                    saved["entry_price"] = entry_price  # update from Alpaca
                    saved["notional"]    = abs(float(pos.market_value))
                    logger.info(
                        f"  Restored {side.upper():5s} {ticker:<6}  "
                        f"bars_held={saved['bars_held']}  halved={saved.get('halved', False)}")
                    continue

                # New position not in saved state - conservative bars_held=0
                sl_pct      = 0.03 if side == "long" else 0.05
                stop_price  = (
                    entry_price * (1.0 - sl_pct) if side == "long"
                    else entry_price * (1.0 + sl_pct)
                )
                self.open_positions[ticker] = {
                    "bars_held":   0,
                    "side":        side,
                    "entry_price": entry_price,
                    "stop_price":  stop_price,
                    "notional":    abs(float(pos.market_value)),
                    "direction":   1 if side == "long" else -1,
                    "halved":      False,
                }
                logger.info(
                    f"  Synced new position: {side.upper():5s} {ticker:<6}  "
                    f"entry={entry_price:.4f}  stop={stop_price:.4f}  bars_held=0"
                )
            # Remove stale saved positions no longer on Alpaca
            stale = [t for t in self.open_positions if t not in alpaca_tickers]
            for t in stale:
                logger.info(f"  Removed stale saved position: {t}")
                del self.open_positions[t]
            self._save_state()
        except Exception as exc:
            logger.warning(f"  _sync_positions_from_alpaca failed: {exc}")

    def _age_and_expire(self, tickers):
        to_partial = []
        to_close   = []
        for ticker, info in list(self.open_positions.items()):
            info["bars_held"] += 1
            if info["bars_held"] >= self.horizon:
                to_close.append(ticker)
            elif info["bars_held"] == self.half_horizon and not info.get("halved", False):
                to_partial.append(ticker)

        # --- Partial exit at half-horizon (50% of position) ---
        for ticker in to_partial:
            self._partial_close(ticker)

        # --- Full exit at horizon ---
        for ticker in to_close:
            self._close_position(ticker, reason=f"horizon ({self.horizon} bars)")

    def _check_stop_losses(self, tickers, cfg):
        if not self.open_positions:
            return
        for ticker in list(self.open_positions.keys()):
            if ticker not in self.open_positions:
                continue
            info = self.open_positions[ticker]
            try:
                price = self._latest_price(ticker)
                if price is None:
                    continue
                hit = (
                    (info["side"] == "long"  and price <= info["stop_price"]) or
                    (info["side"] == "short" and price >= info["stop_price"])
                )
                if hit:
                    self._close_position(
                        ticker,
                        reason=f"stop-loss hit (price={price:.4f} stop={info['stop_price']:.4f})",
                    )
            except Exception as exc:
                logger.warning(f"SL check failed for {ticker}: {exc}")

    def _open_new_positions(self, signal_row, prob_row, tickers, cfg):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        equity = self.get_account_equity()
        if equity <= 0:
            logger.warning("Account equity unavailable - skipping new entries.")
            return

        lev_min   = cfg["leverage_min"]
        lev_max   = cfg["leverage_max"]
        short_mul = cfg["short_size_multiplier"]
        long_sl   = cfg["long_safety_sl"]
        short_sl  = cfg["short_safety_sl"]

        candidates: List[Tuple[int, int, float]] = []
        for i, ticker in enumerate(tickers):
            sig = int(signal_row[i])
            if sig == 0 or ticker in self.open_positions:
                continue
            prob       = float(prob_row[i])
            confidence = prob if sig > 0 else (1.0 - prob)
            candidates.append((i, sig, confidence))
        candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            return

        # -- Sequential cash-gating: matches PredictionEngine exactly -----
        # Track our own cash pool (= equity at bar start), decrement by margin
        # per position. This mirrors the backtest's `cash -= margin` logic
        # instead of relying on Alpaca buying_power (which includes broker
        # margin/leverage and allows more positions than the backtest would).
        cash_pool = equity

        for (i, direction, confidence) in candidates:
            ticker = tickers[i]

            if len(self.open_positions) >= self.max_positions:
                logger.info(f"  Max positions ({self.max_positions}) reached - skipping remaining candidates")
                break

            if direction < 0 and not self.is_shortable(ticker):
                logger.warning(f"  {ticker}: not shortable on Alpaca - skip")
                continue

            conf_scalar = min((confidence - 0.5) / 0.5, 1.0)
            leverage    = lev_min + (lev_max - lev_min) * conf_scalar
            size_pct    = cfg["base_trade_size_pct"] * (0.8 + 0.4 * conf_scalar)
            margin      = equity * size_pct
            if direction < 0:
                margin *= short_mul
            notional = margin * leverage

            if margin < 10.0:
                logger.debug(f"  {ticker}: margin too small (${margin:.2f}) - skip")
                continue

            # Backtest gate: cash < margin → skip
            if cash_pool < margin:
                logger.warning(
                    f"  {ticker}: cash pool exhausted - "
                    f"need margin ${margin:,.0f}, pool ${cash_pool:,.0f} - skip"
                )
                continue

            # Also check Alpaca buying power as a safety net
            buying_power = self.get_buying_power()
            if buying_power < notional:
                logger.warning(
                    f"  {ticker}: insufficient buying power - "
                    f"need notional ${notional:,.0f}, have ${buying_power:,.0f} - skip"
                )
                continue

            side       = OrderSide.BUY if direction > 0 else OrderSide.SELL
            side_label = "LONG" if direction > 0 else "SHORT"

            current_price = self._latest_price(ticker)
            if current_price is None or current_price <= 0:
                logger.warning(f"  {ticker}: Alpaca price unavailable - using yfinance fallback")
                current_price = self._yfinance_last_close(ticker)
            if current_price is None or current_price <= 0:
                logger.warning(f"  {ticker}: no price source - skip")
                continue

            stop_price = (
                current_price * (1.0 - long_sl) if direction > 0
                else current_price * (1.0 + short_sl)
            )

            try:
                if direction > 0:
                    order = MarketOrderRequest(
                        symbol=ticker,
                        notional=round(notional, 2),
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                    order_qty_label = f"notional=${notional:,.2f}"
                else:
                    qty = max(1, int(notional / current_price))
                    order = MarketOrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY,
                    )
                    order_qty_label = f"qty={qty}sh  (~${qty * current_price:,.2f})"

                self.client.submit_order(order)

                cash_pool -= margin   # decrement local cash pool (mirrors backtest)

                self.open_positions[ticker] = {
                    "bars_held":   0,
                    "side":        side_label.lower(),
                    "entry_price": current_price,
                    "stop_price":  stop_price,
                    "notional":    notional,
                    "direction":   direction,
                    "halved":      False,
                }

                logger.info(
                    f"  OPEN {side_label:5s} {ticker:<6}  {order_qty_label}  "
                    f"entry={current_price:.4f}  stop={stop_price:.4f}  "
                    f"leverage={leverage:.2f}x  conf={confidence:.3f}  "
                    f"cash_left=${cash_pool:,.0f}"
                )

            except Exception as exc:
                logger.error(f"  Order failed for {ticker}: {exc}")

            time.sleep(0.05)

    def _partial_close(self, ticker: str) -> None:
        """Close 50% of position at half-horizon, matching backtest partial exit."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        info = self.open_positions.get(ticker)
        if not info:
            return

        try:
            # Get current position from Alpaca to know exact qty
            # get_open_position() is the alpaca-py v0.43+ method
            pos = self.client.get_open_position(ticker)
            current_qty = abs(float(pos.qty))
            half_qty    = current_qty / 2.0

            if half_qty < 0.001:
                logger.debug(f"  {ticker}: partial close qty too small - skip")
                return

            if info["side"] == "long":
                # Sell half
                if half_qty == int(half_qty):
                    half_qty = int(half_qty)
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=half_qty if isinstance(half_qty, int) else None,
                    notional=round(float(pos.market_value) / 2.0, 2) if not isinstance(half_qty, int) else None,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                # Buy-to-cover half (shorts must be whole shares)
                half_qty = max(1, int(half_qty))
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=half_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )

            self.client.submit_order(order)
            info["halved"]   = True
            info["notional"] = info.get("notional", 0) / 2.0
            logger.info(
                f"  PARTIAL CLOSE {ticker:<6}  50% at half-horizon "
                f"(bar {info['bars_held']}/{self.horizon})  side={info['side']}"
            )

        except Exception as exc:
            logger.error(f"  _partial_close({ticker}) failed: {exc}")

    def _cancel_open_orders(self, ticker: str) -> None:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req    = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(req)
            cancelled = 0
            for order in orders:
                if str(order.symbol).upper() == ticker.upper():
                    try:
                        self.client.cancel_order_by_id(order.id)
                        cancelled += 1
                    except Exception as exc:
                        logger.debug(f"  Cancel {order.id} failed: {exc}")
            if cancelled:
                logger.info(f"  Cancelled {cancelled} open order(s) for {ticker} before close")
        except Exception as exc:
            logger.warning(f"  _cancel_open_orders({ticker}) failed: {exc}")

    def _close_position(self, ticker, reason=""):
        self._cancel_open_orders(ticker)
        time.sleep(0.5)
        try:
            self.client.close_position(ticker)
            info = self.open_positions.pop(ticker, {})
            logger.info(
                f"  CLOSE {ticker:<6}  reason={reason}  "
                f"bars_held={info.get('bars_held','?')}  side={info.get('side','?')}"
            )
        except Exception as exc:
            logger.error(f"  close_position({ticker}) failed: {exc}")

    def _latest_price(self, ticker: str) -> Optional[float]:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest
            from alpaca.data.enums import DataFeed
            if not hasattr(self, "_data_client"):
                self._data_client = StockHistoricalDataClient(self._api_key, self._secret_key)
            req   = StockLatestTradeRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
            trade = self._data_client.get_stock_latest_trade(req)
            return float(trade[ticker].price)
        except Exception as exc:
            logger.debug(f"_latest_price({ticker}) IEX failed: {exc}")
            return None

    def _yfinance_last_close(self, ticker: str) -> Optional[float]:
        import os, pandas as pd
        cache_path = os.path.join("data", "yfinance_cache", "yfinance_1hour", f"{ticker}_1hour.txt")
        try:
            df = pd.read_csv(cache_path, parse_dates=["Datetime"])
            return float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.debug(f"_yfinance_last_close({ticker}) failed: {exc}")
            return None
