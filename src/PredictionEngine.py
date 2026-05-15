import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional


class PortfolioEngine:
    """
    Optimised Long/Short Portfolio Engine with Asymmetric Exits.

    Design principles:
      1. LONG/SHORT          - Handles +1 (Long) and -1 (Short) signals.
      2. REALISTIC FEES      - 0.02 % one-way (0.04 % round-trip). Defensible for
                               large-cap US equities across the 2004-2021 window.
      3. PARTIAL PROFITS     - 50 % of position closed at half-horizon (2 h).
                               PnL stored silently in partial_profit[]; combined
                               with final exit before appending to self.trades so
                               one position = one trade record.
      4. TRAILING STOPS (widened)
                             - At +1.5 % profit: stop moves to breakeven.
                             - At +2.0 % profit: stop trails 0.75 % behind the
                               running high (long) / 0.75 % above running low (short).
                             - Thresholds widened from 1.0/1.5 % to 1.5/2.0 % so
                               normal hourly volatility on large-caps no longer
                               prematurely cuts winning positions before they can
                               develop toward the 4-hour horizon.
      5. VOL REGIME FILTER   - Signals already zeroed upstream in ensemble_model.py;
                               engine does not need to know about it.
      6. ASYMMETRIC SHORT SIZE
                             - Shorts are sized smaller than longs to reflect the
                               asymmetric risk profile (unlimited upside risk) and
                               to prevent short-heavy tickers (e.g., CAT) from
                               dominating portfolio PnL.
      7. REGIME-ADAPTIVE     - If regime_series and regime_config are provided, the
                               engine reads per-timestep config for position sizing,
                               leverage bounds, stop-loss distances, and short sizing.
                               Falls back to constructor defaults when not provided.
      8. EXIT STRATEGIES     - If self.exit_strategy is set (injected by sweep_stage3.py),
                               the engine uses one of six pluggable exit behaviours
                               instead of the default fixed-horizon close.
                               Strategies: exit_extend, exit_momentum, exit_atr_horizon,
                               exit_ladder, exit_breakeven_trail, exit_vol_regime.
    """

    # Class-level exit_strategy slot.  sweep_stage3.py assigns here before
    # pipeline.run_combined_backtest() is called, so every PortfolioEngine
    # instance created during that run picks it up automatically.
    exit_strategy: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        initial_capital: float     = 10_000.0,
        base_trade_size_pct: float = 0.20,
        transaction_fee: float     = 0.0002,
        long_safety_sl: float      = 0.05,
        short_safety_sl: float     = 0.03,
        horizon: int               = 4,
        short_size_multiplier: float = 0.50,
        results_dir: str           = "results",
        regime_series: Optional[np.ndarray] = None,
        regime_config: Optional[Dict[str, Dict[str, Any]]] = None,
        glm_guard                  = None,
        ticker_names: Optional[List[str]] = None,
        glm_context: Optional[Dict]       = None,
        regime_confidence: Optional[List]  = None,
    transition_buffer: Optional[np.ndarray] = None,
    ):
        self._transition_buffer = transition_buffer
        self.initial_capital       = initial_capital
        self.base_trade_size_pct   = base_trade_size_pct
        self.transaction_fee       = transaction_fee
        self.long_safety_sl        = long_safety_sl
        self.short_safety_sl       = short_safety_sl
        self.horizon               = horizon
        self.half_horizon          = max(1, horizon // 2)
        self.short_size_multiplier = short_size_multiplier
        self.results_dir           = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.regime_series         = regime_series
        self.regime_config         = regime_config
        self._glm_guard            = glm_guard
        self._ticker_names         = ticker_names
        self._glm_context          = glm_context
        self._regime_confidence    = regime_confidence
        self._glm_vetoes           = 0
        self._glm_early_exits      = 0
        self._glm_size_adjustments = 0

        self.equity_curve: List[float] = []
        self.trades:        List[float] = []

    # ------------------------------------------------------------------
    # Internal helper: fetch per-timestep regime params
    # ------------------------------------------------------------------

    def _regime_params(self, t: int) -> Dict[str, Any]:
        """Returns the regime config dict for timestep t, falling back to defaults."""
        if self.regime_series is not None and self.regime_config is not None:
            label = self.regime_series[t] if t < len(self.regime_series) else "BULL"
            return self.regime_config.get(label, self.regime_config.get("BULL", {}))
        return {}

    def _ticker_name(self, i: int) -> str:
        """Return the ticker symbol for column index i, or a placeholder."""
        if self._ticker_names and i < len(self._ticker_names):
            return self._ticker_names[i]
        return f"TICKER_{i}"

    def _regime_label(self, t: int) -> str:
        """Return the regime label for timestep t."""
        if self.regime_series is not None and t < len(self.regime_series):
            return str(self.regime_series[t])
        return "BULL"

    def _is_transition_period(self, t: int) -> bool:
        """Check if timestep t is within a regime transition buffer period.
        Returns False safely if buffer is not available or t is out of bounds.
        """
        if self._transition_buffer is None:
            return False
        if t >= len(self._transition_buffer):
            return False
        if len(self._transition_buffer) == 0:
            return False
        return bool(self._transition_buffer[t])

    # ------------------------------------------------------------------
    # Internal helper: resolve effective horizon for a position at entry
    # Used by exit_atr_horizon and exit_vol_regime strategies.
    # ------------------------------------------------------------------

    def _entry_horizon(self, vol_ratio: float, vol_percentile: float) -> int:
        """
        Returns the per-position horizon based on the active exit strategy.
        Falls back to self.horizon when no dynamic strategy is active.
        """
        es = self.__class__.exit_strategy
        if es is None:
            return self.horizon

        name = es.get("name", "")

        if name == "exit_atr_horizon":
            base   = es["atr_base_horizon"]
            scale  = es["atr_scale_factor"]
            cap    = es["atr_max_horizon"]
            return int(np.clip(base + round(vol_ratio * scale), base, cap))

        if name == "exit_vol_regime":
            lo = es["vol_low_threshold"]
            hi = es["vol_high_threshold"]
            horizons = es["vol_horizons"]
            if vol_percentile < lo:
                return horizons["low"]
            elif vol_percentile >= hi:
                return horizons["high"]
            else:
                return horizons["mid"]

        return self.horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_portfolio_backtest(
        self,
        price_matrix:  np.ndarray,   # (T, N)
        signal_matrix: np.ndarray,   # (T, N)  -1=Short  0=Flat  +1=Long
        prob_matrix:   np.ndarray,   # (T, N)  UP probability in [0, 1]
        vol_matrix:    Optional[np.ndarray] = None,    # (T, N) ATR/price ratio
        volpct_matrix: Optional[np.ndarray] = None,   # (T, N) vol_percentile_60
    ) -> Dict[str, Any]:
        T, N = price_matrix.shape
        cash = self.initial_capital

        es   = self.__class__.exit_strategy  # shorthand; may be None
        es_name = es["name"] if es else ""

        # ----------------------------------------------------------------
        # Per-position state arrays
        # ----------------------------------------------------------------
        shares         = np.zeros(N)
        entry_price    = np.zeros(N)
        stop_loss      = np.zeros(N)
        entry_time     = np.full(N, -1, dtype=int)
        alloc          = np.zeros(N)
        notional_alloc = np.zeros(N)
        highest_px     = np.zeros(N)
        lowest_px      = np.full(N, float('inf'))
        partial_profit = np.zeros(N)
        entry_long_sl  = np.zeros(N)
        entry_short_sl = np.zeros(N)

        # Per-position dynamic horizon (set at entry; default = self.horizon)
        pos_horizon    = np.full(N, self.horizon, dtype=int)
        # Per-position entry ATR/price ratio (needed for breakeven_trail)
        entry_vol_ratio = np.zeros(N)
        # Track how many ladder tranches have fired per position
        ladder_fired   = np.zeros(N, dtype=int)
        # Track whether the extension/momentum extension has already fired
        extended       = np.zeros(N, dtype=bool)
        # Track whether breakeven has been locked
        be_locked      = np.zeros(N, dtype=bool)
        # Track whether the trailing phase (strategy 5) is active
        trail_active   = np.zeros(N, dtype=bool)
        trail_start_t  = np.full(N, -1, dtype=int)

        # Ladder: cache initial share counts to size each tranche correctly
        initial_shares     = np.zeros(N)
        initial_alloc      = np.zeros(N)
        initial_notional   = np.zeros(N)

        self.equity_curve = []
        self.trades       = []

        for t in range(T):
            prices  = price_matrix[t]
            sigs    = signal_matrix[t]
            probs   = prob_matrix[t]
            vr_row  = vol_matrix[t]    if vol_matrix    is not None else np.zeros(N)
            vp_row  = volpct_matrix[t] if volpct_matrix is not None else np.full(N, 0.5)

            # -- 1. MANAGE OPEN POSITIONS ----------------------------------
            for i in range(N):
                if shares[i] == 0:
                    continue

                direction = 1 if shares[i] > 0 else -1
                px        = prices[i]
                entry     = entry_price[i]
                hold_time = t - entry_time[i]
                h         = int(pos_horizon[i])

                # Update running extremes
                if direction > 0:
                    highest_px[i] = max(highest_px[i], px)
                else:
                    lowest_px[i]  = min(lowest_px[i], px)

                # -- Breakeven lock (Strategy 5) ---------------------------
                if es_name == "exit_breakeven_trail" and not be_locked[i]:
                    unrealised_pct = (direction * (px - entry) / entry)
                    trigger = es.get("breakeven_trigger_pct", 0.015)
                    if unrealised_pct >= trigger:
                        stop_loss[i] = entry  # lock to breakeven
                        be_locked[i] = True

                # -- Safety stop-loss --------------------------------------
                hit_sl = (px <= stop_loss[i]) if direction > 0 else (px >= stop_loss[i])

                # -- Trailing phase activation (Strategy 5) ---------------
                if es_name == "exit_breakeven_trail" and not trail_active[i]:
                    if hold_time >= h:
                        # Normal expiry reached - enter trailing phase if profitable
                        unrealised_pct = direction * (px - entry) / entry
                        if unrealised_pct > 0:
                            trail_active[i]  = True
                            trail_start_t[i] = t
                            # Don't close yet; fall through to trail logic below
                        else:
                            hit_sl = True  # not profitable, force close now

                if trail_active[i]:
                    trail_lookback = es.get("trail_lookback", 6)
                    trail_elapsed  = t - trail_start_t[i]
                    atr_dist       = es.get("trail_atr_fraction", 0.5) * entry_vol_ratio[i] * entry
                    atr_dist       = max(atr_dist, entry * 0.003)  # floor 0.3%
                    if direction > 0:
                        retrace = highest_px[i] - px
                        trail_hit = retrace >= atr_dist
                    else:
                        retrace = px - lowest_px[i]
                        trail_hit = retrace >= atr_dist
                    if trail_hit or trail_elapsed >= trail_lookback:
                        hit_sl = True  # close via trail

                # -- Ladder partial exits (Strategy 4) --------------------
                if es_name == "exit_ladder":
                    ladder_hours  = es.get("ladder_hours",  [3, 6, 9])
                    ladder_exits  = es.get("ladder_exits",  [0.33, 0.33, 0.34])
                    fired_so_far  = int(ladder_fired[i])
                    if fired_so_far < len(ladder_hours) - 1:
                        if hold_time == ladder_hours[fired_so_far]:
                            frac       = ladder_exits[fired_so_far]
                            exit_sh    = initial_shares[i] * frac
                            exit_not   = initial_notional[i] * frac
                            exit_alloc = initial_alloc[i] * frac
                            if direction > 0:
                                net  = abs(exit_sh) * px * (1.0 - self.transaction_fee)
                                pnl  = net - exit_not
                                cash += exit_alloc + pnl
                            else:
                                cover = abs(exit_sh) * px * (1.0 + self.transaction_fee)
                                pnl   = exit_not - cover
                                cash += exit_alloc + pnl
                            shares[i]         -= exit_sh * direction  # direction-aware
                            alloc[i]          -= exit_alloc
                            notional_alloc[i] -= exit_not
                            partial_profit[i] += pnl
                            ladder_fired[i]   += 1

                # -- Standard 50% partial at half-horizon (non-ladder) ----
                is_standard_half = (
                    es_name not in ("exit_ladder",)
                    and hold_time == self.half_horizon
                    and not extended[i]
                    and not trail_active[i]
                )
                # For strategies with extended horizon, scale half-horizon too
                if es_name in ("exit_atr_horizon", "exit_vol_regime"):
                    is_standard_half = (
                        hold_time == max(1, h // 2)
                        and not extended[i]
                    )

                if is_standard_half:
                    exit_sh           = shares[i] / 2.0
                    margin_released   = alloc[i]  / 2.0
                    notional_released = notional_alloc[i] / 2.0
                    if direction > 0:
                        net               = abs(exit_sh) * px * (1.0 - self.transaction_fee)
                        partial_profit[i] += net - notional_released
                        cash             += margin_released + (net - notional_released)
                    else:
                        cover             = abs(exit_sh) * px * (1.0 + self.transaction_fee)
                        partial_profit[i] += notional_released - cover
                        cash             += margin_released + (notional_released - cover)
                    shares[i]         -= exit_sh
                    alloc[i]          -= margin_released
                    notional_alloc[i] -= notional_released

                # -- Extension logic (Strategies 1 & 2) -------------------
                should_extend = False
                if not extended[i] and not trail_active[i]:

                    if es_name == "exit_extend" and hold_time == h:
                        prob_thresh = es.get("extend_prob_threshold", 0.60)
                        vol_pct     = float(vp_row[i])
                        if vol_pct > prob_thresh:
                            excess_ratio = min((vol_pct - prob_thresh) / (1.0 - prob_thresh), 1.0)
                            extra        = max(1, round(es["extend_hours"] * excess_ratio))
                            pos_horizon[i] += extra
                            extended[i]     = True
                            should_extend   = True

                    elif es_name == "exit_momentum" and hold_time == h:
                        prob_thresh = es.get("momentum_prob_threshold", 0.58)
                        prob_val    = probs[i] if direction > 0 else (1.0 - probs[i])
                        if prob_val >= prob_thresh:
                            pos_horizon[i] += es.get("momentum_extra_hours", 4)
                            extended[i]     = True
                            should_extend   = True

                # -- Full exit (time or stop-loss) -------------------------
                # For ladder: final tranche fires at ladder_hours[-1]
                if es_name == "exit_ladder":
                    last_ladder_hour = es["ladder_hours"][-1]
                    should_close = hit_sl or (hold_time >= last_ladder_hour)
                else:
                    should_close = hit_sl or (
                        hold_time >= h and not should_extend and not trail_active[i]
                    )

                if should_close:
                    if direction > 0:
                        net          = shares[i] * px * (1.0 - self.transaction_fee)
                        final_profit = net - notional_alloc[i]
                        cash        += alloc[i] + final_profit
                    else:
                        cover        = abs(shares[i]) * px * (1.0 + self.transaction_fee)
                        final_profit = notional_alloc[i] - cover
                        cash        += alloc[i] + final_profit

                    self.trades.append(partial_profit[i] + final_profit)

                    shares[i]           = 0.0
                    entry_price[i]      = 0.0
                    stop_loss[i]        = 0.0
                    entry_time[i]       = -1
                    alloc[i]            = 0.0
                    notional_alloc[i]   = 0.0
                    highest_px[i]       = 0.0
                    lowest_px[i]        = float('inf')
                    partial_profit[i]   = 0.0
                    pos_horizon[i]      = self.horizon
                    entry_vol_ratio[i]  = 0.0
                    ladder_fired[i]     = 0
                    extended[i]         = False
                    be_locked[i]        = False
                    trail_active[i]     = False
                    trail_start_t[i]    = -1
                    initial_shares[i]   = 0.0
                    initial_alloc[i]    = 0.0
                    initial_notional[i] = 0.0

            # -- 1b. GLM HOOK 3 - EARLY EXIT REVIEW -----------------------
            if self._glm_guard is not None:
                open_pos_dict = {}
                for i in range(N):
                    if shares[i] == 0:
                        continue
                    open_pos_dict[self._ticker_name(i)] = {
                        "side":        "long" if shares[i] > 0 else "short",
                        "bars_held":   t - entry_time[i],
                        "entry_price": float(entry_price[i]),
                        "stop_price":  float(stop_loss[i]),
                    }
                if open_pos_dict:
                    regime = self._regime_label(t)
                    extra = ""
                    if self._regime_confidence and t < len(self._regime_confidence):
                        rc = self._regime_confidence[t]
                        extra += (
                            f"Regime confidence: BULL={rc.get('BULL', 0):.0%} "
                            f"BEAR={rc.get('BEAR', 0):.0%} "
                            f"CRISIS={rc.get('CRISIS', 0):.0%}\n"
                        )
                    eq   = self.equity_curve[-1] if self.equity_curve else self.initial_capital
                    peak = max(self.equity_curve) if self.equity_curve else self.initial_capital
                    dd   = (peak - eq) / peak * 100 if peak > 0 else 0
                    extra += f"Equity: {eq:,.0f} (drawdown: {dd:.1f}% from peak)"
                    early_exits = self._glm_guard.review_open_positions(
                        open_pos_dict, regime, extra_context=extra,
                    )
                    for ticker_to_close in early_exits:
                        idx = None
                        if self._ticker_names:
                            for ci in range(N):
                                if self._ticker_name(ci) == ticker_to_close and shares[ci] != 0:
                                    idx = ci
                                    break
                        if idx is None:
                            continue
                        i = idx
                        direction = 1 if shares[i] > 0 else -1
                        px = prices[i]
                        if direction > 0:
                            net          = shares[i] * px * (1.0 - self.transaction_fee)
                            final_profit = net - notional_alloc[i]
                            cash        += alloc[i] + final_profit
                        else:
                            cover        = abs(shares[i]) * px * (1.0 + self.transaction_fee)
                            final_profit = notional_alloc[i] - cover
                            cash        += alloc[i] + final_profit
                        self.trades.append(partial_profit[i] + final_profit)
                        shares[i]           = 0.0
                        entry_price[i]      = 0.0
                        stop_loss[i]        = 0.0
                        entry_time[i]       = -1
                        alloc[i]            = 0.0
                        notional_alloc[i]   = 0.0
                        highest_px[i]       = 0.0
                        lowest_px[i]        = float('inf')
                        partial_profit[i]   = 0.0
                        pos_horizon[i]      = self.horizon
                        entry_vol_ratio[i]  = 0.0
                        ladder_fired[i]     = 0
                        extended[i]         = False
                        be_locked[i]        = False
                        trail_active[i]     = False
                        trail_start_t[i]    = -1
                        initial_shares[i]   = 0.0
                        initial_alloc[i]    = 0.0
                        initial_notional[i] = 0.0
                        self._glm_early_exits += 1

            # -- 2. MARK-TO-MARKET EQUITY ----------------------------------
            open_value = 0.0
            for i in range(N):
                if shares[i] > 0:
                    unrealized_pnl = (shares[i] * prices[i]) - notional_alloc[i]
                    open_value += alloc[i] + unrealized_pnl
                elif shares[i] < 0:
                    unrealized_pnl = notional_alloc[i] - (abs(shares[i]) * prices[i])
                    open_value += alloc[i] + unrealized_pnl
            self.equity_curve.append(cash + open_value)

            # -- 3. OPEN NEW POSITIONS -------------------------------------
            rp = self._regime_params(t)
            r_size_pct  = rp.get("base_trade_size_pct",   self.base_trade_size_pct)
            r_lev_min   = rp.get("leverage_min",           2.0)
            r_lev_max   = rp.get("leverage_max",           5.0)
            r_long_sl   = rp.get("long_safety_sl",         self.long_safety_sl)
            r_short_sl  = rp.get("short_safety_sl",        self.short_safety_sl)
            r_short_mul = rp.get("short_size_multiplier",  self.short_size_multiplier)

            candidates = []
            for i in range(N):
                if sigs[i] != 0 and shares[i] == 0:
                    confidence = probs[i] if sigs[i] > 0 else (1.0 - probs[i])
                    candidates.append((i, int(sigs[i]), confidence))
            candidates.sort(key=lambda x: x[2], reverse=True)

            # Check if we're in a regime transition period for conservative sizing
            is_transition = self._is_transition_period(t)
            if is_transition and t % 50 == 0:  # Log periodically
                print(f"  [TRANSITION BUFFER] t={t}: Conservative sizing active")

            for (i, direction, confidence) in candidates:
                confidence_scalar = min((confidence - 0.5) / 0.5, 1.0)
                leverage = r_lev_min + (r_lev_max - r_lev_min) * confidence_scalar
                size_pct = r_size_pct * (0.8 + 0.4 * confidence_scalar)
                margin   = self.equity_curve[-1] * size_pct

                if direction < 0:
                    margin *= r_short_mul

                notional = margin * leverage

                if margin < 10.0 or cash < margin:
                    continue

                # -- GLM HOOK 2 - PER-TRADE GUARD -------------------------
                if self._glm_guard is not None:
                    regime = self._regime_label(t)
                    trade_enrichment = {}
                    if self._glm_context is not None:
                        ctx = self._glm_context
                        lp = ctx.get("lstm_probs")
                        if lp is not None and t < lp.shape[0] and i < lp.shape[1]:
                            trade_enrichment["model_agreement"] = {
                                "lstm": round(float(ctx["lstm_probs"][t, i]), 3),
                                "xgb":  round(float(ctx["xgb_probs"][t, i]), 3),
                                "rf":   round(float(ctx["rf_probs"][t, i]), 3),
                            }
                            trade_enrichment["sentiment"] = {
                                "score":     round(float(ctx["sent_scores"][t, i]), 3),
                                "magnitude": round(float(ctx["sent_mag"][t, i]), 3),
                            }
                            trade_enrichment["technicals"] = ctx["technicals"].get((i, t), {})
                            trade_enrichment["headlines"]  = ctx["headlines"].get((i, t), [])
                    if self._regime_confidence and t < len(self._regime_confidence):
                        trade_enrichment["regime_confidence"] = self._regime_confidence[t]
                    eq   = self.equity_curve[-1] if self.equity_curve else self.initial_capital
                    peak = max(self.equity_curve) if self.equity_curve else self.initial_capital
                    n_open  = int(np.sum(shares != 0))
                    n_long  = int(np.sum(shares > 0))
                    n_short = int(np.sum(shares < 0))
                    trade_enrichment["portfolio"] = {
                        "equity":          round(eq, 2),
                        "drawdown_pct":    round((peak - eq) / peak * 100, 1) if peak > 0 else 0,
                        "open_positions":  n_open,
                        "long_count":      n_long,
                        "short_count":     n_short,
                    }
                    decision = self._glm_guard.review_trade(
                        ticker=self._ticker_name(i),
                        direction=direction,
                        confidence=confidence,
                        regime=regime,
                        notional=notional,
                        current_price=float(prices[i]),
                        enrichment=trade_enrichment,
                    )
                    if not decision["approved"]:
                        self._glm_vetoes += 1
                        continue
                    size_mult = decision.get("size_multiplier", 1.0)
                    if size_mult != 1.0:
                        margin   *= size_mult
                        notional  = margin * leverage
                        self._glm_size_adjustments += 1

                # Resolve per-position horizon at entry
                h_entry = self._entry_horizon(
                    vol_ratio    = float(vr_row[i]),
                    vol_percentile = float(vp_row[i]),
                )
                pos_horizon[i] = h_entry

                if direction > 0:
                    sl            = prices[i] * (1.0 - r_long_sl)
                    shares[i]     = notional * (1.0 - self.transaction_fee) / prices[i]
                    highest_px[i] = prices[i]
                    lowest_px[i]  = float('inf')
                else:
                    sl            = prices[i] * (1.0 + r_short_sl)
                    shares[i]     = -(notional * (1.0 - self.transaction_fee) / prices[i])
                    lowest_px[i]  = prices[i]
                    highest_px[i] = 0.0

                cash              -= margin
                entry_price[i]     = prices[i]
                stop_loss[i]       = sl
                entry_time[i]      = t
                alloc[i]           = margin
                notional_alloc[i]  = notional
                partial_profit[i]  = 0.0
                entry_vol_ratio[i] = float(vr_row[i])
                ladder_fired[i]    = 0
                extended[i]        = False
                be_locked[i]       = False
                trail_active[i]    = False
                trail_start_t[i]   = -1
                initial_shares[i]  = shares[i]
                initial_alloc[i]   = margin
                initial_notional[i]= notional

        # -- 4. FORCE-CLOSE ALL REMAINING AT END ---------------------------
        for i in range(N):
            if shares[i] > 0:
                net          = shares[i] * price_matrix[-1, i] * (1.0 - self.transaction_fee)
                final_profit = net - notional_alloc[i]
                cash        += alloc[i] + final_profit
                self.trades.append(partial_profit[i] + final_profit)
            elif shares[i] < 0:
                cover        = abs(shares[i]) * price_matrix[-1, i] * (1.0 + self.transaction_fee)
                final_profit = notional_alloc[i] - cover
                cash        += alloc[i] + final_profit
                self.trades.append(partial_profit[i] + final_profit)

        if self.equity_curve:
            self.equity_curve[-1] = cash

        return self._calculate_metrics()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_metrics(self) -> Dict[str, Any]:
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        winning  = [t for t in self.trades if t > 0]
        losing   = [t for t in self.trades if t <= 0]
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0.0

        eq     = np.array(self.equity_curve)
        peaks  = np.maximum.accumulate(eq)
        dd     = (peaks - eq) / np.where(peaks > 0, peaks, 1.0)
        max_dd = float(np.max(dd)) * 100 if len(dd) else 0.0

        avg_win  = float(np.mean(winning)) if winning else 0.0
        avg_loss = float(np.mean(losing))  if losing  else 0.0
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        # Annualised Sharpe for hourly data (~1638 trading hours/year)
        if len(eq) > 1:
            rets = np.diff(eq) / eq[:-1]
            vol  = np.std(rets)
            sharpe_ratio = (np.mean(rets) / vol) * np.sqrt(1638) if vol > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        metrics = {
            "initial_capital" : self.initial_capital,
            "final_equity"    : final_equity,
            "total_return_pct": total_return,
            "total_trades"    : len(self.trades),
            "win_rate_pct"    : win_rate,
            "max_drawdown_pct": max_dd,
            "avg_win"         : avg_win,
            "avg_loss"        : avg_loss,
            "reward_risk"     : rr_ratio,
            "sharpe_ratio"    : sharpe_ratio,
        }
        if self._glm_guard is not None:
            metrics["glm_vetoes"]           = self._glm_vetoes
            metrics["glm_early_exits"]      = self._glm_early_exits
            metrics["glm_size_adjustments"] = self._glm_size_adjustments
        return metrics

    def plot_equity_curve(self, save_path: str = "equity_curve_portfolio.png"):
        if not self.equity_curve:
            return
        es_name = (self.__class__.exit_strategy or {}).get("name", "default")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.equity_curve, color="purple", linewidth=1.8, label="Portfolio Value")
        ax.axhline(self.initial_capital, color="red", linestyle="--",
                   alpha=0.7, label=f"Start (\u00a3{self.initial_capital:,.0f})")
        ax.fill_between(range(len(self.equity_curve)), self.initial_capital,
                        self.equity_curve,
                        where=[v >= self.initial_capital for v in self.equity_curve],
                        alpha=0.15, color="green", label="Profit zone")
        ax.fill_between(range(len(self.equity_curve)), self.initial_capital,
                        self.equity_curve,
                        where=[v < self.initial_capital for v in self.equity_curve],
                        alpha=0.15, color="red", label="Loss zone")
        ax.set_title(f"Long/Short Portfolio (£10,000) - Exit: {es_name}")
        ax.set_xlabel("Time Step (Synchronized Test-Set Hours)")
        ax.set_ylabel("Portfolio Balance (£)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.results_dir / save_path
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Equity curve saved to {out}")
