"""
sweep_stage3.py - Run Stage 3 (portfolio backtest) only for a sweep config.

Loads survivors from the existing group_tournament.csv (reuses trained models),
patches the requested constants in-memory, then calls run_combined_backtest().

Usage:
python src/sweep_stage3.py <config_name>

Configs (20 total):
high_leverage, tight_stops, no_vol_filter,
aggressive, conservative, conservative_leverage,
aggressive_novol, highleverage_novol,
trade_size_half, trade_size_double,
sl_tight_1pct, sl_wide_10pct,
bear_short_sl_tight, bear_short_sl_wide,
bear_short_threshold_lower, bear_short_threshold_higher,
bear_short_larger_trades, bear_short_higher_leverage,
bear_no_longs, bear_short_combined,
exit_breakeven_trail
"""

import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

CONFIG    = sys.argv[1] if len(sys.argv) > 1 else "baseline"
BEAR_MODE = "--bear" in sys.argv
LIVE_MODE = "--live" in sys.argv
CSV_PATH  = "results/group_tournament.csv"
THRESHOLD = 0.5

# Optional --sentiment_alpha 0.25
_sa_idx = next((i for i, a in enumerate(sys.argv) if a == "--sentiment_alpha"), None)
SENTIMENT_ALPHA = float(sys.argv[_sa_idx + 1]) if _sa_idx is not None else 0.0

# Optional --oos_holdout_pct 15
_oos_idx = next((i for i, a in enumerate(sys.argv) if a == "--oos_holdout_pct"), None)
OOS_HOLDOUT_PCT = float(sys.argv[_oos_idx + 1]) if _oos_idx is not None else 0.0

# Optional --oos_holdout_pct 15
_oos_idx = next((i for i, a in enumerate(sys.argv) if a == "--oos_holdout_pct"), None)
OOS_HOLDOUT_PCT = float(sys.argv[_oos_idx + 1]) if _oos_idx is not None else 0.0

print(f"\n{'='*60}")
print(f"  SWEEP STAGE-3-ONLY: {CONFIG.upper()}")
print(f"{'='*60}")

# -- Load survivors from existing tournament CSV ------------------------------
import csv, pathlib

def load_survivors(csv_path, threshold, max_survivors=15, min_ticker_positive_ratio=0.5):
    survivors = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                vs = float(row["val_sharpe"])
            except (ValueError, TypeError):
                continue
            if vs > threshold:
                ticker_ratio = float(row.get("ticker_positive_ratio", 1.0))
                if ticker_ratio < min_ticker_positive_ratio:
                    continue
                survivors.append({
                    "group_id":   int(row["group_id"]),
                    "stocks":     row["stocks"].split(","),
                    "val_sharpe": vs,
                })
    # Cap to top N by val_sharpe
    if max_survivors and len(survivors) > max_survivors:
        survivors.sort(key=lambda r: r["val_sharpe"], reverse=True)
        dropped = survivors[max_survivors:]
        survivors = survivors[:max_survivors]
        print(f"  Capped survivors to {max_survivors} (dropped {len(dropped)})")
    return survivors

survivors = load_survivors(CSV_PATH, THRESHOLD)
print(f"Loaded {len(survivors)} survivors from {CSV_PATH}")
if not survivors:
    print("No survivors above threshold - nothing to backtest.")
    sys.exit(0)

# -- Import pipeline (gets default constants) ---------------------------------
import pipeline          # noqa: E402
import ensemble_model    # noqa: E402
import PredictionEngine  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────────
# Fix 4: Specialist blend weight sweep
# Sweeps SPECIALIST_BLEND_BASE over [0.50, 0.60, 0.70] (specialist weight = 1 - base)
# ────────────────────────────────────────────────────────────────────────────────
SPECIALIST_BLEND_SWEEP_CONFIGS = []
for base_w in [0.50, 0.60, 0.70]:
    specialist_w = round(1.0 - base_w, 2)
    SPECIALIST_BLEND_SWEEP_CONFIGS.append({
        'name': f'SPECIALIST_BLEND_{int(base_w*100)}_{int(specialist_w*100)}',
        'specialist_base_weight': base_w,
        'specialist_specialist_weight': specialist_w,
    })
# Example entries:
# SPECIALIST_BLEND_50_50 -> base=0.50, specialist=0.50 (equal weighting)
# SPECIALIST_BLEND_60_40 -> base=0.60, specialist=0.40 (current hardcoded)
# SPECIALIST_BLEND_70_30 -> base=0.70, specialist=0.30 (base-heavy)
# -- Apply config-specific patches --------------------------------------------
if CONFIG == "high_leverage":
    # Increase leverage and trade size across all regimes
    pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.30
    pipeline.REGIME_CONFIG["BULL"]["leverage_min"]        = 3.0
    pipeline.REGIME_CONFIG["BULL"]["leverage_max"]        = 7.0
    pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.25
    pipeline.REGIME_CONFIG["CRISIS"]["leverage_min"]        = 2.5
    pipeline.REGIME_CONFIG["CRISIS"]["leverage_max"]        = 6.0
    pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"]  = 0.20
    pipeline.REGIME_CONFIG["BEAR"]["leverage_min"]         = 2.0
    pipeline.REGIME_CONFIG["BEAR"]["leverage_max"]         = 5.0
    print("  Patched: HIGH LEVERAGE (BULL 3-7x, CRISIS 2.5-6x, BEAR 2-5x)")

elif CONFIG == "tight_stops":
    # Halve stop-loss distances; trigger trailing stop earlier
    pipeline.REGIME_CONFIG["BULL"]["long_safety_sl"]  = 0.025
    pipeline.REGIME_CONFIG["BULL"]["short_safety_sl"] = 0.030
    pipeline.REGIME_CONFIG["BEAR"]["long_safety_sl"]  = 0.020
    pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.030
    # Monkey-patch trailing-stop threshold directly in the loaded module
    import PredictionEngine as _pe
    # The PortfolioEngine._manage_trailing_stop checks profit_pct >= threshold
    # Patch the module-level constant if present, else rely on REGIME_CONFIG SL
    if hasattr(_pe, "TRAIL_ACTIVATE_PCT"):
        _pe.TRAIL_ACTIVATE_PCT = 0.010
    print("  Patched: TIGHT STOPS (SL: BULL 2.5%/3%, BEAR 2%/3%)")

elif CONFIG == "no_vol_filter":
    # Remove volatility gate - all signals pass regardless of ATR
    pipeline.VOL_REGIME_PERCENTILE  = 100
    ensemble_model.MIN_VOL_RATIO    = 0.0
    ensemble_model.VOL_REGIME_PERCENTILE = 100
    print("  Patched: NO VOL FILTER (all signals admitted)")

elif CONFIG == "aggressive":
    # Lower entry thresholds for more signals
    pipeline.REGIME_CONFIG["BULL"]["long_threshold"]   = 0.51
    pipeline.REGIME_CONFIG["BULL"]["short_threshold"]  = 0.47
    pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.52
    pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"]= 0.46
    pipeline.REGIME_CONFIG["BEAR"]["long_threshold"]   = 0.55
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"]  = 0.48
    pipeline.LONG_THRESHOLD  = 0.51
    pipeline.SHORT_THRESHOLD = 0.47
    print("  Patched: AGGRESSIVE THRESHOLDS (BULL 0.51/0.47, BEAR 0.55/0.48)")

elif CONFIG == "conservative":
    # Raise entry thresholds for fewer, higher-quality signals
    pipeline.REGIME_CONFIG["BULL"]["long_threshold"]   = 0.58
    pipeline.REGIME_CONFIG["BULL"]["short_threshold"]  = 0.42
    pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.60
    pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"]= 0.41
    pipeline.REGIME_CONFIG["BEAR"]["long_threshold"]   = 0.65
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"]  = 0.42
    pipeline.LONG_THRESHOLD  = 0.58
    pipeline.SHORT_THRESHOLD = 0.42
    print("  Patched: CONSERVATIVE THRESHOLDS (BULL 0.58/0.42, BEAR 0.65/0.42)")

elif CONFIG == "conservative_leverage":
    # Ultra-low leverage across all regimes
    for regime in ["BULL", "CRISIS", "BEAR"]:
        pipeline.REGIME_CONFIG[regime]["leverage_min"] = 1.0
        pipeline.REGIME_CONFIG[regime]["leverage_max"] = 1.5
    print("  Patched: CONSERVATIVE LEVERAGE (1.0-1.5x all regimes)")

elif CONFIG == "aggressive_novol":
    # Combine: aggressive thresholds + no vol filter
    pipeline.REGIME_CONFIG["BULL"]["long_threshold"]   = 0.51
    pipeline.REGIME_CONFIG["BULL"]["short_threshold"]  = 0.47
    pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.52
    pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"]= 0.46
    pipeline.REGIME_CONFIG["BEAR"]["long_threshold"]   = 0.55
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"]  = 0.48
    pipeline.LONG_THRESHOLD  = 0.51
    pipeline.SHORT_THRESHOLD = 0.47
    pipeline.VOL_REGIME_PERCENTILE  = 100
    ensemble_model.MIN_VOL_RATIO    = 0.0
    ensemble_model.VOL_REGIME_PERCENTILE = 100
    print("  Patched: AGGRESSIVE + NO VOL FILTER")

elif CONFIG == "highleverage_novol":
    # Combine: high leverage + no vol filter
    pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.30
    pipeline.REGIME_CONFIG["BULL"]["leverage_min"]        = 3.0
    pipeline.REGIME_CONFIG["BULL"]["leverage_max"]        = 7.0
    pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.25
    pipeline.REGIME_CONFIG["CRISIS"]["leverage_min"]        = 2.5
    pipeline.REGIME_CONFIG["CRISIS"]["leverage_max"]        = 6.0
    pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"]  = 0.20
    pipeline.REGIME_CONFIG["BEAR"]["leverage_min"]         = 2.0
    pipeline.REGIME_CONFIG["BEAR"]["leverage_max"]         = 5.0
    pipeline.VOL_REGIME_PERCENTILE  = 100
    ensemble_model.MIN_VOL_RATIO    = 0.0
    ensemble_model.VOL_REGIME_PERCENTILE = 100
    print("  Patched: HIGH LEVERAGE + NO VOL FILTER")

elif CONFIG == "trade_size_half":
    # Halve position sizes across all regimes
    pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"]   = 0.10
    pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.10
    pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"]   = 0.06
    print("  Patched: TRADE SIZE HALF (BULL/CRISIS 10%, BEAR 6%)")

elif CONFIG == "trade_size_double":
    # Double position sizes across all regimes
    pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"]   = 0.40
    pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.40
    pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"]   = 0.24
    print("  Patched: TRADE SIZE DOUBLE (BULL/CRISIS 40%, BEAR 24%)")

elif CONFIG == "sl_tight_1pct":
    # Very tight stop-losses: 1% for all
    for regime in ["BULL", "CRISIS", "BEAR"]:
        pipeline.REGIME_CONFIG[regime]["long_safety_sl"]  = 0.01
        pipeline.REGIME_CONFIG[regime]["short_safety_sl"] = 0.01
    print("  Patched: TIGHT SL 1% (all regimes, both directions)")

elif CONFIG == "sl_wide_10pct":
    # Very wide stop-losses: 10% for all
    for regime in ["BULL", "CRISIS", "BEAR"]:
        pipeline.REGIME_CONFIG[regime]["long_safety_sl"]  = 0.10
        pipeline.REGIME_CONFIG[regime]["short_safety_sl"] = 0.10
    print(" Patched: WIDE SL 10% (all regimes, both directions)")

elif CONFIG == "bear_short_sl_tight":
    # Tighten short stop-loss in BEAR regime only
    pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.03
    print("  Patched: BEAR short_safety_sl 0.05 → 0.03 (tighter short stops)")

elif CONFIG == "bear_short_sl_wide":
    # Widen short stop-loss in BEAR regime only - more room for volatility
    pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.08
    print("  Patched: BEAR short_safety_sl 0.05 → 0.08 (wider short stops)")

elif CONFIG == "bear_short_threshold_lower":
    # Lower short entry threshold in BEAR regime - more aggressive short entry
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.40
    print("  Patched: BEAR short_threshold 0.45 → 0.40 (more short signals)")

elif CONFIG == "bear_short_threshold_higher":
    # Raise short entry threshold in BEAR regime - selective high-quality shorts only
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.50
    print("  Patched: BEAR short_threshold 0.45 → 0.50 (selective shorts only)")

elif CONFIG == "bear_short_larger_trades":
    # Increase base trade size in BEAR regime only
    pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.18
    print("  Patched: BEAR base_trade_size_pct 0.12 → 0.18 (larger BEAR positions)")

elif CONFIG == "bear_short_higher_leverage":
    # Increase leverage range in BEAR regime only - amplify short returns
    pipeline.REGIME_CONFIG["BEAR"]["leverage_min"] = 2.0
    pipeline.REGIME_CONFIG["BEAR"]["leverage_max"] = 4.5
    print("  Patched: BEAR leverage 1.5-3.0 → 2.0-4.5 (higher short leverage)")

elif CONFIG == "bear_no_longs":
    # Disable long trades in BEAR regime - pure short mode
    pipeline.REGIME_CONFIG["BEAR"]["long_threshold"] = 1.0
    print("  Patched: BEAR long_threshold → 1.0 (longs disabled, pure short mode)")

elif CONFIG == "bear_short_combined":
    # Combined best-of: tighter threshold, full size, tighter SL, higher leverage
    pipeline.REGIME_CONFIG["BEAR"]["short_threshold"]       = 0.42
    pipeline.REGIME_CONFIG["BEAR"]["short_size_multiplier"] = 1.00
    pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"]       = 0.04
    pipeline.REGIME_CONFIG["BEAR"]["leverage_max"]          = 4.0
    pipeline.REGIME_CONFIG["BEAR"]["leverage_min"]          = 2.0
    print("  Patched: BEAR combined (threshold→0.42, size→1.0, SL→0.04, lev→2-4x)")

# ===============================================================================
# EXIT STRATEGY CONFIGS
# Each strategy patches PortfolioEngine by injecting an exit_strategy dict
# onto the class.  PortfolioEngine.run_portfolio_backtest() reads this dict
# at the full-exit decision point inside the per-timestep loop.
#
# The dict schema is:
#   {
#     "name": str,                 # human-readable label
#     "extend_hours": int,         # Strategy 1 - fixed extension hours
#     "extend_prob_threshold": float, # Strategy 1 - min vol_percentile to trigger
#     "momentum_prob_threshold": float, # Strategy 2 - LSTM prob to continue
#     "momentum_extra_hours": int, # Strategy 2 - how long to extend
#     "atr_scale_factor": float,   # Strategy 3 - multiplier on ATR/price ratio
#     "atr_base_horizon": int,     # Strategy 3 - minimum base horizon
#     "atr_max_horizon": int,      # Strategy 3 - cap on dynamic horizon
#     "ladder_exits": list[float], # Strategy 4 - fractions e.g. [0.33, 0.33, 0.34]
#     "ladder_hours": list[int],   # Strategy 4 - timestep offsets for each tranche
#     "breakeven_trigger_pct": float, # Strategy 5 - profit % to lock breakeven
#     "trail_atr_fraction": float, # Strategy 5 - fraction of ATR for trailing window
#     "trail_lookback": int,       # Strategy 5 - rolling window length in hours
#     "vol_horizons": dict,        # Strategy 6 - {low: h, mid: h, high: h}
#     "vol_low_threshold": float,  # Strategy 6 - vol_percentile boundary low/mid
#     "vol_high_threshold": float, # Strategy 6 - vol_percentile boundary mid/high
#   }
#
# PortfolioEngine reads `self.exit_strategy` if present; falls back to default
# behaviour (fixed horizon, single 50% partial) when the attribute is absent.
# ===============================================================================

elif CONFIG == "exit_breakeven_trail":
    # -- Strategy 5 ---------------------------------------------------------
    # Two-phase exit:
    #   Phase 1 (standard): run until horizon=8 with the fixed safety SL.
    #   Breakeven lock: as soon as unrealised profit exceeds
    #       breakeven_trigger_pct (e.g., 1.5%), move stop-loss to entry price
    #       so the trade cannot lose money from that point.
    #   Phase 2 (trailing window): at normal expiry, if position is still
    #       profitable, switch to a rolling trailing close rather than hard
    #       closing.  Close when price retraces by more than
    #       trail_atr_fraction * ATR_at_entry from the running peak (long)
    #       or running low (short).  trail_lookback caps how many extra hours
    #       this phase can run before force-closing anyway.
    # -----------------------------------------------------------------------
    PredictionEngine.PortfolioEngine.exit_strategy = {
        "name":                   "exit_breakeven_trail",
        "breakeven_trigger_pct":  0.015,  # +1.5% unrealised → lock breakeven SL
        "trail_atr_fraction":     0.50,   # retrace > 0.5 * ATR triggers trail exit
        "trail_lookback":         6,      # max extra hours in trailing phase
    }
    print(" Patched: EXIT_BREAKEVEN_TRAIL - BE lock @+1.5%, trailing window @expiry up to +6h")

elif CONFIG.startswith("SPECIALIST_BLEND_"):
    # Apply specialist blend weights (config applied at runtime before backtest)
    print(f" Using SPECIALIST_BLEND config: {CONFIG}")

elif CONFIG == "baseline":
    print("  No patches - using default constants")

else:
    print(f"  Unknown config '{CONFIG}' - running with default constants")

# -- Run Stage 3 --------------------------------------------------------------
if LIVE_MODE:
    from yfinance_loader import get_yfinance_data_path
    data_path    = get_yfinance_data_path()
    anchor_end   = None
    train_anchor = None
    _sent_tag    = f"_s{SENTIMENT_ALPHA}" if SENTIMENT_ALPHA > 0.0 else ""
    eq_name      = f"sweep/equity_{CONFIG}_live{_sent_tag}.png"
    split_months = (12, 4, 6)   # matches --live pipeline split
    print(f"  LIVE MODE: yfinance data ({data_path}), split 12/4/6 months")
elif BEAR_MODE:
    data_path    = pipeline.DATA_PATH
    anchor_end   = pipeline.BEAR_TEST_END
    train_anchor = pipeline.BEAR_TRAIN_END
    eq_name      = f"sweep/equity_{CONFIG}_bear.png"
    split_months = (12, 2, 3)
    print(f"  BEAR MODE: train anchor={train_anchor}, test anchor={anchor_end}")
else:
    data_path    = pipeline.DATA_PATH
    anchor_end   = None
    train_anchor = None
    eq_name      = f"sweep/equity_{CONFIG}.png"
    split_months = None

pathlib.Path("results/sweep").mkdir(parents=True, exist_ok=True)

if SENTIMENT_ALPHA > 0.0:
    print(f"  Sentiment gate: alpha={SENTIMENT_ALPHA} (adaptive confidence-weighted)")

# Fix 4: Apply specialist blend weights
if CONFIG.startswith("SPECIALIST_BLEND_"):
    for cfg in SPECIALIST_BLEND_SWEEP_CONFIGS:
        if cfg["name"] == CONFIG:
            import ensemble_model as _em
            _em.SPECIALIST_BLEND_BASE = cfg["specialist_base_weight"]
            _em.SPECIALIST_BLEND_SPECIALIST = cfg["specialist_specialist_weight"]
            print(f" [SPECIALIST_BLEND] base={_em.SPECIALIST_BLEND_BASE}, specialist={_em.SPECIALIST_BLEND_SPECIALIST}")
            break

metrics = pipeline.run_combined_backtest(
    survivors=survivors,
    data_path=data_path,
    anchor_end_date=anchor_end,
    equity_curve_name=eq_name,
    sentiment_alpha=SENTIMENT_ALPHA,
    train_anchor_date=train_anchor,
    use_full_split=False,
    mtl_model=None,
    split_months=split_months,
    oos_holdout_pct=OOS_HOLDOUT_PCT,
)

print(f"\nConfig '{CONFIG}' complete.")
print(f"  Return : {metrics.get('total_return_pct', 0):+.2f}%")
print(f"  Sharpe : {metrics.get('sharpe_ratio', 0):.3f}")
print(f"  MaxDD  : {metrics.get('max_drawdown_pct', 0):.2f}%")
print(f"  WinRate: {metrics.get('win_rate_pct', 0):.1f}%")
print(f"  Trades : {metrics.get('total_trades', 0)}")
