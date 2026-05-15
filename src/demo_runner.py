"""
demo_runner.py  -  Hourly Live Demo Scheduler

Runs the pipeline's inference step (NOT a full retrain) once per market
hour, extracts the latest signal row from the ensemble, and delegates
order execution to AlpacaBridge.

Schedule
--------
  Market hours: 09:30-16:00 ET  =  14:30-21:00 UTC
  The scheduler fires at :00 of each UTC hour inside that window.
  If started mid-hour while markets are already open, the first bar
  fires immediately (no waiting until the next :00 mark).

Model retraining
----------------
  The pipeline is NOT retrained on every bar - that would take hours.
  Instead, pre-trained weights from a prior  `python src/pipeline.py --live`
  run are reused.  A weekly retrain (Sunday night) is recommended to keep
  the model fresh; set RETRAIN_DAY below to automate this.

Usage
-----
  # 1. One-time: train the model and save weights
  python src/pipeline.py --live --n_stocks 100

  # 2. Set your Alpaca paper credentials as environment variables:
  export ALPACA_API_KEY="your_key_here"
  export ALPACA_SECRET_KEY="your_secret_here"

  # 3. Start the demo runner:
  python src/demo_runner.py

  # Optional flags (passed as env vars or edited below):
  #   DEMO_SENTIMENT_ALPHA=0.25   - enable FinBERT sentiment gating
  #   DEMO_N_STOCKS=100           - how many stocks the live pipeline uses
  #   DEMO_DRY_RUN=1              - print signals without submitting orders
  #   DEMO_FORCE=1                - bypass market hours gate (useful for testing)

  # Example: test signal generation right now regardless of market hours:
  DEMO_DRY_RUN=1 DEMO_FORCE=1 python src/demo_runner.py

Dependencies
------------
  pip install alpaca-py schedule
"""

import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# -- Configuration (override via environment variables) ------------------------
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER_MODE        = os.environ.get("PAPER_MODE", "1") != "0"   # default: paper
SENTIMENT_ALPHA   = float(os.environ.get("DEMO_SENTIMENT_ALPHA", "0.0"))
N_STOCKS          = int(os.environ.get("DEMO_N_STOCKS", "280"))
DRY_RUN           = os.environ.get("DEMO_DRY_RUN", "0") != "0"
FORCE             = os.environ.get("DEMO_FORCE", "0") != "0"   # bypass market hours
USE_QWEN          = os.environ.get("DEMO_QWEN", "0") != "0"    # enable Qwen LLM signal review
RETRAIN_DAY       = int(os.environ.get("DEMO_RETRAIN_DAY", "6"))  # 6=Sunday

# Pipeline constants (must match pipeline.py)
LOOKBACK  = 24
HORIZON   = 8
DATA_PATH = "data/yfinance_cache"   # populated by --live run

# Split used by --live mode in pipeline.py
LIVE_SPLIT_MONTHS = (12, 4, 6)

# NYSE market hours in US/Eastern local time (handles EST/EDT automatically)
MARKET_OPEN_HOUR  = 9
MARKET_OPEN_MIN   = 30
MARKET_CLOSE_HOUR = 16   # 4:00 PM ET (exclusive)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# Market hours helpers
# -----------------------------------------------------------------------------

def _is_market_open() -> bool:
    """Returns True if current time is within NYSE market hours on a weekday.
    Uses US/Eastern timezone so EST/EDT is handled automatically."""
    if FORCE:
        return True
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # Python 3.8 fallback
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    open_minutes    = MARKET_OPEN_HOUR  * 60 + MARKET_OPEN_MIN
    close_minutes   = MARKET_CLOSE_HOUR * 60
    current_minutes = now.hour * 60 + now.minute
    return open_minutes <= current_minutes < close_minutes


_first_bar_fired = False

def _seconds_until_next_bar() -> float:
    """
    Returns seconds until the next :00 mark (hourly boundary).
    On the very first call during market hours, returns 0 so the first bar
    fires immediately.  After that, waits for the next hour boundary.
    If FORCE is set, always returns 0 (single-pass mode).
    """
    global _first_bar_fired
    if FORCE:
        return 0.0
    if _is_market_open() and not _first_bar_fired:
        # First bar fires immediately so we don't wait up to 59 minutes
        _first_bar_fired = True
        return 0.0
    now = datetime.now(timezone.utc)
    seconds_past = now.minute * 60 + now.second
    return max(0.0, 3600.0 - seconds_past)


# -----------------------------------------------------------------------------
# Data refresh - update cache with latest candles before inference
# -----------------------------------------------------------------------------

def _refresh_cache() -> None:
    """
    Incrementally update the yfinance cache for all tickers in the manifest.

    Reads each ticker's cache file, finds the last timestamp, and downloads
    only the missing candles since then.  If a ticker has no cache file yet,
    does a full 2-year download.  Uses the direct Yahoo API - no curl_cffi.
    """
    manifest_path = Path("models/master_manifest.json")
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest: List[dict] = json.load(f)

    tickers = sorted({t for entry in manifest for t in entry["stocks"]})
    if not tickers:
        return

    logger.info(f"Refreshing cache for {len(tickers)} tickers (incremental)...")

    try:
        from yfinance_loader import update_cache_incremental
    except ImportError as exc:
        logger.warning(f"Cannot refresh cache - import failed: {exc}")
        return

    stats = update_cache_incremental(tickers, cache_dir=DATA_PATH, delay=0.05)
    logger.info(
        f"Cache refresh: {stats['updated']} updated, {stats['fresh']} new, "
        f"{stats['skipped']} already current, {stats['failed']} failed"
    )


# -----------------------------------------------------------------------------
# Signal extraction (inference-only - no full retrain)
# -----------------------------------------------------------------------------

def extract_latest_signals(
    sentiment_alpha: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Runs a lightweight inference pass using pre-trained model weights to
    produce the latest signal row.

    Returns a dict with keys:
        signal_row  : np.ndarray of int  (-1 / 0 / +1), shape (N_tickers,)
        prob_row    : np.ndarray of float (UP probability), shape (N_tickers,)
        tickers     : List[str] of ticker symbols
        regime      : str - current portfolio regime (BULL / BEAR / CRISIS)

    Returns None if no trained models are found or data cannot be fetched.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    manifest_path = Path("models/master_manifest.json")
    if not manifest_path.exists():
        logger.error(
            "No trained models found at models/master_manifest.json. "
            "Run  `python src/pipeline.py --live`  first."
        )
        return None

    with open(manifest_path) as f:
        manifest: List[dict] = json.load(f)

    if not manifest:
        logger.warning("master_manifest.json is empty - nothing to run.")
        return None

    # -- Import pipeline helpers -------------------------------------------
    try:
        from ensemble_model import run_ensemble
        from regime_detector import RegimeDetector
        from pipeline import REGIME_CONFIG, LONG_THRESHOLD, SHORT_THRESHOLD, VOL_REGIME_PERCENTILE
    except ImportError as exc:
        logger.error(f"Pipeline import failed: {exc}")
        return None

    all_prices, all_probs   = [], []
    all_long_g, all_short_g = [], []
    all_atr_vol             = []
    all_enriched, all_row_idx = [], []
    all_tickers_ordered     = []

    for entry in manifest:
        group_id = entry["group_id"]
        tickers  = entry["stocks"]
        logger.info(f"Generating signals for group {group_id}: {tickers}")
        try:
            signals = run_ensemble(
                tickers=tickers,
                group_id=group_id,
                lookback=LOOKBACK,
                horizon=HORIZON,
                data_path=DATA_PATH,
                anchor_end_date=None,
                sentiment_alpha=sentiment_alpha,
                use_full_split=False,
                split_months=LIVE_SPLIT_MONTHS,
            )
        except Exception as exc:
            logger.warning(f"Group {group_id} signal generation failed: {exc}")
            continue

        all_prices.extend(signals["prices"])
        all_probs.extend(signals["probs"])
        all_long_g.extend(signals["long_g"])
        all_short_g.extend(signals["short_g"])
        all_atr_vol.extend(signals["atr_vol"])
        all_enriched.extend(signals["enriched"])
        all_row_idx.extend(signals["row_idx"])
        all_tickers_ordered.extend(tickers)
        gc.collect()

    if not all_prices:
        logger.warning("No signal data produced - check data and model weights.")
        return None

    # -- Align timelines (same logic as run_combined_backtest) -------------
    lengths = [len(p) for p in all_prices]
    min_len = int(np.percentile(lengths, 10))
    n_dropped = sum(1 for l in lengths if l < min_len)
    if n_dropped:
        print(f" {n_dropped} ticker(s) below 10th-percentile length dropped from alignment")
    # Filter out tickers below min_len before stacking to avoid dimension mismatch
    filtered_prices = [p[-min_len:] for p in all_prices if len(p) >= min_len]
    filtered_probs = [p[-min_len:] for p in all_probs if len(p) >= min_len]
    filtered_long_g = [g[-min_len:] for g in all_long_g if len(g) >= min_len]
    filtered_short_g = [g[-min_len:] for g in all_short_g if len(g) >= min_len]
    filtered_vol = [v[-min_len:] for v in all_atr_vol if len(v) >= min_len]
    if not filtered_prices:
        logger.error("No tickers remaining after length filtering")
        return None
    price_matrix = np.column_stack(filtered_prices)
    prob_matrix = np.column_stack(filtered_probs)
    long_g_mat = np.column_stack(filtered_long_g)
    short_g_mat = np.column_stack(filtered_short_g)
    vol_matrix = np.column_stack(filtered_vol)

    vol_p95         = np.percentile(vol_matrix, VOL_REGIME_PERCENTILE, axis=0)
    high_vol_regime = vol_matrix > vol_p95[np.newaxis, :]

    # -- Regime detection --------------------------------------------------
    regime        = "BULL"
    detector_path = Path("models/regime_detector.pkl")
    if detector_path.exists():
        try:
            detector = RegimeDetector.load(str(detector_path))
            regime_series_list = []
            for enriched_df, row_idx in zip(all_enriched, all_row_idx):
                aligned_idx    = row_idx[-min_len:]
                ticker_regimes = detector.predict_regime_series(
                    enriched_df.iloc[aligned_idx]
                )
                regime_series_list.append(ticker_regimes[-min_len:])
            regime_matrix = np.array(regime_series_list)
            last_col      = regime_matrix[:, -1]
            unique, counts = np.unique(last_col, return_counts=True)
            regime = unique[np.argmax(counts)]
            logger.info(f"Current regime: {regime}")
        except Exception as exc:
            logger.warning(f"RegimeDetector failed - defaulting to BULL. ({exc})")

    # -- Build signal matrix (identical to pipeline.py run_combined_backtest) --
    cfg       = REGIME_CONFIG[regime]
    long_thr  = cfg["long_threshold"]
    short_thr = cfg["short_threshold"]

    signal_matrix = np.zeros_like(prob_matrix, dtype=int)
    for t in range(min_len):
        for i in range(prob_matrix.shape[1]):
            prob = prob_matrix[t, i]
            if regime == "BULL":
                if prob >= long_thr:
                    signal_matrix[t, i] = 1
                elif prob <= short_thr:
                    signal_matrix[t, i] = -1
            else:
                if prob >= long_thr and long_g_mat[t, i] and not high_vol_regime[t, i]:
                    signal_matrix[t, i] = 1
                elif prob <= short_thr and short_g_mat[t, i]:
                    signal_matrix[t, i] = -1

    # -- Return ONLY the latest bar's signals ------------------------------
    signal_row = signal_matrix[-1]
    prob_row   = prob_matrix[-1]

    logger.info(
        f"Latest bar - Long: {np.sum(signal_row==1)}  "
        f"Short: {np.sum(signal_row==-1)}  "
        f"Flat: {np.sum(signal_row==0)}  "
        f"Regime: {regime}"
    )

    return {
        "signal_row": signal_row,
        "prob_row":   prob_row,
        "tickers":    all_tickers_ordered,
        "regime":     str(regime),
    }


def load_optimized_config(max_age_days: int = 7) -> Optional[Dict[str, Any]]:
    """
    Load optimized sweep config if it exists and is not stale.
    
    Args:
        max_age_days: Maximum age of config file before ignoring it
        
    Returns:
        Config dict with 'config' key and params, or None if not available
    """
    config_path = Path("models/optimized_config.json")
    
    if not config_path.exists():
        logger.info("No optimized config found - using defaults")
        return None
    
    try:
        with open(config_path) as f:
            data = json.load(f)
        
        # Check age
        timestamp = data.get("timestamp", "")
        if timestamp:
            config_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - config_time).days
            
            if age_days > max_age_days:
                logger.warning(
                    f"Optimized config is {age_days} days old (max={max_age_days}) - "
                    "using defaults. Run sweep_selector.py to refresh."
                )
                return None
        
        logger.info(
            f"Loaded optimized config: {data['config']} "
            f"(Sharpe={data['sharpe']:.3f}, {age_days}d old)"
        )
        return data
        
    except Exception as exc:
        logger.warning(f"Failed to load optimized config: {exc} - using defaults")
        return None


def apply_config_patches(config_data: Dict[str, Any]) -> None:
    """
    Apply optimized config patches to pipeline modules.
    
    Uses sweep_stage3.py's patching logic by importing and running the
    equivalent patches based on config name.
    """
    config_name = config_data.get("config", "baseline")
    
    if config_name == "baseline":
        logger.info("Optimized config is baseline - no patches needed")
        return
    
    # Import pipeline modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    import pipeline
    import ensemble_model
    import PredictionEngine
    
    logger.info(f"Applying patches for config: {config_name}")
    
    # Apply patches based on config name
    # (Matches sweep_stage3.py logic)
    if config_name == "high_leverage":
        pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.30
        pipeline.REGIME_CONFIG["BULL"]["leverage_min"] = 3.0
        pipeline.REGIME_CONFIG["BULL"]["leverage_max"] = 7.0
        pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.25
        pipeline.REGIME_CONFIG["CRISIS"]["leverage_min"] = 2.5
        pipeline.REGIME_CONFIG["CRISIS"]["leverage_max"] = 6.0
        pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.20
        pipeline.REGIME_CONFIG["BEAR"]["leverage_min"] = 2.0
        pipeline.REGIME_CONFIG["BEAR"]["leverage_max"] = 5.0
        
    elif config_name == "tight_stops":
        pipeline.REGIME_CONFIG["BULL"]["long_safety_sl"] = 0.025
        pipeline.REGIME_CONFIG["BULL"]["short_safety_sl"] = 0.030
        pipeline.REGIME_CONFIG["BEAR"]["long_safety_sl"] = 0.020
        pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.030
        
    elif config_name == "no_vol_filter":
        pipeline.VOL_REGIME_PERCENTILE = 100
        ensemble_model.MIN_VOL_RATIO = 0.0
        ensemble_model.VOL_REGIME_PERCENTILE = 100
        
    elif config_name == "aggressive":
        pipeline.REGIME_CONFIG["BULL"]["long_threshold"] = 0.51
        pipeline.REGIME_CONFIG["BULL"]["short_threshold"] = 0.47
        pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.52
        pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"] = 0.46
        pipeline.REGIME_CONFIG["BEAR"]["long_threshold"] = 0.55
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.48
        pipeline.LONG_THRESHOLD = 0.51
        pipeline.SHORT_THRESHOLD = 0.47
        
    elif config_name == "conservative":
        pipeline.REGIME_CONFIG["BULL"]["long_threshold"] = 0.58
        pipeline.REGIME_CONFIG["BULL"]["short_threshold"] = 0.42
        pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.60
        pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"] = 0.41
        pipeline.REGIME_CONFIG["BEAR"]["long_threshold"] = 0.65
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.42
        pipeline.LONG_THRESHOLD = 0.58
        pipeline.SHORT_THRESHOLD = 0.42
        
    elif config_name == "conservative_leverage":
        for regime in ["BULL", "CRISIS", "BEAR"]:
            pipeline.REGIME_CONFIG[regime]["leverage_min"] = 1.0
            pipeline.REGIME_CONFIG[regime]["leverage_max"] = 1.5
            
    elif config_name == "aggressive_novol":
        pipeline.REGIME_CONFIG["BULL"]["long_threshold"] = 0.51
        pipeline.REGIME_CONFIG["BULL"]["short_threshold"] = 0.47
        pipeline.REGIME_CONFIG["CRISIS"]["long_threshold"] = 0.52
        pipeline.REGIME_CONFIG["CRISIS"]["short_threshold"] = 0.46
        pipeline.REGIME_CONFIG["BEAR"]["long_threshold"] = 0.55
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.48
        pipeline.LONG_THRESHOLD = 0.51
        pipeline.SHORT_THRESHOLD = 0.47
        pipeline.VOL_REGIME_PERCENTILE = 100
        ensemble_model.MIN_VOL_RATIO = 0.0
        ensemble_model.VOL_REGIME_PERCENTILE = 100
        
    elif config_name == "highleverage_novol":
        pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.30
        pipeline.REGIME_CONFIG["BULL"]["leverage_min"] = 3.0
        pipeline.REGIME_CONFIG["BULL"]["leverage_max"] = 7.0
        pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.25
        pipeline.REGIME_CONFIG["CRISIS"]["leverage_min"] = 2.5
        pipeline.REGIME_CONFIG["CRISIS"]["leverage_max"] = 6.0
        pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.20
        pipeline.REGIME_CONFIG["BEAR"]["leverage_min"] = 2.0
        pipeline.REGIME_CONFIG["BEAR"]["leverage_max"] = 5.0
        pipeline.VOL_REGIME_PERCENTILE = 100
        ensemble_model.MIN_VOL_RATIO = 0.0
        ensemble_model.VOL_REGIME_PERCENTILE = 100
        
    elif config_name == "trade_size_half":
        pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.10
        pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.10
        pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.06
        
    elif config_name == "trade_size_double":
        pipeline.REGIME_CONFIG["BULL"]["base_trade_size_pct"] = 0.40
        pipeline.REGIME_CONFIG["CRISIS"]["base_trade_size_pct"] = 0.40
        pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.24
        
    elif config_name == "sl_tight_1pct":
        for regime in ["BULL", "CRISIS", "BEAR"]:
            pipeline.REGIME_CONFIG[regime]["long_safety_sl"] = 0.01
            pipeline.REGIME_CONFIG[regime]["short_safety_sl"] = 0.01
            
    elif config_name == "sl_wide_10pct":
        for regime in ["BULL", "CRISIS", "BEAR"]:
            pipeline.REGIME_CONFIG[regime]["long_safety_sl"] = 0.10
            pipeline.REGIME_CONFIG[regime]["short_safety_sl"] = 0.10
            
    elif config_name == "bear_short_sl_tight":
        pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.03
        
    elif config_name == "bear_short_sl_wide":
        pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.08
        
    elif config_name == "bear_short_threshold_lower":
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.40
        
    elif config_name == "bear_short_threshold_higher":
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.50
        
    elif config_name == "bear_short_larger_trades":
        pipeline.REGIME_CONFIG["BEAR"]["base_trade_size_pct"] = 0.18
        
    elif config_name == "bear_short_higher_leverage":
        pipeline.REGIME_CONFIG["BEAR"]["leverage_min"] = 2.0
        pipeline.REGIME_CONFIG["BEAR"]["leverage_max"] = 4.5
        
    elif config_name == "bear_no_longs":
        pipeline.REGIME_CONFIG["BEAR"]["long_threshold"] = 1.0
        
    elif config_name == "bear_short_combined":
        pipeline.REGIME_CONFIG["BEAR"]["short_threshold"] = 0.42
        pipeline.REGIME_CONFIG["BEAR"]["short_size_multiplier"] = 1.00
        pipeline.REGIME_CONFIG["BEAR"]["short_safety_sl"] = 0.04
        pipeline.REGIME_CONFIG["BEAR"]["leverage_max"] = 4.0
        pipeline.REGIME_CONFIG["BEAR"]["leverage_min"] = 2.0
        
    elif config_name == "exit_breakeven_trail":
        PredictionEngine.PortfolioEngine.exit_strategy = {
            "name": "exit_breakeven_trail",
            "breakeven_trigger_pct": 0.015,
            "trail_atr_fraction": 0.50,
            "trail_lookback": 6,
        }
        
    else:
        logger.warning(f"Unknown config '{config_name}' - using defaults")


# -----------------------------------------------------------------------------
# Weekly retrain trigger
# -----------------------------------------------------------------------------

_last_retrain_week: Optional[int] = None

def _maybe_retrain() -> None:
    """
    Triggers a full --live retrain once per week (on RETRAIN_DAY).
    The retrain runs in a subprocess so the scheduler is not blocked.
    """
    global _last_retrain_week
    now  = datetime.now(timezone.utc)
    week = now.isocalendar()[1]  # ISO week number

    if now.weekday() != RETRAIN_DAY:
        return
    if _last_retrain_week == week:
        return  # already retrained this week

    import subprocess, sys
    logger.info(f"[Retrain] Launching weekly retrain (RETRAIN_DAY={RETRAIN_DAY})...")
    try:
        subprocess.Popen(
            [
                sys.executable, "src/pipeline.py",
                "--live",
                f"--n_stocks={N_STOCKS}",
            ],
            stdout=open("logs/retrain.log", "a"),
            stderr=subprocess.STDOUT,
        )
        _last_retrain_week = week
    except Exception as exc:
        logger.error(f"[Retrain] Failed to launch retrain subprocess: {exc}")


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main() -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise EnvironmentError(
            "Alpaca credentials not set. Export ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY as environment variables before running."
        )
    
    # -- Load optimized sweep config if available -----------------------------
    opt_config = load_optimized_config(max_age_days=7)
    if opt_config:
        apply_config_patches(opt_config)

    from demo_bridge import AlpacaBridge

    bridge = AlpacaBridge(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=PAPER_MODE,
        horizon=4,
    )

    # -- Optional Qwen LLM signal guard --------------------------------------
    llm_guard = None
    if USE_QWEN:
        try:
            from llm_guard import GLMGuard
            llm_guard = GLMGuard()
            logger.info("Qwen LLM guard enabled - signals will be reviewed before execution.")
        except Exception as exc:
            logger.warning(f"Qwen LLM guard failed to initialise: {exc} - continuing without it.")

    Path("logs").mkdir(exist_ok=True)
    mode_label = "DRY RUN" if DRY_RUN else ("PAPER" if PAPER_MODE else "LIVE")
    if FORCE:
        mode_label += " (FORCED - market hours bypassed)"
    if llm_guard:
        mode_label += " + QWEN"
    logger.info(f"demo_runner started - mode={mode_label}  sentiment_alpha={SENTIMENT_ALPHA}")

    if _is_market_open():
        logger.info("Markets are open - running first inference pass immediately.")
    else:
        logger.info("Waiting for market open (14:30 UTC)...")

    try:
        while True:
            # Fire immediately if already in market hours; otherwise wait for
            # the next :00 mark so subsequent bars stay on the hour boundary.
            sleep_secs = _seconds_until_next_bar()
            if sleep_secs > 1:
                time.sleep(sleep_secs)

            if not _is_market_open():
                logger.debug("Outside market hours - sleeping 60s")
                time.sleep(60)
                continue

            # -- Optional weekly retrain ------------------------------------
            _maybe_retrain()

            # -- Refresh cache with latest candles ------------------------
            try:
                _refresh_cache()
            except Exception as exc:
                logger.warning(f"Cache refresh failed ({exc}) - using existing data")

            # -- Inference pass --------------------------------------------
            logger.info("Running inference pass...")
            result = extract_latest_signals(sentiment_alpha=SENTIMENT_ALPHA)

            if result is None:
                logger.warning("No signals produced this bar - skipping.")
                time.sleep(60)
                continue

            signal_row = result["signal_row"]
            prob_row   = result["prob_row"]
            tickers    = result["tickers"]
            regime     = result["regime"]

            # -- Qwen LLM signal review (Hook 1) ------------------------
            if llm_guard is not None:
                pre_count = int(np.sum(signal_row != 0))
                try:
                    approved = llm_guard.review_signals(
                        signal_row, prob_row, tickers, regime=regime,
                    )
                    vetoed = 0
                    for i in range(len(tickers)):
                        if signal_row[i] != 0 and not approved[i]:
                            signal_row[i] = 0
                            vetoed += 1
                    if vetoed:
                        logger.info(f"  Qwen vetoed {vetoed}/{pre_count} signals")
                except Exception as exc:
                    logger.warning(f"  Qwen review failed ({exc}) - signals unchanged")

            # -- Execute via bridge (or just log in dry-run mode) ----------
            if DRY_RUN:
                logger.info("[DRY RUN] Signals (no orders submitted):")
                for i, ticker in enumerate(tickers):
                    sig = signal_row[i]
                    if sig != 0:
                        direction = "LONG" if sig > 0 else "SHORT"
                        logger.info(
                            f"  {direction:5s} {ticker:<6}  prob={prob_row[i]:.3f}  "
                            f"regime={regime}"
                        )
            else:
                bridge.step(signal_row, prob_row, tickers, regime=regime)

            gc.collect()

            # In FORCE mode run once then exit cleanly
            if FORCE:
                logger.info("[FORCE] Single inference pass complete - exiting.")
                break

            # Loop back - _seconds_until_next_bar() will wait until the next :00
            # hourly boundary so each bar = 1 hour (matching backtest bars).

    except KeyboardInterrupt:
        logger.info("Interrupted - saving state and shutting down (positions kept open).")
        if not DRY_RUN:
            bridge._save_state()


if __name__ == "__main__":
    main()
