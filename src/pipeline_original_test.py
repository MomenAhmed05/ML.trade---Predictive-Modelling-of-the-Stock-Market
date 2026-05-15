"""
Hierarchical Grouped Training Pipeline

Stage 1 -- Correlation Clustering:
  Clusters top N liquid high-beta stocks into groups using Ward hierarchical
  clustering on pairwise return correlations.

Stage 2 -- Per-Group Training & Val-Set Selection:
  Each group is trained independently (or jointly via MTL).
  Groups are ranked by val-set Sharpe. Only groups exceeding
  `selection_threshold` survive to the backtest.

Stage 3 -- Combined Portfolio Backtest:
  All surviving groups feed signals into a single PortfolioEngine run.

Usage:
    python pipeline.py                                        # default run (anchored 12/2/3 month split)
    python pipeline.py --full                                 # full data run (60/20/20 split across all years)
    python pipeline.py --mtl                                  # use shared/private MTL LSTM
    python pipeline.py --compare                              # run LSTM then MTL on same groups; print comparison
    python pipeline.py --sentiment_alpha 0.25                 # with FinBERT sentiment gate
    python pipeline.py --bear --sentiment_alpha 0.25          # Bear market (2018 Q4)
    python pipeline.py --covid --sentiment_alpha 0.25         # COVID crash (2020 Q1-Q2)

The --bear and --covid flags retrain from scratch on data truncated to the
relevant period and evaluate on the regime test window.

The --full flag disables the anchored 12/2/3-month windowed split and uses
a proportional 60/20/20 percentage split across the full dataset.
All three stages receive use_full_split=True which is forwarded into
DataPreprocessor.preprocess() via train_group and run_*_backtest helpers.

The --mtl flag replaces the per-group LSTM with a single shared/private
Multi-Task Learning LSTM trained across ALL groups simultaneously.

The --compare flag runs BOTH the standard LSTM tournament AND the MTL
tournament on the exact same groups/tickers, then prints a side-by-side
metrics comparison and saves results/comparison.csv.
If results/group_tournament_lstm.csv already exists, the LSTM arm is
skipped and its results are read from disk so you can resume after a crash.
"""

import argparse
import csv
import gc
import json
import os
import random

# -- Deterministic seeding -----------------------------------------------------
# Parse --seed early (before numpy/tf imports) so every RNG is seeded.
import sys as _sys
_seed_idx = next((i for i, a in enumerate(_sys.argv) if a == "--seed"), None)
_GLOBAL_SEED = int(_sys.argv[_seed_idx + 1]) if _seed_idx is not None else 42
random.seed(_GLOBAL_SEED)
os.environ["PYTHONHASHSEED"] = str(_GLOBAL_SEED)

import numpy as np
np.random.seed(_GLOBAL_SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(_GLOBAL_SEED)
    tf.config.optimizer.set_jit(True)
except ImportError:
    pass  # TF imported later when needed

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from regime_specialist import RegimeSpecialistLSTM, route_to_specialist, SPECIALIST_CONFIGS
from saml_meta_learner import SAMLMetaLearner
from quantile_barrier import QuantileBarrierLearner


# --------------------------------------------------------------------------------
DATA_PATH = "data/raw"
LOOKBACK              = 24
HORIZON               = 8
LONG_THRESHOLD        = 0.53
# FIX D: align module-level SHORT_THRESHOLD with the BULL regime config
SHORT_THRESHOLD       = 0.45
VOL_REGIME_PERCENTILE = 95
MIN_TEST_HOURS        = int(3 * 730 * 0.80)   # 1752 hours -- minimum for default/bear/covid modes

# For --full mode the test split is ~20% of ~140K rows ≈ 28K rows.
# Raise the minimum bar proportionally so tiny tickers are still filtered.
MIN_TEST_HOURS_FULL   = int(12 * 730 * 0.80)  # ~7,008 hours (~9.6 months)

# Regime anchor dates
# train_end: data is truncated here for clustering + training
# test_end:  backtest evaluation window ends here (None = tail of dataset)
BEAR_TRAIN_END  = "2018-09-30"
BEAR_TEST_END   = "2019-01-31"
COVID_TRAIN_END = "2019-12-31"
COVID_TEST_END  = "2020-06-30"

# --------------------------------------------------------------------------------
# Regime-adaptive configuration
# The RegimeDetector (HMM) labels each timestep as BULL / BEAR / CRISIS.
# PortfolioEngine and the signal builder read per-timestep config from here.
# --------------------------------------------------------------------------------
REGIME_CONFIG: Dict[str, Dict[str, Any]] = {
    "BULL": {
        # Aggressive thresholds for BULL regime only -- sweep showed Sharpe 3.32
        # in default mode (lowering entry bar admits more profitable signals
        # when the market trend supports them). Reverts to default in BEAR/CRISIS.
        "long_threshold":        0.51,
        "short_threshold":       0.45,
        "long_safety_sl":        0.05,
        "short_safety_sl":       0.05,
        "base_trade_size_pct":   0.25,
        "leverage_min":          3.0,
        "leverage_max":          6.0,
        "short_size_multiplier": 0.50,
        "transition_size_factor": 0.5,  # Reduce position size by 50% during transitions
        "transition_leverage_factor": 0.0,  # Use min leverage during transitions
    },
    "CRISIS": {
        "long_threshold":        0.90,
        "short_threshold":       0.20,
        "long_safety_sl":        0.05,
        "short_safety_sl":       0.05,
        "base_trade_size_pct":   0.25,
        "leverage_min":          2.5,
        "leverage_max":          6.0,
        "short_size_multiplier": 0.50,
        "transition_size_factor": 0.5,
        "transition_leverage_factor": 0.0,
    },
	"BEAR": {
		"long_threshold": 0.55,
		"short_threshold": 0.40,
		"long_safety_sl": 0.04,
		"short_safety_sl": 0.06,
		"base_trade_size_pct": 0.20,
		"leverage_min": 2.0,
		"leverage_max": 4.0,
		"short_size_multiplier": 1.0,
		"transition_size_factor": 0.5,
		"transition_leverage_factor": 0.0,
	},
}

# --------------------------------------------------------------------------------
# Deflated Sharpe Ratio & Walk-Forward Validation Helpers
# --------------------------------------------------------------------------------

MIN_VAL_TRADES = 30  # Minimum trades for a val Sharpe to be meaningful

def _deflated_sharpe_threshold(
    all_sharpes: List[float],
    n_obs: int,
    significance: float = 0.05,
    max_threshold: float = 2.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
    threshold that accounts for multiple testing.

    Given that we tested N groups, the expected maximum Sharpe among
    N i.i.d. draws from N(0, sigma) is approximately:

        E[max(SR)] ~ sigma_SR * [ (1 - gamma) * inv_norm(1 - 1/N_eff)
                                 + gamma * inv_norm(1 - 1/(N_eff*e)) ]

    where gamma ~ 0.5772 (Euler-Mascheroni constant).

    Because all groups share the same market data, features, and model
    architecture, they are highly correlated.  Using the raw trial count N
    produces an absurdly high threshold that rejects everything.  We use
    N_eff = sqrt(N) as the effective number of independent trials (a
    standard correction for correlated tests) and cap the result at
    `max_threshold` to stay in a practically meaningful range.

    Args:
        all_sharpes: Sharpe ratios from ALL groups tested (survivors + rejects).
        n_obs:       Number of observations per val backtest (for SE calculation).
        significance: Significance level (default 0.05 = 95% confidence).
        max_threshold: Hard cap on the DSR threshold (default 2.0).

    Returns:
        Minimum Sharpe that's statistically significant after multiple testing.
    """
    from scipy.stats import norm
    import math

    N = len(all_sharpes)
    if N <= 1:
        return 0.0

    sr_array = np.array(all_sharpes, dtype=float)
    sr_array = sr_array[np.isfinite(sr_array)]
    if len(sr_array) < 2:
        return 0.0

    sigma_sr = sr_array.std()
    if sigma_sr < 1e-8:
        return 0.0

    # Effective independent trials -- groups are correlated (same market,
    # features, architecture), so raw N vastly overstates independence.
    # sqrt(N) is a standard correction for correlated multiple comparisons.
    N_eff = max(math.sqrt(N), 2.0)

    # Expected max Sharpe under the null (all strategies have zero true Sharpe)
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / N_eff)
    z2 = norm.ppf(1.0 - 1.0 / (N_eff * math.e))
    expected_max_sr = sigma_sr * ((1.0 - gamma) * z1 + gamma * z2)

    # Standard error of Sharpe (accounting for non-normality)
    skew = float(pd.Series(sr_array).skew()) if len(sr_array) > 2 else 0.0
    kurt = float(pd.Series(sr_array).kurtosis()) if len(sr_array) > 3 else 0.0
    mean_sr = sr_array.mean()

    se_sr = np.sqrt(
        (1.0 + 0.5 * mean_sr**2 - skew * mean_sr + (kurt / 4.0) * mean_sr**2)
        / max(n_obs, 1)
    )

    # DSR threshold: expected max - z_alpha * SE (lower bound for genuine signal)
    z_alpha = norm.ppf(1.0 - significance)
    threshold = expected_max_sr - z_alpha * se_sr

    # Floor at 0, cap at max_threshold to stay practical
    return max(min(threshold, max_threshold), 0.0)


def _generate_walk_forward_windows(
    base_split: tuple,
    n_windows: int = 3,
    shift_months: int = 2,
) -> List[tuple]:
    """
    Generate multiple walk-forward split configurations by shifting the
    entire train/val/test window backwards in time.

    For a base of (12, 4, 6) with 3 windows and 2-month shift:
      Window 0: (12, 4, 6)  -- most recent (standard)
      Window 1: (12, 4, 6) shifted 2 months earlier
      Window 2: (12, 4, 6) shifted 4 months earlier

    The shift is achieved by growing the test window (which the
    DataPreprocessor discards from the end). The val and train sizes
    stay fixed so the model sees different market conditions.

    Returns list of (train_months, val_months, test_months) tuples.
    Each successive window adds `shift_months` to the test size,
    effectively moving train+val earlier in the dataset.
    """
    if base_split is None:
        base_split = (12, 2, 3)  # default anchored split

    train_m, val_m, test_m = base_split
    windows = []
    # FIX 1: Reverse order so Window 0 gets longest test period (most reliable)
    for i in range(n_windows - 1, -1, -1):
        windows.append((train_m, val_m, test_m + i * shift_months))
    return windows


# --------------------------------------------------------------------------------
# Stage 1: Correlation-based stock clustering
# --------------------------------------------------------------------------------

def cluster_stocks(
    data_path: str,
    n_stocks: int = 100,
    group_size: int = 5,
    min_volume: float = 300_000,
    anchor_end_date: Optional[str] = None,
) -> List[List[str]]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from data_loader import DataLoader
    from lstm_model import get_high_beta_stocks

    n_groups = max(1, n_stocks // group_size)

    print("\n" + "=" * 60)
    print(f"STAGE 1: CORRELATION CLUSTERING ({n_stocks} stocks -> {n_groups} groups)")
    if anchor_end_date:
        print(f"  Data truncated to: {anchor_end_date}")
    else:
        print("  Using full dataset (no date truncation)")
    print("=" * 60)

    pool = get_high_beta_stocks(
        data_path=data_path,
        pool_size=max(n_stocks * 2, 400),
        num_select=n_stocks,
        min_volume=min_volume,
    )
    pool = sorted(pool)
    print(f"\nLoading close price series for {len(pool)} stocks...")

    series_dict: Dict[str, Any] = {}
    for symbol in pool:
        try:
            loader = DataLoader(data_path, symbol)
            fp = loader._find_data_file()
            df = pd.read_csv(fp, header=None, usecols=[0, 4], names=["Datetime", "Close"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df.dropna(subset=["Close"], inplace=True)
            df.set_index("Datetime", inplace=True)
            df = df[~df.index.duplicated(keep="first")]
            if anchor_end_date is not None:
                mask = pd.to_datetime(df.index, errors="coerce") <= pd.Timestamp(anchor_end_date)
                df = df.loc[mask]
            if len(df) > 500:
                series_dict[symbol] = df["Close"]
        except Exception:
            pass

    if len(series_dict) < group_size:
        raise ValueError(f"Too few stocks loaded ({len(series_dict)}) to form groups of {group_size}.")

    available = sorted(series_dict.keys())
    print(f"Successfully loaded {len(available)} price series.")

    price_df   = pd.DataFrame({s: series_dict[s] for s in available})
    price_df   = price_df.ffill().dropna(how="all")
    returns_df = np.log(price_df / price_df.shift(1)).dropna(how="all")
    returns_df = returns_df.dropna(thresh=200, axis=1).ffill().fillna(0)

    corr_matrix = returns_df.corr().fillna(0).values
    tickers_arr = returns_df.columns.tolist()
    n           = len(tickers_arr)

    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = np.array([dist_matrix[i, j] for i in range(n) for j in range(i + 1, n)])

    n_clusters = min(n_groups, n)
    labels     = fcluster(linkage(condensed, method="ward"), t=n_clusters, criterion="maxclust")

    cluster_map: Dict[int, List[str]] = {}
    for ticker, cid in zip(tickers_arr, labels):
        cluster_map.setdefault(cid, []).append(ticker)

    groups: List[List[str]] = []
    for cid in sorted(cluster_map):
        members = cluster_map[cid]
        for start in range(0, len(members), group_size):
            chunk = members[start: start + group_size]
            if len(chunk) >= 2:
                groups.append(chunk)

    print(f"\nFormed {len(groups)} groups from {len(tickers_arr)} clustered stocks.")
    for gid, grp in enumerate(groups):
        print(f"  Group {gid:2d}: {grp}")
    return groups


# --------------------------------------------------------------------------------
# Stage 2: Per-group training + val-set model selection
# --------------------------------------------------------------------------------

def run_tournament(
    groups: List[List[str]],
    selection_threshold: float = 0.0,
    data_path: str = DATA_PATH,
    sentiment_alpha: float = 0.0,
    anchor_end_date: Optional[str] = None,
    use_full_split: bool = False,
    csv_name: str = "results/group_tournament.csv",
    split_months: tuple = None,
    use_min_trades: bool = False,
    use_walk_forward: bool = False,
    max_survivors: int = 15,
    saml_learner: Optional[SAMLMetaLearner] = None,
    quantile_learner: Optional[QuantileBarrierLearner] = None,
) -> List[Dict[str, Any]]:
    """
    Stage 2 tournament with DSR-based selection.

    All modes get:
      - Deflated Sharpe Ratio threshold (corrects for multiple testing)
      - Raw --threshold as floor (DSR raises it if needed)

    --live additionally gets:
      - use_min_trades: reject groups with < 30 val trades
      - use_walk_forward: 3 shifted val windows, require positive in >=2/3
    """
    from lstm_model import train_group
    from ensemble_model import run_val_backtest

    n_walk_forward = 3 if use_walk_forward else 1
    wf_windows = _generate_walk_forward_windows(split_months, n_windows=n_walk_forward)

    print("\n" + "=" * 60)
    print(f"STAGE 2: GROUP TOURNAMENT ({len(groups)} groups)")
    if use_walk_forward:
        print(f"  Walk-forward windows : {n_walk_forward}")
        for i, w in enumerate(wf_windows):
            print(f"    Window {i}: train={w[0]}m / val={w[1]}m / test={w[2]}m")
    else:
        print(f"  Validation window    : {wf_windows[0] if wf_windows else split_months}")
    print(f"  Raw Sharpe threshold : {selection_threshold}")
    print(f"  DSR correction       : applied after all groups evaluated")
    if use_full_split:
        print("  Split mode           : FULL DATA (60/20/20 proportional)")
    if anchor_end_date:
        print(f"  Training data truncated to: {anchor_end_date}")
    if sentiment_alpha > 0.0:
        print(f"  Sentiment alpha      : {sentiment_alpha}")
    print("=" * 60)

    Path("results").mkdir(parents=True, exist_ok=True)
    tournament_rows = []

    # Collect all group results before applying DSR
    group_results = []  # list of (result_dict, sharpe_list, trades_list)

    for group_id, stock_list in enumerate(groups):
        print("\n" + "-" * 60)
        print(f" Training group {group_id}/{len(groups)-1}: {stock_list}")
        print("-" * 60)

        try:
            result = train_group(
                stock_list=stock_list,
                group_id=group_id,
                data_path=data_path,
                lookback_window=LOOKBACK,
                forecast_horizon=HORIZON,
                anchor_end_date=anchor_end_date,
                use_full_split=use_full_split,
                split_months=split_months,
                saml_learner=saml_learner,
                quantile_learner=quantile_learner,
            )
        except Exception as e:
            print(f"  Group {group_id} training failed: {e}")
            tournament_rows.append({
                "group_id": group_id, "stocks": ",".join(stock_list),
                "val_accuracy": "FAILED", "val_sharpe": "FAILED",
                "overall_accuracy": "FAILED", "acc_55": "FAILED",
                "trades_55": "FAILED", "acc_60": "FAILED",
                "trades_60": "FAILED", "acc_65": "FAILED",
                "trades_65": "FAILED", "ticker_positive_ratio": "0.00",
                "selected": "N",
            })
            gc.collect()
            continue

        # -- Walk-forward validation across multiple windows ----------
        window_sharpes = []
        window_trades  = []
        primary_ticker_ratio = 1.0
        for w_idx, w_split in enumerate(wf_windows):
            try:
                val_bt = run_val_backtest(
                    tickers=result["stocks"],
                    group_id=group_id,
                    lookback=LOOKBACK,
                    horizon=HORIZON,
                    data_path=data_path,
                    sentiment_alpha=sentiment_alpha,
                    anchor_end_date=anchor_end_date,
                    use_full_split=use_full_split,
                    split_months=w_split,
                )
                ws = val_bt["val_sharpe"]
                wt = val_bt["val_trades"]
                window_sharpes.append(ws)
                window_trades.append(wt)
                if w_idx == 0:
                    primary_ticker_ratio = val_bt.get("val_ticker_positive_ratio", 1.0)
                print(f"    Window {w_idx} (split {w_split}): "
                      f"Sharpe={ws:.3f}, Trades={wt}")
            except Exception as e:
                print(f"    Window {w_idx} failed: {e}")
                window_sharpes.append(-999.0)
                window_trades.append(0)

        # Use the primary window (most recent) as the headline Sharpe
        result["val_sharpe"] = window_sharpes[0] if window_sharpes else -999.0
        result["val_trades"] = window_trades[0] if window_trades else 0
        result["window_sharpes"] = window_sharpes
        result["window_trades"]  = window_trades
        result["val_ticker_positive_ratio"] = primary_ticker_ratio

        group_results.append(result)

        tournament_rows.append({
            "group_id":         group_id,
            "stocks":           ",".join(result["stocks"]),
            "val_accuracy":     f"{result['val_accuracy']:.4f}",
            "val_sharpe":       f"{result['val_sharpe']:.3f}",
            "overall_accuracy": f"{result.get('overall_accuracy', 0):.4f}",
            "acc_55":           f"{result.get('acc_55', 0):.4f}",
            "trades_55":        result.get("trades_55", 0),
            "acc_60":           f"{result.get('acc_60', 0):.4f}",
            "trades_60":        result.get("trades_60", 0),
            "acc_65":           f"{result.get('acc_65', 0):.4f}",
            "trades_65":        result.get("trades_65", 0),
            "ticker_positive_ratio": f"{result.get('val_ticker_positive_ratio', 1.0):.2f}",
            "selected":         "?",  # deferred until DSR computed
        })
        gc.collect()

    # -- Post-tournament selection: DSR only --------------------------------
    all_primary_sharpes = [r["val_sharpe"] for r in group_results
                          if r["val_sharpe"] > -900]
    avg_trades = np.mean([r["val_trades"] for r in group_results
                         if r["val_trades"] > 0]) if group_results else 100

    dsr_threshold = _deflated_sharpe_threshold(
        all_sharpes=all_primary_sharpes,
        n_obs=int(avg_trades),
    )
    effective_threshold = max(selection_threshold, dsr_threshold)

    import math as _math
    _n_eff = max(_math.sqrt(len(all_primary_sharpes)), 2.0)

    print("\n" + "=" * 60)
    print("POST-TOURNAMENT SELECTION")
    print(f"  Groups evaluated     : {len(group_results)}")
    print(f"  Raw --threshold      : {selection_threshold:.3f}")
    print(f"  DSR threshold        : {dsr_threshold:.3f}  (N={len(all_primary_sharpes)}, N_eff={_n_eff:.1f})")
    print(f"  Effective threshold  : {effective_threshold:.3f}")
    if use_min_trades:
        print(f"  Min val trades       : {MIN_VAL_TRADES}")
    if use_walk_forward:
        print(f"  Walk-forward windows : {n_walk_forward} (require positive in >={n_walk_forward - 1})")
    print("=" * 60)

    survivors = []
    for i, result in enumerate(group_results):
        passes_sharpe = result["val_sharpe"] > effective_threshold

        trades = result.get("val_trades", 0)
        passes_trades = trades >= MIN_VAL_TRADES if use_min_trades else True

        sharpes = result.get("window_sharpes", [result["val_sharpe"]])
        if use_walk_forward:
            n_positive = sum(1 for s in sharpes if s > 0.0)
            required_positive = max(n_walk_forward - 1, 1)
            passes_stability = n_positive >= required_positive
        else:
            passes_stability = True
            n_positive = len(sharpes)

        ticker_ratio = result.get("val_ticker_positive_ratio", 1.0)
        passes_ticker_quality = ticker_ratio >= 0.6

        # -----------------------------------------------------------------
        # Confidence-tail integrity checks (val-only data -- no test leakage).
        # Three patterns that empirically predict negative test return despite
        # high val Sharpe:
        #   1. acc_65 < 0.50 with non-trivial trade count   -> model is
        #      anti-predictive on its highest-confidence calls (the high
        #      Sharpe was masked by a few lucky big trades).
        #   2. acc_65 < acc_55                              -> accuracy doesn't
        #      improve with confidence -> model isn't really discriminating.
        #   3. val_sharpe > 3.5 AND acc_55 < 0.50            -> Sharpe driven
        #      by outliers rather than base accuracy -> overfit Sharpe.
        # -----------------------------------------------------------------
        def _as_float(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        acc_65_f    = _as_float(result.get("acc_65"))
        acc_55_f    = _as_float(result.get("acc_55"))
        trades_65_v = result.get("trades_65", 0)
        if not isinstance(trades_65_v, (int, float)) or trades_65_v < 0:
            trades_65_v = 0

        passes_high_conf_acc      = True
        passes_monotonic_conf     = True
        passes_sharpe_consistency = True

        if trades_65_v >= 20 and acc_65_f is not None:
            if acc_65_f < 0.50:
                passes_high_conf_acc = False
            if acc_55_f is not None and acc_65_f < acc_55_f:
                passes_monotonic_conf = False

        if (acc_55_f is not None
                and result["val_sharpe"] > 3.5
                and acc_55_f < 0.50):
            passes_sharpe_consistency = False

        selected = (passes_sharpe and passes_trades and passes_stability
                    and passes_ticker_quality and passes_high_conf_acc
                    and passes_monotonic_conf and passes_sharpe_consistency)

        reason = []
        if not passes_sharpe:
            reason.append(f"Sharpe {result['val_sharpe']:.3f} < {effective_threshold:.3f}")
        if not passes_trades:
            reason.append(f"Trades {trades} < {MIN_VAL_TRADES}")
        if not passes_stability:
            reason.append(f"Stable in {n_positive}/{n_walk_forward} windows (need {required_positive})")
        if not passes_ticker_quality:
            reason.append(f"Ticker quality {ticker_ratio*100:.0f}% < 60%")
        if not passes_high_conf_acc:
            reason.append(f"acc_65 {acc_65_f:.3f} <0.50 on {int(trades_65_v)} hi-conf trades (anti-signal)")
        if not passes_monotonic_conf:
            reason.append(f"acc_65 {acc_65_f:.3f} < acc_55 {acc_55_f:.3f} (confidence not discriminating)")
        if not passes_sharpe_consistency:
            reason.append(f"Sharpe {result['val_sharpe']:.2f} >3.5 but acc_55 {acc_55_f:.3f} <0.50 (overfit Sharpe)")

        status = "Y" if selected else "N"
        print(f"  Group {result.get('group_id', i):2d}: "
              f"Sharpe={result['val_sharpe']:+.3f}  "
              f"Trades={trades:4d}  "
              f"-> {'SELECTED' if selected else 'REJECTED: ' + '; '.join(reason)}")

        if selected:
            survivors.append(result)

        # Update the deferred "selected" field in tournament_rows
        # Find the matching row by group_id
        for row in tournament_rows:
            if row["group_id"] == result.get("group_id", i):
                row["selected"] = status
                break

    print(f"\n  Survivors after DSR/WF: {len(survivors)} / {len(group_results)} groups")

    # Cap survivors to top N by val Sharpe
    if max_survivors and len(survivors) > max_survivors:
        print(f"\nCapping survivors to top {max_survivors} by val Sharpe...")
        survivors.sort(key=lambda r: r["val_sharpe"], reverse=True)
        dropped = survivors[max_survivors:]
        survivors = survivors[:max_survivors]
        for d in dropped:
            print(f"  Dropped (cap): Group {d['group_id']} Sharpe={d['val_sharpe']:+.3f}")
        # Update tournament_rows for dropped groups
        for d in dropped:
            gid = d.get("group_id")
            for row in tournament_rows:
                if row["group_id"] == gid:
                    row["selected"] = "N"
                    break

    print(f"  Final survivors: {len(survivors)} / {len(group_results)} groups")

    _write_tournament_csv(tournament_rows, csv_path=csv_name)
    survivors = _write_manifest(survivors)
    return survivors


# --------------------------------------------------------------------------------
# Stage 2b: MTL tournament
# --------------------------------------------------------------------------------

def run_tournament_mtl(
    groups: List[List[str]],
    selection_threshold: float = 0.0,
    data_path: str = DATA_PATH,
    sentiment_alpha: float = 0.0,
    anchor_end_date: Optional[str] = None,
    use_full_split: bool = False,
    csv_name: str = "results/group_tournament.csv",
    max_survivors: int = 15,
) -> tuple:
    """
    Returns (survivors, mtl_model) so the caller can pass the live model
    straight into Stage 3 without reloading from disk.
    """
    from mtl_lstm_model import MTLLSTMModel, MTLGroupAdapter
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from lstm_model import make_direction_onehot_from_raw
    from ensemble_model import _make_preprocessor, run_val_backtest
    import gc

    print("\n" + "=" * 60)
    print(f"STAGE 2 (MTL): SHARED/PRIVATE TOURNAMENT ({len(groups)} groups)")
    print(f"  Selection metric : val_sharpe > {selection_threshold}")
    if use_full_split:
        print("  Split mode       : FULL DATA (60/20/20 proportional)")
    if anchor_end_date:
        print(f"  Training data truncated to: {anchor_end_date}")
    print("=" * 60)

    Path("results").mkdir(parents=True, exist_ok=True)
    mtl_dir = Path("models/mtl")
    mtl_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect per-group data ----------------------------------------
    group_data: Dict[int, tuple] = {}
    group_valid_tickers: Dict[int, List[str]] = {}
    n_features: Optional[int] = None

    for group_id, stock_list in enumerate(groups):
        print(f"\n  Collecting data for group {group_id}: {stock_list}")
        X_tr, y_tr, X_vl, y_vl = [], [], [], []
        valid_tickers = []

        for ticker in stock_list:
            try:
                loader   = DataLoader(data_path, ticker)
                raw_df   = loader.load_data()
                enriched = FeatureEngineer().compute_indicators(raw_df)
                pre, splits, enriched_trunc = _make_preprocessor(
                    LOOKBACK, HORIZON, enriched, anchor_end_date,
                    use_full_split=use_full_split,
                )
                labels, masks, _ = make_direction_onehot_from_raw(
                    enriched_df=enriched_trunc,
                    split_indices={
                        "idx_train": splits["idx_train"],
                        "idx_val":   splits["idx_val"],
                        "idx_test":  splits["idx_test"],
                    },
                    lookback_window=LOOKBACK,
                    forecast_horizon=HORIZON,
                    embargo=HORIZON,  # FIX 3: Enable purge/embargo (Lopez de Prado)
                )
                X_tr.append(splits["X_train"][masks["train"]])
                y_tr.append(labels["train"][masks["train"]])  # FIX 3: Apply purge mask
                X_vl.append(splits["X_val"][masks["val"]])
                y_vl.append(labels["val"][masks["val"]])      # FIX 3: Apply purge mask
                valid_tickers.append(ticker)
                if n_features is None:
                    n_features = splits["X_train"].shape[2]
            except Exception as e:
                print(f"    Skipping {ticker}: {e}")
                continue

        if X_tr and X_vl:
            group_data[group_id] = (
                np.concatenate(X_tr),
                np.concatenate(y_tr),
                np.concatenate(X_vl),
                np.concatenate(y_vl),
            )
            group_valid_tickers[group_id] = valid_tickers

        gc.collect()

    if not group_data or n_features is None:
        raise ValueError("MTL tournament: no valid groups collected.")

    # ---- Train MTL model -----------------------------------------------
    print(f"\n[MTL] Training shared trunk + {len(group_data)} private heads...")
    mtl = MTLLSTMModel(group_ids=list(group_data.keys()))
    mtl.build_model(input_shape=(LOOKBACK, n_features))
    mtl.train(group_data=group_data, save_dir=str(mtl_dir))
    mtl.save(str(mtl_dir))
    print(f"[MTL] Model saved to {mtl_dir}")

    # ---- Per-group val backtest using private heads --------------------
    tournament_rows = []
    survivors       = []

    for group_id, tickers in group_valid_tickers.items():
        group_dir = Path(f"models/group_{group_id}")
        group_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(
            str(mtl_dir / "mtl_model.weights.h5"),
            str(group_dir / "lstm_model.weights.h5"),
        )

        try:
            val_bt = run_val_backtest(
                tickers=tickers,
                group_id=group_id,
                lookback=LOOKBACK,
                horizon=HORIZON,
                data_path=data_path,
                sentiment_alpha=sentiment_alpha,
                anchor_end_date=anchor_end_date,
                use_full_split=use_full_split,
                mtl_model=mtl,
            )
            val_sharpe = val_bt["val_sharpe"]
        except Exception as e:
            print(f"  Group {group_id} val backtest failed: {e}")
            val_sharpe = -999.0

        selected = val_sharpe > selection_threshold
        result = {
            "group_id":     group_id,
            "stocks":       tickers,
            "val_accuracy": 0.0,
            "val_sharpe":   val_sharpe,
            "model_path":   str(group_dir / "lstm_model.weights.h5"),
            "n_features":   n_features,
        }
        if selected:
            survivors.append(result)

        tournament_rows.append({
            "group_id":         group_id,
            "stocks":           ",".join(tickers),
            "val_accuracy":     "MTL",
            "val_sharpe":       f"{val_sharpe:.3f}",
            "overall_accuracy": "MTL",
            "acc_55": "MTL", "trades_55": "MTL",
            "acc_60": "MTL", "trades_60": "MTL",
            "acc_65": "MTL", "trades_65": "MTL",
            "ticker_positive_ratio": "1.00",
            "selected": "Y" if selected else "N",
        })
        gc.collect()

    # Cap survivors to top N by val Sharpe
    if max_survivors and len(survivors) > max_survivors:
        print(f"\nCapping MTL survivors to top {max_survivors} by val Sharpe...")
        survivors.sort(key=lambda r: r["val_sharpe"], reverse=True)
        dropped = survivors[max_survivors:]
        survivors = survivors[:max_survivors]
        for d in dropped:
            print(f"  Dropped (cap): Group {d['group_id']} Sharpe={d['val_sharpe']:+.3f}")
        for d in dropped:
            gid = d.get("group_id")
            for row in tournament_rows:
                if row["group_id"] == gid:
                    row["selected"] = "N"
                    break

    _write_tournament_csv(tournament_rows, csv_path=csv_name)
    survivors = _write_manifest(survivors)
    return survivors, mtl


# --------------------------------------------------------------------------------
# Shared tournament helpers
# --------------------------------------------------------------------------------

def _write_tournament_csv(
    rows: List[Dict],
    csv_path: str = "results/group_tournament.csv",
) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_id", "stocks", "val_accuracy", "val_sharpe",
        "overall_accuracy", "acc_55", "trades_55",
        "acc_60", "trades_60", "acc_65", "trades_65",
        "ticker_positive_ratio", "selected",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"{'Grp':>4}  {'ValAcc':>7}  {'ValShrp':>7}  {'OvrAcc':>7}  "
          f"{'Acc@55':>7}  {'Tr@55':>6}  {'Acc@60':>7}  {'Tr@60':>6}  "
          f"{'Acc@65':>7}  {'Tr@65':>6}  {'Sel':>4}  Stocks")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['group_id']:>4}  {row['val_accuracy']:>7}  {row['val_sharpe']:>7}  "
            f"{row['overall_accuracy']:>7}  {row['acc_55']:>7}  {str(row['trades_55']):>6}  "
            f"{row['acc_60']:>7}  {str(row['trades_60']):>6}  "
            f"{row['acc_65']:>7}  {str(row['trades_65']):>6}  "
            f"{row['selected']:>4}  {row['stocks']}"
        )
    print("-" * 100)
    survivors_count = sum(1 for r in rows if r["selected"] == "Y")
    print(f"Survivors : {survivors_count} / {len(rows)}")
    print(f"Table saved to {out}")


def _write_manifest(survivors: List[Dict]) -> List[Dict]:
    manifest_path = Path("models/master_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "test_data"} for r in survivors],
            f, indent=2,
        )
    print(f"Survivor manifest saved to {manifest_path}")
    return survivors


def _load_survivors_from_csv(csv_path: str, selection_threshold: float) -> List[Dict[str, Any]]:
    """
    Reconstruct a survivors list from an existing tournament CSV.
    Used by --compare to skip re-training the LSTM arm.
    """
    survivors = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    val_sharpe = float(row["val_sharpe"])
                except (ValueError, TypeError):
                    continue
                if val_sharpe > selection_threshold:
                    survivors.append({
                        "group_id":     int(row["group_id"]),
                        "stocks":       row["stocks"].split(","),
                        "val_accuracy": 0.0,
                        "val_sharpe":   val_sharpe,
                    })
    except FileNotFoundError:
        pass
    return survivors


# --------------------------------------------------------------------------------
# Stage 3: Combined portfolio backtest
# --------------------------------------------------------------------------------

def run_combined_backtest(
    survivors: List[Dict[str, Any]],
    data_path: str = DATA_PATH,
    anchor_end_date: Optional[str] = None,
    equity_curve_name: str = "equity_curve.png",
    sentiment_alpha: float = 0.0,
    train_anchor_date: Optional[str] = None,
    use_full_split: bool = False,
    mtl_model=None,
    split_months: tuple = None,
    glm_guard=None,
    oos_holdout_pct: float = 0.0,
    use_regime_routing: bool = False,
    saml_meta_learner=None,
    quantile_learner=None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    train_anchor_date : str, optional
        The same anchor date used for training (e.g. BEAR_TRAIN_END).
        The RegimeDetector is fitted ONLY on data up to this date to prevent
        future regime leakage into the HMM.  If None, uses all available data
        up to anchor_end_date (or all data for default / full runs).
    use_full_split : bool
        When True (--full mode) uses the 60/20/20 proportional split and
        raises MIN_TEST_HOURS to MIN_TEST_HOURS_FULL so tiny tickers are
        still filtered correctly.
    mtl_model : optional
        If provided, passes the live MTL model to run_ensemble() so that
        per-group private heads are used instead of loading individual LSTM
        weights from disk.
    use_regime_routing : bool, optional
        If True, enables regime-aware routing using trained specialists.
        Requires specialist models to be trained via --train_all_specialists.
    saml_meta_learner : optional
        SAML meta-learner for specialist initialization.
    quantile_learner : optional
        Quantile barrier learner for adaptive thresholds.
    """
    from ensemble_model import run_ensemble, run_ensemble_regime_aware
    from PredictionEngine import PortfolioEngine
    from feature_engineer import FeatureEngineer
    from data_loader import DataLoader
    from regime_detector import RegimeDetector

    if use_full_split:
        min_test_hours = MIN_TEST_HOURS_FULL
    elif split_months is not None:
        # For custom splits (e.g. --live 13/5/6), derive min from test months
        min_test_hours = int(split_months[2] * 730 * 0.10)  # 10% of test window
    else:
        min_test_hours = MIN_TEST_HOURS

    print("\n" + "=" * 60)
    print(f"STAGE 3: COMBINED BACKTEST ({len(survivors)} surviving groups)")
    if use_full_split:
        print("  Split mode     : FULL DATA (60/20/20 proportional)")
        print(f"  Min test hours : {min_test_hours}")
    if mtl_model is not None:
        print("  Model type     : Shared/Private MTL LSTM")
    if anchor_end_date:
        print(f"  Test window end: {anchor_end_date}")
    if sentiment_alpha > 0.0:
        print(f"  Sentiment alpha: {sentiment_alpha}")
    print("=" * 60)

    all_prices, all_probs, all_votes = [], [], []
    all_long_g, all_short_g, all_atr_vol = [], [], []
    all_enriched, all_row_idx = [], []
    all_ticker_names = []     # parallel to all_prices -- one name per column
    all_group_ids    = []     # parallel to all_ticker_names -- group id per column
    # GLM enrichment lists (parallel to all_prices)
    all_lstm_probs, all_rf_probs, all_xgb_probs = [], [], []
    all_sent_scores, all_sent_mag, all_sent_vol = [], [], []
    all_headlines = []        # list of list-of-lists (per ticker, per timestep, headlines)

    # ------------------------------------------------------------
    # FIX: Initialize regime variables before signal generation loop
    # (used by run_ensemble_regime_aware when use_regime_routing is True)
    # Will be recomputed properly after signal collection
    # ------------------------------------------------------------
    portfolio_regime = np.full(5000, "BULL", dtype=object)  # Large enough buffer, will slice later
    regime_confidence = {i: {"BULL": 1.0, "BEAR": 0.0, "CRISIS": 0.0} for i in range(5000)}

    for result in survivors:
        group_id = result["group_id"]
        tickers = result["stocks"]
        print(f"\nGenerating signals for group {group_id}: {tickers}")
        try:
            # Use regime-aware routing if enabled and specialists available
            if use_regime_routing:
                print(f" [Regime Routing] Using specialist models for group {group_id}")
                signals = run_ensemble_regime_aware(
                    tickers=tickers,
                    group_id=group_id,
                    regime_series=portfolio_regime,
                    regime_confidence=regime_confidence,
                    lookback=LOOKBACK,
                    horizon=HORIZON,
                    data_path=data_path,
                    anchor_end_date=anchor_end_date,
                    sentiment_alpha=sentiment_alpha,
                    use_full_split=use_full_split,
                    mtl_model=mtl_model,
                    split_months=split_months,
                )
            else:
                signals = run_ensemble(
                    tickers=tickers,
                    group_id=group_id,
                    lookback=LOOKBACK,
                    horizon=HORIZON,
                    data_path=data_path,
                    anchor_end_date=anchor_end_date,
                    sentiment_alpha=sentiment_alpha,
                    use_full_split=use_full_split,
                    mtl_model=mtl_model,
                    split_months=split_months,
                )
        except Exception as e:
            print(f"  Group {group_id} signal generation failed: {e}")
            continue

        all_prices.extend(signals["prices"])
        all_probs.extend(signals["probs"])
        all_votes.extend(signals["votes"])
        all_long_g.extend(signals["long_g"])
        all_short_g.extend(signals["short_g"])
        all_atr_vol.extend(signals["atr_vol"])
        all_enriched.extend(signals["enriched"])
        all_row_idx.extend(signals["row_idx"])
        all_ticker_names.extend(tickers)
        all_group_ids.extend([group_id] * len(tickers))
        # GLM enrichment
        all_lstm_probs.extend(signals.get("lstm_probs", []))
        all_rf_probs.extend(signals.get("rf_probs", []))
        all_xgb_probs.extend(signals.get("xgb_probs", []))
        all_sent_scores.extend(signals.get("sentiment_scores", []))
        all_sent_mag.extend(signals.get("sentiment_mag", []))
        all_sent_vol.extend(signals.get("sentiment_vol", []))
        all_headlines.extend(signals.get("headlines_text", []))
        gc.collect()

    if not all_prices:
        raise ValueError("No signal data collected from surviving groups.")

    indices_keep = [i for i, p in enumerate(all_prices) if len(p) >= min_test_hours]
    n_dropped    = len(all_prices) - len(indices_keep)
    if n_dropped:
        print(f"\nDropping {n_dropped} ticker(s) with test window < {min_test_hours} h")
    all_prices       = [all_prices[i]       for i in indices_keep]
    all_probs        = [all_probs[i]        for i in indices_keep]
    all_votes        = [all_votes[i]        for i in indices_keep]
    all_long_g       = [all_long_g[i]       for i in indices_keep]
    all_short_g      = [all_short_g[i]      for i in indices_keep]
    all_atr_vol      = [all_atr_vol[i]      for i in indices_keep]
    all_enriched     = [all_enriched[i]     for i in indices_keep]
    all_row_idx      = [all_row_idx[i]      for i in indices_keep]
    all_ticker_names = [all_ticker_names[i] for i in indices_keep] if all_ticker_names else []
    all_group_ids    = [all_group_ids[i]    for i in indices_keep] if all_group_ids    else []
    # GLM enrichment
    if all_lstm_probs:
        all_lstm_probs  = [all_lstm_probs[i]  for i in indices_keep]
        all_rf_probs    = [all_rf_probs[i]    for i in indices_keep]
        all_xgb_probs   = [all_xgb_probs[i]  for i in indices_keep]
        all_sent_scores = [all_sent_scores[i] for i in indices_keep]
        all_sent_mag    = [all_sent_mag[i]    for i in indices_keep]
        all_sent_vol    = [all_sent_vol[i]    for i in indices_keep]
        all_headlines   = [all_headlines[i]   for i in indices_keep]

    if not all_prices:
        raise ValueError(
            f"All tickers dropped (test window < {min_test_hours} h). "
            "Lower MIN_TEST_HOURS or use more data."
        )

    lengths = [len(p) for p in all_prices]
    min_len = int(np.percentile(lengths, 10))
    indices_keep = [i for i, l in enumerate(lengths) if l >= min_len]
    n_dropped = len(all_prices) - len(indices_keep)
    if n_dropped:
        print(f"\n {n_dropped} ticker(s) below 10th-percentile length dropped from alignment")
        # Filter all lists to remove short timelines
        all_prices = [all_prices[i] for i in indices_keep]
        all_probs = [all_probs[i] for i in indices_keep]
        all_votes = [all_votes[i] for i in indices_keep]
        all_long_g = [all_long_g[i] for i in indices_keep]
        all_short_g = [all_short_g[i] for i in indices_keep]
        all_atr_vol = [all_atr_vol[i] for i in indices_keep]
        all_enriched = [all_enriched[i] for i in indices_keep]
        all_row_idx = [all_row_idx[i] for i in indices_keep]
        all_ticker_names = [all_ticker_names[i] for i in indices_keep] if all_ticker_names else []
        all_group_ids    = [all_group_ids[i]    for i in indices_keep] if all_group_ids    else []
        # GLM enrichment lists
        if all_lstm_probs:
            all_lstm_probs = [all_lstm_probs[i] for i in indices_keep]
            all_rf_probs = [all_rf_probs[i] for i in indices_keep]
            all_xgb_probs = [all_xgb_probs[i] for i in indices_keep]
            all_sent_scores = [all_sent_scores[i] for i in indices_keep]
            all_sent_mag = [all_sent_mag[i] for i in indices_keep]
            all_sent_vol = [all_sent_vol[i] for i in indices_keep]
            all_headlines = [all_headlines[i] for i in indices_keep]
    print(f"\nAligning {len(all_prices)} ticker timelines to {min_len} shared hours...")

    price_matrix   = np.column_stack([p[-min_len:] for p in all_prices])
    prob_matrix    = np.column_stack([p[-min_len:] for p in all_probs])
    long_g_matrix  = np.column_stack([g[-min_len:] for g in all_long_g])
    short_g_matrix = np.column_stack([g[-min_len:] for g in all_short_g])
    vol_matrix     = np.column_stack([v[-min_len:] for v in all_atr_vol])

    vol_p95         = np.percentile(vol_matrix, VOL_REGIME_PERCENTILE, axis=0)
    high_vol_regime = vol_matrix > vol_p95[np.newaxis, :]
    print(f"Vol-regime halt active on {high_vol_regime.mean()*100:.1f}% of timesteps")

    # ------------------------------------------------------------
    # FIX 5: Build per-ticker val Sharpe weight matrix
    # ------------------------------------------------------------
    sharpes_positive = [r["val_sharpe"] for r in survivors if r["val_sharpe"] > 0]
    if sharpes_positive:
        sharpes_sum = sum(sharpes_positive)
        group_weights = {r["group_id"]: r["val_sharpe"] / sharpes_sum for r in survivors if r["val_sharpe"] > 0}
    else:
        group_weights = {r["group_id"]: 1.0 / len(survivors) for r in survivors}

    ticker_weight_map = {}
    for result in survivors:
        gid = result["group_id"]
        w = group_weights.get(gid, 0.0)
        for tkr in result["stocks"]:
            ticker_weight_map[tkr] = w

    ticker_weights = np.array([ticker_weight_map.get(name, 0.0) for name in all_ticker_names])
    if ticker_weights.sum() > 0:
        ticker_weights = ticker_weights / ticker_weights.sum() * len(survivors)
    weight_matrix = np.tile(ticker_weights, (min_len, 1))
    print(f"[Signal Weight] Range: {weight_matrix.min():.3f} - {weight_matrix.max():.3f}")

    # ------------------------------------------------------------
    # FIX A+D: Train and save the RegimeDetector on the TRAINING window
    # (anchor-bounded to prevent future regime leakage into the HMM).
    # This must run before the regime-adaptive signal matrix is built.
    # ------------------------------------------------------------
    detector_path    = Path("models/regime_detector.pkl")
    # Use train_anchor_date if provided (regime runs), else anchor_end_date
    # (which is None for default / full runs -- in that case we fit on all data
    # available in all_enriched, which is already the truncated set).
    hmm_fit_cutoff   = train_anchor_date or anchor_end_date
    portfolio_regime = np.full(min_len, "BULL", dtype=object)

    print("\n" + "-" * 60)
    print("[RegimeDetector] Fitting HMM on training-window enriched data...")

    train_enriched_dfs = []
    for enriched_df in all_enriched:
        try:
            df_fit = enriched_df
            if hmm_fit_cutoff is not None:
                mask = pd.to_datetime(enriched_df.index, errors="coerce") <= pd.Timestamp(hmm_fit_cutoff)
                df_fit = enriched_df.loc[mask]
            if len(df_fit) > 200:
                train_enriched_dfs.append(df_fit)
        except Exception:
            pass

    if train_enriched_dfs:
        try:
            detector = RegimeDetector.fit_and_save(
                enriched_dfs=train_enriched_dfs,
                save_path=str(detector_path),
            )
        except Exception as e:
            print(f"[RegimeDetector] fit_and_save failed ({e}); will default to BULL.")
            detector = None
    else:
        print("[RegimeDetector] No valid enriched DataFrames for HMM fitting; defaulting to BULL.")
        detector = None

    # Per-timestep regime confidence for GLM (default: full BULL confidence)
    regime_confidence = {i: {"BULL": 1.0, "BEAR": 0.0, "CRISIS": 0.0} for i in range(min_len)}

    # ------------------------------------------------------------
    # Regime detection: majority-vote across tickers at each timestep
    # ------------------------------------------------------------
    if detector is not None:
        try:
            regime_series_list = []
            regime_probs_list  = []
            for enriched_df, row_idx in zip(all_enriched, all_row_idx):
                aligned_idx = row_idx[-min_len:] if len(row_idx) >= min_len else row_idx
                if glm_guard is not None:
                    ticker_regimes, ticker_probs = detector.predict_regime_series_with_confidence(
                        enriched_df.iloc[aligned_idx]
                    )
                    regime_probs_list.append(ticker_probs[-min_len:])
                else:
                    ticker_regimes = detector.predict_regime_series(
                        enriched_df.iloc[aligned_idx]
                    )
                regime_series_list.append(ticker_regimes[-min_len:])

            regime_matrix = np.array(regime_series_list)  # (N_tickers, min_len)
            for t in range(min_len):
                col    = regime_matrix[:, t]
                unique, counts = np.unique(col, return_counts=True)
                portfolio_regime[t] = unique[np.argmax(counts)]

            # Average regime probabilities across tickers for GLM
            if regime_probs_list:
                for t in range(min_len):
                    avg_probs = {"BULL": 0.0, "BEAR": 0.0, "CRISIS": 0.0}
                    n = len(regime_probs_list)
                    for ticker_probs in regime_probs_list:
                        if t < len(ticker_probs):
                            for label in avg_probs:
                                avg_probs[label] += ticker_probs[t].get(label, 0.0)
                    for label in avg_probs:
                        avg_probs[label] = round(avg_probs[label] / max(n, 1), 3)
                    regime_confidence[t] = avg_probs

            unique_r, counts_r = np.unique(portfolio_regime, return_counts=True)
            print("\nRegime distribution over test window:")
            for r, c in zip(unique_r, counts_r):
                print(f"  {r}: {c} timesteps ({c/min_len*100:.1f}%)")
        except Exception as e:
            print(f"[RegimeDetector] Prediction failed -- defaulting to BULL. ({e})")
            portfolio_regime = np.full(min_len, "BULL", dtype=object)

    # Persist portfolio_regime for post-hoc audit
    regime_npy_path = Path("results/portfolio_regime.npy")
    regime_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(regime_npy_path), portfolio_regime)
    print(f"[RegimeDetector] Regime series saved to {regime_npy_path}")

    # ------------------------------------------------------------
    # Build signal matrix with per-timestep regime-adaptive thresholds
    # ------------------------------------------------------------
    signal_matrix = np.zeros_like(prob_matrix, dtype=int)
    for t in range(min_len):
        regime    = portfolio_regime[t]
        cfg       = REGIME_CONFIG[regime]
        long_thr  = cfg["long_threshold"]
        short_thr = cfg["short_threshold"]
        for i in range(prob_matrix.shape[1]):
            prob = prob_matrix[t, i]
            w = weight_matrix[t, i]
            # Boost probability for high-quality signals (BULL only)
            if regime == "BULL":
                prob_boosted = np.clip(prob * (1.0 + w * 0.3), 0.0, 1.0)
            else:
                # BEAR/CRISIS: no probability boost
                prob_boosted = prob
            if regime == "BULL":
                # BULL: no vol filter -- sweep showed removing vol gating in
                # bull regimes improves Sharpe without increasing drawdown.
                if prob_boosted >= long_thr:
                    signal_matrix[t, i] = 1
                elif prob_boosted <= short_thr:
                    signal_matrix[t, i] = -1
            else:
                # BEAR/CRISIS: keep all vol filters to protect capital
                if prob_boosted >= long_thr and long_g_matrix[t, i] and not high_vol_regime[t, i]:
                    signal_matrix[t, i] = 1
                elif prob_boosted <= short_thr and short_g_matrix[t, i]:
                    signal_matrix[t, i] = -1

    # Cap concurrent positions per timestep to top 8 by conviction
    MAX_CONCURRENT = 8
    for t in range(min_len):
        active = np.where(signal_matrix[t] != 0)[0]
        if len(active) > MAX_CONCURRENT:
            convictions = np.abs(prob_matrix[t, active] - 0.5)
            keep_idx = np.argsort(convictions)[-MAX_CONCURRENT:]
            drop_idx = np.setdiff1d(np.arange(len(active)), keep_idx)
            for di in drop_idx:
                signal_matrix[t, active[di]] = 0
    print(f"\n[Position Cap] Max concurrent positions capped at {MAX_CONCURRENT}")

    # Per-regime signal audit (helps verify BEAR/CRISIS configs are firing)
    print("\nSignal breakdown by regime:")
    for regime_label in ["BULL", "BEAR", "CRISIS"]:
        mask_t = portfolio_regime == regime_label
        if mask_t.sum() == 0:
            continue
        sig_slice = signal_matrix[mask_t, :]
        print(f"  {regime_label:6s}: "
              f"long={np.sum(sig_slice==1):5d}  "
              f"short={np.sum(sig_slice==-1):5d}  "
              f"flat={np.sum(sig_slice==0):6d}  "
              f"({mask_t.sum()} timesteps)")

    print(f"\nTotal -- Long: {np.sum(signal_matrix==1)} | "
          f"Short: {np.sum(signal_matrix==-1)} | "
          f"Flat: {np.sum(signal_matrix==0)}")

    # ------------------------------------------------------------
    # GLM enrichment matrices (only built when --glm is active)
    # ------------------------------------------------------------
    glm_context = None
    if glm_guard is not None and all_lstm_probs:
        lstm_prob_matrix  = np.column_stack([p[-min_len:] for p in all_lstm_probs])
        rf_prob_matrix    = np.column_stack([p[-min_len:] for p in all_rf_probs])
        xgb_prob_matrix   = np.column_stack([p[-min_len:] for p in all_xgb_probs])
        sent_score_matrix = np.column_stack([s[-min_len:] for s in all_sent_scores])
        sent_mag_matrix   = np.column_stack([s[-min_len:] for s in all_sent_mag])
        sent_vol_matrix   = np.column_stack([s[-min_len:] for s in all_sent_vol])
        # Extract key technical indicators from enriched DataFrames
        tech_indicators = {}
        for col_idx, (enriched_df, row_idx) in enumerate(zip(all_enriched, all_row_idx)):
            aligned_idx = row_idx[-min_len:]
            for t_offset, raw_idx in enumerate(aligned_idx):
                if raw_idx < len(enriched_df):
                    row = enriched_df.iloc[raw_idx]
                    tech_indicators[(col_idx, t_offset)] = {
                        "RSI_14":         round(float(row.get("RSI_14", 0)), 1),
                        "MACD_hist":      round(float(row.get("MACD_hist", 0)), 4),
                        "BB_PctB":        round(float(row.get("BB_PctB", 0)), 2),
                        "vol_percentile": round(float(row.get("vol_percentile_60", 0)), 0),
                    }
        # Align headlines to (ticker_idx, timestep)
        headlines_matrix = {}
        for col_idx, headlines_list in enumerate(all_headlines):
            aligned = headlines_list[-min_len:] if len(headlines_list) >= min_len else headlines_list
            for t_offset, hdl in enumerate(aligned):
                if hdl:
                    headlines_matrix[(col_idx, t_offset)] = hdl
        glm_context = {
            "lstm_probs":  lstm_prob_matrix,
            "rf_probs":    rf_prob_matrix,
            "xgb_probs":   xgb_prob_matrix,
            "sent_scores": sent_score_matrix,
            "sent_mag":    sent_mag_matrix,
            "sent_vol":    sent_vol_matrix,
            "technicals":  tech_indicators,
            "headlines":   headlines_matrix,
        }
        print(f"[GLM] Built enrichment context: {len(tech_indicators)} technical snapshots, "
              f"{len(headlines_matrix)} headline entries")

    # ------------------------------------------------------------
    # GLM Hook 1 -- batch signal review (pre-backtest)
    # ------------------------------------------------------------
    if glm_guard is not None:
        pre_glm_signals = int(np.sum(signal_matrix != 0))
        glm_vetoed = 0
        active_timesteps = [t for t in range(min_len) if np.any(signal_matrix[t] != 0)]
        print(f"\n[GLM] Reviewing {len(active_timesteps)} active timesteps "
              f"across {len(all_ticker_names)} tickers...")

        for t in active_timesteps:
            regime = portfolio_regime[t]
            enrichment = None
            if glm_context is not None:
                enrichment = {
                    "regime_confidence": regime_confidence[t],
                    "signal_balance": {
                        "long":  int(np.sum(signal_matrix[t] == 1)),
                        "short": int(np.sum(signal_matrix[t] == -1)),
                    },
                }
                per_ticker = {}
                for i, ticker in enumerate(all_ticker_names):
                    if signal_matrix[t, i] == 0:
                        continue
                    td = {}
                    td["model_agreement"] = {
                        "votes": int(
                            (glm_context["lstm_probs"][t, i] > 0.5).astype(int) +
                            (glm_context["rf_probs"][t, i]   > 0.5).astype(int) +
                            (glm_context["xgb_probs"][t, i]  > 0.5).astype(int)
                        ),
                        "lstm": round(float(glm_context["lstm_probs"][t, i]), 3),
                        "xgb":  round(float(glm_context["xgb_probs"][t, i]), 3),
                        "rf":   round(float(glm_context["rf_probs"][t, i]), 3),
                    }
                    td["sentiment"] = {
                        "score":           round(float(glm_context["sent_scores"][t, i]), 3),
                        "magnitude":       round(float(glm_context["sent_mag"][t, i]), 3),
                        "headline_count":  int(np.expm1(max(0, glm_context["sent_vol"][t, i]))),
                    }
                    td["technicals"] = glm_context["technicals"].get((i, t), {})
                    td["headlines"]  = glm_context["headlines"].get((i, t), [])
                    per_ticker[ticker] = td
                enrichment["per_ticker"] = per_ticker

            approved_mask = glm_guard.review_signals(
                signal_row=signal_matrix[t],
                prob_row=prob_matrix[t],
                tickers=all_ticker_names,
                regime=regime,
                enrichment=enrichment,
            )
            for i in range(len(all_ticker_names)):
                if signal_matrix[t, i] != 0 and not approved_mask[i]:
                    signal_matrix[t, i] = 0
                    glm_vetoed += 1

        post_glm_signals = int(np.sum(signal_matrix != 0))
        print(f"[GLM] Hook 1 vetoed {glm_vetoed} signals "
              f"({pre_glm_signals} -> {post_glm_signals})")

    Path("results").mkdir(parents=True, exist_ok=True)
    engine = PortfolioEngine(
        initial_capital=10_000.0,
        transaction_fee=0.0002,
        horizon=HORIZON,
        results_dir="results",
        regime_series=portfolio_regime,
        regime_config=REGIME_CONFIG,
        glm_guard=glm_guard,
        ticker_names=all_ticker_names if glm_guard else None,
        glm_context=glm_context,
        regime_confidence=regime_confidence if glm_guard else None,
    )
    metrics = engine.run_portfolio_backtest(price_matrix, signal_matrix, prob_matrix)
    engine.plot_equity_curve(save_path=equity_curve_name)

    print("\n" + "=" * 60)
    print("COMBINED PORTFOLIO RESULTS (ALL SURVIVING GROUPS)")
    print("=" * 60)
    print(f"Starting Capital:      \u00a3{metrics['initial_capital']:,.2f}")
    print(f"Final Capital:         \u00a3{metrics['final_equity']:,.2f}")
    print(f"Aggregate Return:      {metrics['total_return_pct']:+.2f}%")
    print(f"Max Drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:          {metrics['total_trades']}")
    print(f"Win Rate:              {metrics['win_rate_pct']:.1f}%")
    print(f"Avg Win / Avg Loss:    {metrics['avg_win']:.2f} / {metrics['avg_loss']:.2f}")
    print(f"Reward:Risk Ratio:     {metrics['reward_risk']:.2f}")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")

    # ------------------------------------------------------------
    # Per-group breakdown of the combined portfolio result.
    # Re-runs the engine on each group's column-slice so we can see
    # whether the headline number is dominated by 1-2 groups.
    # ------------------------------------------------------------
    if all_group_ids:
        gid_arr = np.asarray(all_group_ids)
        unique_gids = sorted(set(all_group_ids))
        print("\n" + "=" * 60)
        print("PER-GROUP PORTFOLIO BREAKDOWN")
        print("=" * 60)
        print(f"  {'Group':<8} {'Tickers':<28} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>6}")
        print(f"  {'-'*76}")
        per_group_rows = []
        for gid in unique_gids:
            cols = np.where(gid_arr == gid)[0]
            if len(cols) == 0:
                continue
            tk_names = [all_ticker_names[i] for i in cols]
            tk_label = ",".join(tk_names)
            if len(tk_label) > 26:
                tk_label = tk_label[:23] + "..."
            try:
                eng_g = PortfolioEngine(
                    initial_capital=10_000.0,
                    transaction_fee=0.0002,
                    horizon=HORIZON,
                    results_dir="results",
                    regime_series=portfolio_regime[:min_len],
                    regime_config=REGIME_CONFIG,
                    glm_guard=None,
                    ticker_names=None,
                    glm_context=None,
                    regime_confidence=None,
                )
                m_g = eng_g.run_portfolio_backtest(
                    price_matrix[:, cols],
                    signal_matrix[:, cols],
                    prob_matrix[:, cols],
                )
                per_group_rows.append((gid, m_g))
                print(f"  group_{gid:<2}  {tk_label:<28} "
                      f"{m_g['total_return_pct']:>+9.2f} "
                      f"{m_g['sharpe_ratio']:>8.2f} "
                      f"{m_g['max_drawdown_pct']:>8.2f} "
                      f"{m_g['total_trades']:>7d} "
                      f"{m_g['win_rate_pct']:>6.1f}")
            except Exception as e:
                print(f"  group_{gid:<2}  {tk_label:<28}  (failed: {e})")
        print(f"  {'-'*76}")
        if per_group_rows:
            returns = [m['total_return_pct'] for _, m in per_group_rows]
            r_max = max(returns); r_min = min(returns)
            top_gid = per_group_rows[returns.index(r_max)][0]
            top_share = (r_max / sum(r for r in returns if r > 0) * 100.0
                         if sum(r for r in returns if r > 0) > 0 else 0.0)
            print(f"  Top group: group_{top_gid}  ({r_max:+.2f}%)   "
                  f"Worst group: {r_min:+.2f}%   "
                  f"Top group's share of positive return: {top_share:.1f}%")

    # ------------------------------------------------------------
    # Buy-and-hold equal-weight baseline on the same tickers / window.
    # Same Sharpe annualization (sqrt(1638)) the engine uses, so the
    # numbers are directly comparable to the strategy result above.
    # ------------------------------------------------------------
    try:
        bh_initial = float(metrics['initial_capital'])
        # Equal-weight: each ticker normalised to its first price, then averaged.
        norm_prices = price_matrix / price_matrix[0, :][np.newaxis, :]
        port_value  = norm_prices.mean(axis=1) * bh_initial
        bh_returns  = np.diff(port_value) / port_value[:-1]
        bh_total    = (port_value[-1] / port_value[0] - 1.0) * 100.0
        running_max = np.maximum.accumulate(port_value)
        bh_maxdd    = float(np.max((running_max - port_value) / running_max) * 100.0)
        bh_vol      = float(np.std(bh_returns))
        bh_sharpe   = (float(np.mean(bh_returns)) / bh_vol * np.sqrt(1638)) if bh_vol > 0 else 0.0
        bh_final    = float(port_value[-1])

        print("\n" + "=" * 60)
        print("BUY-AND-HOLD BASELINE (equal-weight, same tickers & window)")
        print("=" * 60)
        print(f"Starting Capital:      GBP{bh_initial:,.2f}")
        print(f"Final Capital:         GBP{bh_final:,.2f}")
        print(f"Aggregate Return:      {bh_total:+.2f}%")
        print(f"Max Drawdown:          {bh_maxdd:.2f}%")
        print(f"Sharpe Ratio:          {bh_sharpe:.2f}")
        print(f"  (no trades, no fees -- passive baseline)")
        # Strategy-vs-baseline delta
        delta_ret    = metrics['total_return_pct'] - bh_total
        delta_sharpe = metrics['sharpe_ratio']     - bh_sharpe
        delta_dd     = metrics['max_drawdown_pct'] - bh_maxdd
        print(f"\n  Strategy vs B&H:  Return {delta_ret:+.2f}pp   "
              f"Sharpe {delta_sharpe:+.2f}   "
              f"MaxDD {delta_dd:+.2f}pp (strategy minus B&H)")
        if delta_ret <= 0:
            print(f"  [!] Strategy underperforms passive buy-and-hold on return.")
        if delta_sharpe <= 0:
            print(f"  [!] Strategy Sharpe is no better than passive buy-and-hold.")
    except Exception as e:
        print(f"\n[B&H baseline failed: {e}]")

    if "glm_vetoes" in metrics:
        print(f"\n--- GLM Guard Impact ---")
        print(f"  Hook 1 -- Signals vetoed pre-backtest:  (see [GLM] output above)")
        print(f"  Hook 2 -- Trades vetoed per-order:      {metrics['glm_vetoes']}")
        print(f"  Hook 2 -- Trades resized:               {metrics['glm_size_adjustments']}")
        print(f"  Hook 3 -- Early exits triggered:        {metrics['glm_early_exits']}")

    # ------------------------------------------------------------
    # OOS Holdout Backtest (if oos_holdout_pct > 0)
    # ------------------------------------------------------------
    if oos_holdout_pct > 0.0:
        oos_frac = oos_holdout_pct / 100.0
        oos_len = int(min_len * oos_frac)
        test_len = min_len - oos_len

        if oos_len < 50:
            print(f"\n[OOS] OOS holdout too small ({oos_len} rows) -- skipping OOS backtest")
        else:
            print(f"\n{'='*60}")
            print(f"OUT-OF-SAMPLE BACKTEST (last {oos_holdout_pct:.0f}% = {oos_len} timesteps)")
            print(f"{'='*60}")

            # Split matrices
            price_test = price_matrix[:test_len, :]
            price_oos = price_matrix[test_len:, :]
            prob_test = prob_matrix[:test_len, :]
            prob_oos = prob_matrix[test_len:, :]
            signal_test = signal_matrix[:test_len, :]
            signal_oos = signal_matrix[test_len:, :]

            # Regime series split
            regime_test = portfolio_regime[:test_len]
            regime_oos = portfolio_regime[test_len:]

            # OOS backtest
            oos_equity_name = equity_curve_name.replace(".png", "_oos.png")
            engine_oos = PortfolioEngine(
                initial_capital=10_000.0,
                transaction_fee=0.0002,
                horizon=HORIZON,
                results_dir="results",
                regime_series=regime_oos,
                regime_config=REGIME_CONFIG,
                glm_guard=None,  # No GLM review on OOS
                ticker_names=None,
                glm_context=None,
                regime_confidence=None,
            )
            oos_metrics = engine_oos.run_portfolio_backtest(price_oos, signal_oos, prob_oos)
            engine_oos.plot_equity_curve(save_path=oos_equity_name)

            print(f"\n OOS Results ({oos_len} timesteps):")
            print(f"  Return:   {oos_metrics['total_return_pct']:+.2f}%")
            print(f"  Sharpe:   {oos_metrics['sharpe_ratio']:.3f}")
            print(f"  Max DD:   {oos_metrics['max_drawdown_pct']:.2f}%")
            print(f"  Win Rate: {oos_metrics['win_rate_pct']:.1f}%")
            print(f"  Trades:   {oos_metrics['total_trades']}")

            # Compute degradation
            degradation = {
                "sharpe_degradation": metrics['sharpe_ratio'] - oos_metrics['sharpe_ratio'],
                "return_degradation": metrics['total_return_pct'] - oos_metrics['total_return_pct'],
                "winrate_degradation": metrics['win_rate_pct'] - oos_metrics['win_rate_pct'],
                "maxdd_degradation": metrics['max_drawdown_pct'] - oos_metrics['max_drawdown_pct'],
            }

            print(f"\n{'='*60}")
            print("OUT-OF-SAMPLE DEGRADATION REPORT")
            print(f"{'='*60}")
            print(f"  {'Metric':<20}  {'Test':>10}  {'OOS':>10}  {'Degradation':>14}")
            print(f"  {'-'*58}")
            print(f"  {'Sharpe':<20}  {metrics['sharpe_ratio']:>10.3f}  {oos_metrics['sharpe_ratio']:>10.3f}  {degradation['sharpe_degradation']:>+14.3f}")
            print(f"  {'Return (%)':<20}  {metrics['total_return_pct']:>10.2f}  {oos_metrics['total_return_pct']:>10.2f}  {degradation['return_degradation']:>+14.2f}")
            print(f"  {'Win Rate (%)':<20}  {metrics['win_rate_pct']:>10.1f}  {oos_metrics['win_rate_pct']:>10.1f}  {degradation['winrate_degradation']:>+14.1f}")
            print(f"  {'Max Drawdown (%)':<20}  {metrics['max_drawdown_pct']:>10.2f}  {oos_metrics['max_drawdown_pct']:>10.2f}  {degradation['maxdd_degradation']:>+14.2f}")
            print(f"  {'Total Trades':<20}  {metrics['total_trades']:>10d}  {oos_metrics['total_trades']:>10d}  {metrics['total_trades'] - oos_metrics['total_trades']:>+14d}")
            print(f"  {'-'*58}")
            if degradation['sharpe_degradation'] > 0.5:
                print(f"  [!] WARNING: Sharpe degradation > 0.5 -- possible overfitting")
            elif degradation['sharpe_degradation'] < -0.3:
                print(f"  [i] OOS Sharpe exceeds test -- model may generalize well")

            # -- Monte Carlo Robustness Test (on FULL test period) --
            mc_results = _monte_carlo_robustness(
                price_matrix=price_matrix,
                signal_matrix=signal_matrix,
                prob_matrix=prob_matrix,
                regime_series=portfolio_regime,
                n_sims=1000,
            )

            # Wrap return with test/OOS/degradation/MC
            return {
                "test_metrics": metrics,
                "oos_metrics": oos_metrics,
                "degradation": degradation,
                "monte_carlo": mc_results,
                "initial_capital": metrics['initial_capital'],
                "final_equity": metrics['final_equity'],
                "total_return_pct": metrics['total_return_pct'],
                "max_drawdown_pct": metrics['max_drawdown_pct'],
                "total_trades": metrics['total_trades'],
                "win_rate_pct": metrics['win_rate_pct'],
                "avg_win": metrics['avg_win'],
                "avg_loss": metrics['avg_loss'],
                "reward_risk": metrics['reward_risk'],
                "sharpe_ratio": metrics['sharpe_ratio'],
                "glm_vetoes": metrics.get("glm_vetoes", 0),
                "glm_size_adjustments": metrics.get("glm_size_adjustments", 0),
                "glm_early_exits": metrics.get("glm_early_exits", 0),
            }

    # -- Monte Carlo Robustness Test (no OOS case) --
    mc_results = _monte_carlo_robustness(
        price_matrix=price_matrix,
        signal_matrix=signal_matrix,
        prob_matrix=prob_matrix,
        regime_series=portfolio_regime,
        n_sims=1000,
    )

    metrics["monte_carlo"] = mc_results
    return metrics


# --------------------------------------------------------------------------------
# Monte Carlo Robustness Tests (always on)
# --------------------------------------------------------------------------------

def _monte_carlo_robustness(
    price_matrix: np.ndarray,
    signal_matrix: np.ndarray,
    prob_matrix: np.ndarray,
    regime_series: np.ndarray,
    n_sims: int = 1000,
    block_size: int = 20,
) -> Dict[str, Any]:
    """
    Monte Carlo robustness test using Circular Block Bootstrap (CBB).

    Resamples blocks of timestep returns to build a distribution of outcomes.
    Quantifies:
    1. Probability of Backtest Overfitting (PBO)
    2. Robustness to return sequence perturbations

    Based on:
    - Bailey et al. (2016) "The Probability of Backtest Overfitting"
    - Lopez de Prado (2018) AFML Ch.16
    """
    from PredictionEngine import PortfolioEngine

    print(f"\n{'='*60}")
    print(f"MONTE CARLO ROBUSTNESS TEST ({n_sims} simulations)")
    print(f"{'='*60}")

    # -- Step 1: Run baseline backtest to get equity curve --
    engine_base = PortfolioEngine(
        initial_capital=10_000.0,
        transaction_fee=0.0002,
        horizon=HORIZON,
        results_dir="results",
        regime_series=regime_series,
        regime_config=REGIME_CONFIG,
        glm_guard=None,
        ticker_names=None,
        glm_context=None,
        regime_confidence=None,
    )
    base_metrics = engine_base.run_portfolio_backtest(price_matrix, signal_matrix, prob_matrix)
    base_sharpe = base_metrics['sharpe_ratio']

    # Extract timestep returns from equity curve
    equity = np.array(engine_base.equity_curve, dtype=float)
    if len(equity) < 2:
        print(f"  [MC] Insufficient equity curve data ({len(equity)} points)")
        return {
            "pbo": None,
            "mc_sharpe_mean": None,
            "mc_sharpe_std": None,
            "mc_sharpe_5th": None,
            "mc_sharpe_95th": None,
            "robust": None,
            "n_sims": n_sims,
        }

    # Compute per-timestep returns (as percentages)
    timestep_returns = np.diff(equity) / equity[:-1] * 100.0
    n_returns = len(timestep_returns)
    print(f"  Extracted {n_returns} timestep returns for bootstrap")
    print(f"  Base Sharpe: {base_sharpe:.3f} | Base Return: {base_metrics['total_return_pct']:+.2f}%")

    # -- Step 2: Circular Block Bootstrap --
    mc_sharpes = []
    mc_returns = []
    mc_maxdds = []

    for sim in range(n_sims):
        # Resample with circular blocks (preserves autocorrelation)
        bootstrapped = _circular_block_bootstrap(timestep_returns, block_size=block_size)

        # Compute metrics for this simulation
        sim_sharpe = _sharpe_from_returns(bootstrapped)
        sim_return = np.prod(1 + bootstrapped / 100.0) - 1.0
        sim_maxdd = _max_drawdown_from_returns(bootstrapped)

        mc_sharpes.append(sim_sharpe)
        mc_returns.append(sim_return * 100)
        mc_maxdds.append(sim_maxdd)

    mc_sharpes = np.array(mc_sharpes)
    mc_returns = np.array(mc_returns)
    mc_maxdds = np.array(mc_maxdds)

    # -- Step 3: Compute PBO --
    # PBO = fraction of bootstrap samples where Sharpe >= original
    # Interpretation: HIGH PBO means most resampled scenarios beat original
    #   -> Original is conservative (good), not an optimistic outlier
    # LOW PBO means few resampled scenarios beat original
    #   -> Original may be an optimistic outlier (bad, possible overfitting)
    pbo = float(np.mean(mc_sharpes >= base_sharpe))

    # -- Step 4: Robustness check --
    # Strategy is robust if 5th percentile of MC Sharpe > 0
    robust_95 = float(np.percentile(mc_sharpes, 5)) > 0

    # -- Step 5: Interpret PBO correctly --
    mc_mean = float(np.mean(mc_sharpes))
    if mc_mean > base_sharpe:
        # Bootstrap distribution is ABOVE original -> original is conservative
        pbo_interpretation = "conservative"
    else:
        # Bootstrap distribution is BELOW original -> original may be optimistic
        pbo_interpretation = "optimistic"

    # -- Step 6: Report --
    print(f"\n  Monte Carlo Results ({n_sims} bootstrap simulations):")
    print(f"  {'Metric':<25}  {'Value':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Original Sharpe':<25}  {base_sharpe:>12.3f}")
    print(f"  {'MC Sharpe Mean':<25}  {mc_mean:>12.3f}")
    print(f"  {'MC Sharpe Std':<25}  {np.std(mc_sharpes):>12.3f}")
    print(f"  {'MC Sharpe 5th pct':<25}  {np.percentile(mc_sharpes, 5):>12.3f}")
    print(f"  {'MC Sharpe 95th pct':<25}  {np.percentile(mc_sharpes, 95):>12.3f}")
    print(f"  {'MC Return Mean (%)':<25}  {np.mean(mc_returns):>12.2f}")
    print(f"  {'MC MaxDD Mean (%)':<25}  {np.mean(mc_maxdds):>12.2f}")
    print(f"  {'-'*40}")
    print(f"  {'MC Sharpe vs Original':<25}  {pbo_interpretation:>12}")
    print(f"  {'PBO (MC >= Original)':<25}  {pbo:>12.4f}")
    print(f"  {'Robust (5% CI > 0)':<25}  {'Yes' if robust_95 else 'No':>12}")
    print(f"  {'-'*40}")

    if pbo_interpretation == "conservative" and robust_95:
        print(f"  [OK] MC Sharpe mean ({mc_mean:.3f}) > Original ({base_sharpe:.3f}) -- result is conservative")
        print(f"  [OK] 5th percentile of MC Sharpe > 0 -- strategy is robust")
    elif pbo_interpretation == "conservative" and not robust_95:
        print(f"  [!] MC Sharpe mean ({mc_mean:.3f}) > Original ({base_sharpe:.3f}) -- result is conservative")
        print(f"  [X] 5th percentile of MC Sharpe <= 0 -- some resampled scenarios lose money")
    elif pbo_interpretation == "optimistic" and robust_95:
        print(f"  [!] MC Sharpe mean ({mc_mean:.3f}) < Original ({base_sharpe:.3f}) -- result may be optimistic")
        print(f"  [OK] 5th percentile of MC Sharpe > 0 -- strategy is still robust")
    else:
        print(f"  [X] MC Sharpe mean ({mc_mean:.3f}) < Original ({base_sharpe:.3f}) -- possible overfitting")
        print(f"  [X] 5th percentile of MC Sharpe <= 0 -- strategy may not be robust")

    return {
        "pbo": pbo,
        "mc_sharpe_mean": float(np.mean(mc_sharpes)),
        "mc_sharpe_std": float(np.std(mc_sharpes)),
        "mc_sharpe_5th": float(np.percentile(mc_sharpes, 5)),
        "mc_sharpe_95th": float(np.percentile(mc_sharpes, 95)),
        "mc_return_mean": float(np.mean(mc_returns)),
        "mc_maxdd_mean": float(np.mean(mc_maxdds)),
        "robust": robust_95,
        "n_sims": n_sims,
        "original_sharpe": base_sharpe,
    }


def _circular_block_bootstrap(returns: np.ndarray, block_size: int = 20) -> np.ndarray:
    """
    Circular Block Bootstrap (CBB) for time series with autocorrelation.

    Divides returns into blocks of size `block_size`, then resamples blocks
    with replacement. The 'circular' aspect means the last block wraps around
    to the beginning, ensuring all starting positions are equally likely.

    This preserves the autocorrelation structure of returns, which is critical
    for financial time series where consecutive returns are often correlated.
    """
    n = len(returns)
    # Extend array circularly
    extended = np.concatenate([returns, returns[:block_size]])

    bootstrapped = []
    pos = 0
    while pos < n:
        # Random start position (circular)
        start = np.random.randint(0, n)
        block = extended[start:start + block_size]
        bootstrapped.extend(block)
        pos += block_size

    return np.array(bootstrapped[:n])


def _sharpe_from_returns(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from a series of returns (in %)."""
    returns = returns / 100.0
    excess = returns - risk_free_rate
    if np.std(excess) == 0:
        return 0.0
    # Annualize (assuming hourly data, ~4380 trading hours/year)
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(4380)
    return sharpe


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Compute maximum drawdown from a series of returns (in %)."""
    returns = returns / 100.0
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown) * 100)


# --------------------------------------------------------------------------------
# Compare mode: LSTM vs MTL on the same groups
# --------------------------------------------------------------------------------

def run_compare(
    groups: List[List[str]],
    selection_threshold: float,
    data_path: str,
    sentiment_alpha: float,
    train_anchor: Optional[str],
    test_anchor: Optional[str],
    use_full_split: bool,
) -> None:
    """Run LSTM then MTL on identical groups, print side-by-side comparison."""

    Path("results").mkdir(parents=True, exist_ok=True)

    LSTM_CSV = "results/group_tournament_lstm.csv"

    # -- LSTM arm (with resume) -------------------------------------------
    if Path(LSTM_CSV).exists():
        print("\n" + "#" * 60)
        print(f"#  COMPARE -- ARM 1: LSTM (RESUMING from {LSTM_CSV})")
        print("#" * 60)
        lstm_survivors = _load_survivors_from_csv(LSTM_CSV, selection_threshold)
        print(f"  Loaded {len(lstm_survivors)} LSTM survivors from existing CSV.")
        print("  Skipping tournament re-training; jumping straight to Stage 3.")
    else:
        print("\n" + "#" * 60)
        print("#  COMPARE -- ARM 1: STANDARD LSTM")
        print("#" * 60)
        lstm_survivors = run_tournament(
            groups=groups,
            selection_threshold=selection_threshold,
            data_path=data_path,
            sentiment_alpha=sentiment_alpha,
            anchor_end_date=train_anchor,
            use_full_split=use_full_split,
            csv_name=LSTM_CSV,
        )

    lstm_metrics: Dict[str, Any] = {}
    if lstm_survivors:
        lstm_metrics = run_combined_backtest(
            survivors=lstm_survivors,
            data_path=data_path,
            anchor_end_date=test_anchor,
            equity_curve_name="equity_curve_lstm.png",
            sentiment_alpha=sentiment_alpha,
            train_anchor_date=train_anchor,
            use_full_split=use_full_split,
            mtl_model=None,
        )
    else:
        print("  LSTM arm: no survivors -- skipping Stage 3.")

    # -- MTL arm -----------------------------------------------------------
    print("\n" + "#" * 60)
    print("#  COMPARE -- ARM 2: MTL LSTM")
    print("#" * 60)
    mtl_survivors, mtl_model = run_tournament_mtl(
        groups=groups,
        selection_threshold=selection_threshold,
        data_path=data_path,
        sentiment_alpha=sentiment_alpha,
        anchor_end_date=train_anchor,
        use_full_split=use_full_split,
        csv_name="results/group_tournament_mtl.csv",
    )

    mtl_metrics: Dict[str, Any] = {}
    if mtl_survivors:
        if mtl_model is None:
            try:
                from mtl_lstm_model import MTLLSTMModel
                from data_loader import DataLoader
                from feature_engineer import FeatureEngineer
                from ensemble_model import _make_preprocessor
                first_ticker = groups[0][0] if groups else None
                if first_ticker:
                    loader   = DataLoader(data_path, first_ticker)
                    enriched = FeatureEngineer().compute_indicators(loader.load_data())
                    _, splits, _ = _make_preprocessor(LOOKBACK, HORIZON, enriched, train_anchor, use_full_split)
                    n_feats   = splits["X_train"].shape[2]
                    group_ids = [r["group_id"] for r in mtl_survivors]
                    mtl_model = MTLLSTMModel.load(
                        save_dir="models/mtl",
                        input_shape=(LOOKBACK, n_feats),
                        group_ids=group_ids,
                    )
            except Exception as e:
                print(f"[MTL] Could not reload model for Stage 3: {e}")

        mtl_metrics = run_combined_backtest(
            survivors=mtl_survivors,
            data_path=data_path,
            anchor_end_date=test_anchor,
            equity_curve_name="equity_curve_mtl.png",
            sentiment_alpha=sentiment_alpha,
            train_anchor_date=train_anchor,
            use_full_split=use_full_split,
            mtl_model=mtl_model,
        )
    else:
        print("  MTL arm: no survivors -- skipping Stage 3.")

    # -- Per-group val Sharpe comparison CSV -------------------------------
    lstm_csv_rows: Dict[int, str] = {}
    mtl_csv_rows:  Dict[int, str] = {}
    for path, store in [
        (LSTM_CSV,                             lstm_csv_rows),
        ("results/group_tournament_mtl.csv",   mtl_csv_rows),
    ]:
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    store[int(row["group_id"])] = row["val_sharpe"]
        except FileNotFoundError:
            pass

    all_group_ids = sorted(set(lstm_csv_rows) | set(mtl_csv_rows))
    comparison_rows = []
    for gid in all_group_ids:
        ls = lstm_csv_rows.get(gid, "N/A")
        ms = mtl_csv_rows.get(gid,  "N/A")
        try:
            winner = "LSTM" if float(ls) >= float(ms) else "MTL"
        except (ValueError, TypeError):
            winner = "N/A"
        comparison_rows.append({
            "group_id":        gid,
            "lstm_val_sharpe": ls,
            "mtl_val_sharpe":  ms,
            "winner":          winner,
        })

    comp_csv = Path("results/comparison.csv")
    with open(comp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "lstm_val_sharpe", "mtl_val_sharpe", "winner"])
        writer.writeheader()
        writer.writerows(comparison_rows)

    # -- Side-by-side portfolio metrics ------------------------------------
    METRIC_KEYS = [
        ("total_return_pct",  "Return (%)",       "+.2f"),
        ("sharpe_ratio",      "Sharpe",            ".3f"),
        ("max_drawdown_pct",  "Max Drawdown (%)",  ".2f"),
        ("win_rate_pct",      "Win Rate (%)",      ".1f"),
        ("total_trades",      "Total Trades",      "d"),
        ("reward_risk",       "Reward:Risk",       ".2f"),
        ("final_equity",      "Final Equity (GBP)",  ",.2f"),
    ]

    print("\n" + "=" * 60)
    print("LSTM vs MTL -- SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<22}  {'LSTM':>12}  {'MTL':>12}  {'Winner':>8}")
    print("  " + "-" * 56)
    for key, label, fmt in METRIC_KEYS:
        l_val = lstm_metrics.get(key)
        m_val = mtl_metrics.get(key)
        if l_val is None and m_val is None:
            l_str, m_str, w_str = "N/A", "N/A", "N/A"
        else:
            l_str = (f"{l_val:{fmt}}" if l_val is not None else "N/A")
            m_str = (f"{m_val:{fmt}}" if m_val is not None else "N/A")
            if l_val is not None and m_val is not None:
                if key == "max_drawdown_pct":
                    w_str = "LSTM" if l_val <= m_val else "MTL "
                else:
                    w_str = "LSTM" if l_val >= m_val else "MTL "
            else:
                w_str = "N/A"
        print(f"  {label:<22}  {l_str:>12}  {m_str:>12}  {w_str:>8}")

    print("  " + "-" * 56)

    print(f"\n  {'Grp':>4}  {'LSTM Sharpe':>12}  {'MTL Sharpe':>12}  {'Winner':>8}")
    print("  " + "-" * 44)
    for row in comparison_rows:
        print(
            f"  {row['group_id']:>4}  {row['lstm_val_sharpe']:>12}  "
            f"{row['mtl_val_sharpe']:>12}  {row['winner']:>8}"
        )
    lstm_wins = sum(1 for r in comparison_rows if r["winner"] == "LSTM")
    mtl_wins  = sum(1 for r in comparison_rows if r["winner"] == "MTL")
    print("  " + "-" * 44)
    print(f"  Group wins -- LSTM: {lstm_wins}  MTL: {mtl_wins}")
    print(f"\nComparison saved to {comp_csv}")
    print("Equity curves: results/equity_curve_lstm.png  |  results/equity_curve_mtl.png")


def train_or_load_saml(
    groups: List[List[str]],
    data_path: str,
    anchor_end_date: Optional[str],
    use_full_split: bool,
    split_months: tuple,
) -> SAMLMetaLearner:
    """Train or load SAML meta-learner."""
    from pathlib import Path
    import numpy as np
    saml_dir = Path("models/saml")
    meta_weights_path = saml_dir / "meta_init.weights.h5"

    if meta_weights_path.exists():
        print(f"\n[SAML] Loading existing meta-learner from {saml_dir}")
        from data_loader import DataLoader
        from feature_engineer import FeatureEngineer
        from ensemble_model import _make_preprocessor

        first_ticker = groups[0][0] if groups else None
        if first_ticker:
            loader = DataLoader(data_path, first_ticker)
            enriched = FeatureEngineer().compute_indicators(loader.load_data())
            _, splits, _ = _make_preprocessor(
                24, 8, enriched, anchor_end_date, use_full_split=use_full_split
            )
            n_features = splits["X_train"].shape[2]
            saml_learner = SAMLMetaLearner.load(str(saml_dir), input_shape=(24, n_features))
            return saml_learner
        raise ValueError("No groups available")

    print("\n" + "=" * 60)
    print("STEP 1: TRAINING SAML META-LEARNER")
    print("=" * 60)
    saml_dir.mkdir(parents=True, exist_ok=True)

    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from data_preprocessor import DataPreprocessor
    from regime_detector import RegimeDetector
    from lstm_model import make_direction_onehot_from_raw

    first_ticker = groups[0][0] if groups else None
    if not first_ticker:
        raise ValueError("No groups available for SAML training")

    loader = DataLoader(data_path, first_ticker)
    enriched = FeatureEngineer().compute_indicators(loader.load_data())

    preprocessor = DataPreprocessor(
        lookback_window=24,
        forecast_horizon=8,
        use_walk_forward=not use_full_split,
        use_full_split=use_full_split,
        split_months=split_months,
    )
    splits = preprocessor.preprocess(enriched)
    n_features = splits["X_train"].shape[2]

    regime_detector = RegimeDetector()
    regime_detector.fit(enriched)

    all_regime_data = {"BULL": [], "BEAR": [], "CRISIS": []}

    for group in groups:
        for ticker in group:
            try:
                ticker_loader = DataLoader(data_path, ticker)
                ticker_enriched = FeatureEngineer().compute_indicators(ticker_loader.load_data())
                ticker_preprocessor = DataPreprocessor(lookback_window=24, forecast_horizon=8)
                ticker_splits = ticker_preprocessor.preprocess(ticker_enriched)
                if len(ticker_splits["X_train"]) < 100:
                    continue

                vol_pct = ticker_enriched['vol_percentile_60'].values
                sharpe = ticker_enriched['sharpe_ratio_20'].values

                regime_labels = np.zeros(len(ticker_enriched), dtype=int)
                regime_labels[(sharpe < 0) & (vol_pct < 70)] = 1  # BEAR
                regime_labels[vol_pct >= 70] = 2  # CRISIS

                labels_dict, _, _ = make_direction_onehot_from_raw(
                    ticker_enriched,
                    {
                        "idx_train": ticker_splits["idx_train"],
                        "idx_val": ticker_splits["idx_val"],
                        "idx_test": ticker_splits["idx_test"],
                    },
                    lookback_window=24,
                    forecast_horizon=8,
                )

                lookback = 24
                for regime_idx, regime_name in enumerate(["BULL", "BEAR", "CRISIS"]):
                    regime_seq_indices = []
                    for i, seq_start_idx in enumerate(ticker_splits["idx_train"]):
                        data_idx = seq_start_idx + lookback
                        if data_idx < len(regime_labels) and regime_labels[data_idx] == regime_idx:
                            regime_seq_indices.append(i)

                    if len(regime_seq_indices) > 0:
                        regime_seq_indices = np.array(regime_seq_indices)
                        X_regime = ticker_splits["X_train"][regime_seq_indices]
                        y_regime = labels_dict["train"][regime_seq_indices]

                        max_samples = 100
                        if len(X_regime) > max_samples:
                            indices = np.random.choice(len(X_regime), max_samples, replace=False)
                            X_regime = X_regime[indices]
                            y_regime = y_regime[indices]

                        all_regime_data[regime_name].append((X_regime, y_regime))

            except Exception as e:
                print(f"[SAML] Warning: Skipping {ticker}: {e}")
                continue

    meta_train_data = {}
    for regime_name in ["BULL", "BEAR", "CRISIS"]:
        if all_regime_data[regime_name]:
            X_list = [d[0] for d in all_regime_data[regime_name]]
            y_list = [d[1] for d in all_regime_data[regime_name]]
            # Concatenate all data for this regime
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        # Split into train/val (80/20)
        split_idx = int(0.8 * len(X_all))
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_val = X_all[split_idx:]
        y_val = y_all[split_idx:]

        meta_train_data[regime_name] = (X_train, y_train, X_val, y_val)
        print(f"[SAML] {regime_name}: {len(X_train)} train, {len(X_val)} val sequences")

    if not meta_train_data:
        raise ValueError("No regime data collected for SAML training")

    saml_learner = SAMLMetaLearner(input_shape=(24, n_features))
    saml_learner.meta_train_step(meta_train_data, epochs=50)
    saml_learner.save(str(saml_dir))

    print(f"\n[SAML] Training complete. Saved to {saml_dir}")
    return saml_learner


def train_or_load_quantile_barriers(
    groups: List[List[str]],
    data_path: str,
    anchor_end_date: Optional[str],
    use_full_split: bool,
    split_months: tuple,
) -> QuantileBarrierLearner:
    """
    Train or load Quantile Barrier learner.

    If models/quantile_barriers.json exists, loads it.
    Otherwise, trains on historical data by:
    1. Loading data for each group
    2. Computing historical returns
    3. Calculating optimal barrier quantiles per regime (BULL/BEAR/CRISIS)
    4. Learning tau_bull, tau_bear, tau_crisis via quantile regression
    5. Saving to models/quantile_barriers.json
    """
    from pathlib import Path
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from regime_detector import RegimeDetector
    import tensorflow as tf

    barrier_path = Path("models/quantile_barriers.json")

    # Step 1: Check if already trained
    if barrier_path.exists():
        print(f"\n[QuantileBarrier] Loading existing from {barrier_path}")
        learner = QuantileBarrierLearner.load(str(barrier_path), trainable=False)
        print(f"[QuantileBarrier] Loaded tau_bull={learner._init_tau_bull:.3f}, "
              f"tau_bear={learner._init_tau_bear:.3f}, tau_crisis={learner._init_tau_crisis:.3f}")
        return learner

    print("\n" + "=" * 60)
    print("STEP 2: TRAINING QUANTILE BARRIERS")
    print("=" * 60)
    barrier_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 2: Collect data from all groups
    all_returns = []
    all_regimes = []
    all_atr = []
    all_prices = []

    print(f"[QuantileBarrier] Loading data from {len(groups)} groups...")

    # First, collect data from representative tickers to fit regime detector
    regime_training_data = []
    for group_id, stock_list in enumerate(groups[:5]):  # Use first 5 groups for regime training
        for ticker in stock_list[:2]:  # Use first 2 tickers per group
            try:
                loader = DataLoader(data_path, ticker)
                raw_df = loader.load_data()
                if anchor_end_date is not None:
                    mask = pd.to_datetime(raw_df.index, errors="coerce") <= pd.Timestamp(anchor_end_date)
                    raw_df = raw_df.loc[mask]
                if len(raw_df) > 500:
                    enriched = FeatureEngineer().compute_indicators(raw_df)
                    regime_training_data.append(enriched)
            except Exception as e:
                print(f"  [Warning] Failed to load {ticker}: {e}")
                continue

    # Fit regime detector
    print("[QuantileBarrier] Training regime detector...")
    try:
        regime_detector = RegimeDetector.fit_and_save(
            regime_training_data,
            save_path="models/regime_detector_quantile.pkl",
            n_components=3
        )
    except Exception as e:
        print(f"  [Warning] Regime detector training failed: {e}")
        print("  [Warning] Using default regime assignments (BULL=0, BEAR=1, CRISIS=2)")
        regime_detector = None

    # Step 3: Collect historical returns with regime labels
    print("[QuantileBarrier] Computing historical returns and regime labels...")

    for group_id, stock_list in enumerate(groups):
        for ticker in stock_list:
            try:
                loader = DataLoader(data_path, ticker)
                raw_df = loader.load_data()

                # Truncate to training window if specified
                if anchor_end_date is not None:
                    mask = pd.to_datetime(raw_df.index, errors="coerce") <= pd.Timestamp(anchor_end_date)
                    raw_df = raw_df.loc[mask]

                if len(raw_df) < 100:
                    continue

                # Compute features
                enriched = FeatureEngineer().compute_indicators(raw_df)

                # Get regime labels
                if regime_detector is not None:
                    regime_labels = regime_detector.predict_regime_series(enriched)
                else:
                    # Fallback: simple volatility-based regime detection
                    vol_pct = enriched['vol_percentile_60'].values
                    sharpe = enriched['sharpe_ratio_20'].values
                    regime_labels = np.full(len(enriched), 'BULL')
                    regime_labels[(sharpe < 0) & (vol_pct < 70)] = 'BEAR'
                    regime_labels[vol_pct >= 70] = 'CRISIS'

                # Compute returns with different forecast horizons
                closes = enriched['Close'].values
                atrs = enriched['ATR_14'].values

                for horizon in [4, 8, 12, 16]:  # Multiple horizons for robustness
                    if len(closes) > horizon + 10:
                        future_returns = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
                        aligned_closes = closes[:-horizon]
                        aligned_atr = atrs[:-horizon]
                        aligned_regimes = regime_labels[:-horizon]

                        # Filter valid values
                        valid_mask = ~(np.isnan(future_returns) | np.isnan(aligned_atr) | np.isnan(aligned_closes))

                        all_returns.extend(future_returns[valid_mask].tolist())
                        all_regimes.extend(aligned_regimes[valid_mask].tolist())
                        all_atr.extend(aligned_atr[valid_mask].tolist())
                        all_prices.extend(aligned_closes[valid_mask].tolist())

            except Exception as e:
                print(f"  [Warning] Failed to process {ticker}: {e}")
                continue

        if group_id % 5 == 0:
            print(f"  Processed {group_id + 1}/{len(groups)} groups...")

    if len(all_returns) < 1000:
        print(f"[QuantileBarrier] WARNING: Insufficient data ({len(all_returns)} samples). Using defaults.")
        learner = QuantileBarrierLearner(
            tau_bull=0.35,
            tau_bear=0.55,
            tau_crisis=0.70,
            trainable=False
        )
        learner.save(barrier_path)
        return learner

    print(f"[QuantileBarrier] Collected {len(all_returns)} return samples")

    # Step 4: Calculate optimal quantiles per regime
    # Convert to numpy arrays
    returns = np.array(all_returns)
    regimes = np.array(all_regimes)
    atrs = np.array(all_atr)
    prices = np.array(all_prices)

    # Calculate absolute returns (for barrier sizing)
    abs_returns = np.abs(returns)

    # Regime mapping: BULL=0, BEAR=1, CRISIS=2
    regime_map = {'BULL': 0, 'BEAR': 1, 'CRISIS': 2}
    regime_indices = np.array([regime_map.get(r, 0) for r in regimes])

    # Calculate quantiles per regime
    bull_mask = regime_indices == 0
    bear_mask = regime_indices == 1
    crisis_mask = regime_indices == 2

    # Use median absolute return as baseline for each regime
    bull_returns = abs_returns[bull_mask]
    bear_returns = abs_returns[bear_mask]
    crisis_returns = abs_returns[crisis_mask]

    # Calculate optimal tau values based on return distributions
    # tau = target_barrier_pct * price / ATR
    # We want barriers that capture ~60% of moves in bull, ~65% in bear, ~75% in crisis
    target_quantiles = [0.60, 0.65, 0.75]

    optimal_taus = []
    for mask, target_q, name in [(bull_mask, target_quantiles[0], 'BULL'),
                                   (bear_mask, target_quantiles[1], 'BEAR'),
                                   (crisis_mask, target_quantiles[2], 'CRISIS')]:
        if np.sum(mask) > 100:
            regime_rets = abs_returns[mask]
            regime_atr = atrs[mask]
            regime_prices = prices[mask]

            # Target barrier percentage based on return quantile
            target_barrier_pct = np.quantile(regime_rets, target_q)

            # Compute optimal tau: tau = barrier_pct * price / ATR
            # Average across samples
            taus = target_barrier_pct * regime_prices / regime_atr
            optimal_tau = np.median(taus)

            # Clip to valid range [0.1, 0.9]
            optimal_tau = np.clip(optimal_tau, 0.1, 0.9)
            optimal_taus.append(optimal_tau)

            print(f"  {name}: n={np.sum(mask)}, target_q={target_q:.2f}, "
                  f"optimal_tau={optimal_tau:.3f}")
        else:
            # Default values if insufficient data
            default_tau = [0.35, 0.55, 0.70][len(optimal_taus)]
            optimal_taus.append(default_tau)
            print(f"  {name}: Insufficient data, using default tau={default_tau:.3f}")

    tau_bull, tau_bear, tau_crisis = optimal_taus

    # Step 5: Create and save the learner
    print(f"[QuantileBarrier] Creating learner with tau_bull={tau_bull:.3f}, "
          f"tau_bear={tau_bear:.3f}, tau_crisis={tau_crisis:.3f}")

    learner = QuantileBarrierLearner(
        tau_bull=tau_bull,
        tau_bear=tau_bear,
        tau_crisis=tau_crisis,
        trainable=False  # Pre-computed values
    )

    # Build the layer to initialize variables
    learner.build()

    # Save
    learner.save(barrier_path)

    print(f"[QuantileBarrier] Training complete. Saved to {barrier_path}")

    return learner


def detect_primary_regime(train_anchor: Optional[str], test_anchor: Optional[str]) -> str:
    """
    Detect PRIMARY regime for the training period using date-based classification.
    
    Uses market history knowledge to determine overall regime context:
    - 2008-2009: CRISIS (Global Financial Crisis)
    - 2018 Q4: BEAR (Oct-Dec bear market)
    - 2020 Q1-Q2: CRISIS (COVID crash)
    - 2022: BEAR (Rate hike bear market)
    - Default: BULL
    
    Args:
        train_anchor: End date of training period
        test_anchor: End date of test period
    
    Returns:
        Primary regime: "BULL", "BEAR", or "CRISIS"
    """
    from datetime import datetime
    
    # Default to BULL
    primary_regime = "BULL"
    
    if train_anchor is None:
        return primary_regime
    
    try:
        train_date = datetime.strptime(train_anchor, "%Y-%m-%d")
        
        # Check if training period INCLUDES these regime periods
        # 2008-2009 Financial Crisis (includes buildup and recovery)
        if datetime(2008, 8, 1) <= train_date <= datetime(2009, 8, 31):
            primary_regime = "CRISIS"
        # 2018 Q4 Bear Market (train ends around Oct-Sep, includes bear period)
        elif datetime(2018, 9, 1) <= train_date <= datetime(2018, 12, 31):
            primary_regime = "BEAR"
        # 2020 COVID Crash
        elif datetime(2020, 2, 1) <= train_date <= datetime(2020, 8, 31):
            primary_regime = "CRISIS"
        # 2022 Bear Market
        elif datetime(2022, 1, 1) <= train_date <= datetime(2022, 12, 31):
            primary_regime = "BEAR"
        
        print(f"[Hybrid Regime] Primary regime for {train_anchor}: {primary_regime}")
        return primary_regime
        
    except (ValueError, TypeError):
        return primary_regime


def run_specialist_training(
    groups: List[List[str]],
    survivors: List[Dict[str, Any]],
    data_path: str,
    train_anchor: Optional[str],
    test_anchor: Optional[str],
    use_full_split: bool,
    split_months: tuple,
    saml_learner: SAMLMetaLearner,
    quantile_learner: QuantileBarrierLearner,
) -> Dict[str, Dict[int, str]]:
    """Train all 3 regime specialists for each surviving group using hybrid regime detection."""
    from regime_specialist import RegimeSpecialistLSTM, SPECIALIST_CONFIGS, Regime, REGIME_TO_IDX
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from data_preprocessor import DataPreprocessor
    from lstm_model import make_direction_onehot_from_raw
    from pathlib import Path

    print("\n" + "=" * 60)
    print("STEP 4: TRAINING REGIME SPECIALISTS (HYBRID)")
    print("=" * 60)

    specialist_paths = {name: {} for name in SPECIALIST_CONFIGS.keys()}
    print(f"Training specialists for {len(survivors)} surviving groups")
    
    # HYBRID: Detect primary regime for this period
    primary_regime = detect_primary_regime(train_anchor, test_anchor)
    print(f"[Hybrid] Primary regime: {primary_regime} (determines which specialists to train)")
    
    # Create output directory
    specialists_dir = Path("models/regime_specialists")
    specialists_dir.mkdir(parents=True, exist_ok=True)
    
    # Train 3 specialists for each surviving group
    for survivor in survivors:
        group_id = survivor["group_id"]
        stock_list = survivor["stocks"]
        
        print(f"\n--- Training specialists for Group {group_id}: {stock_list} ---")
        
        # Load and preprocess data for this group
        all_X_train = []
        all_y_train = []
        all_X_val = []
        all_y_val = []
        all_regime_train = []
        all_regime_val = []
        
        for ticker in stock_list:
            try:
                loader = DataLoader(data_path, ticker)
                raw_df = loader.load_data()
                
                if train_anchor is not None:
                    mask = pd.to_datetime(raw_df.index, errors="coerce") <= pd.Timestamp(train_anchor)
                    raw_df = raw_df.loc[mask]
                
                if len(raw_df) < 500:
                    continue
                
                enriched = FeatureEngineer().compute_indicators(raw_df)
                pre = DataPreprocessor(lookback_window=24, forecast_horizon=8)
                splits = pre.preprocess(enriched)
                
                if len(splits["X_train"]) < 100:
                    continue
                
                # Get labels
                labels_dict, _, _ = make_direction_onehot_from_raw(
                    enriched,
                    {
                        "idx_train": splits["idx_train"],
                        "idx_val": splits["idx_val"],
                        "idx_test": splits["idx_test"],
                    },
                    lookback_window=24,
                    forecast_horizon=8,
                )
                    
                # HYBRID REGIME DETECTION
                # Primary: Period-based (determined by detect_primary_regime above)
                # Secondary: Sample-based (volatility percentile within this period)
                sharpe_col = -3
                vol_col = -4

                X_train_ticker = splits["X_train"]
                sharpe_mean = np.mean(X_train_ticker[:, :, sharpe_col], axis=1)
                vol_mean = np.mean(X_train_ticker[:, :, vol_col], axis=1)

                # Apply PRIMARY regime to all samples in this period
                # This ensures BEAR/CRISIS specialists get samples when training on bear market data
                primary_regime_idx = {"BULL": 0, "BEAR": 1, "CRISIS": 2}.get(primary_regime, 0)
                regime_train_ticker = np.full(len(X_train_ticker), primary_regime_idx, dtype=int)

                # SECONDARY: Add volatility percentile for fine-grained routing
                # Store in secondary_regime column (not used for filtering, but for routing confidence)
                vol_percentile = np.percentile(vol_mean, [33, 67])
                secondary_regime_train = np.zeros(len(X_train_ticker), dtype=int)
                secondary_regime_train[vol_mean >= vol_percentile[1]] = 2  # High vol
                secondary_regime_train[vol_mean >= vol_percentile[0]] = 1  # Medium vol

                all_X_train.append(X_train_ticker)
                all_y_train.append(labels_dict["train"])
                all_regime_train.append(regime_train_ticker)

                X_val_ticker = splits["X_val"]
                sharpe_mean_val = np.mean(X_val_ticker[:, :, sharpe_col], axis=1)
                vol_mean_val = np.mean(X_val_ticker[:, :, vol_col], axis=1)

                # Apply same PRIMARY regime to validation data
                regime_val_ticker = np.full(len(X_val_ticker), primary_regime_idx, dtype=int)

                all_X_val.append(X_val_ticker)
                all_y_val.append(labels_dict["val"])
                all_regime_val.append(regime_val_ticker)
                
            except Exception as e:
                print(f"  Failed to load {ticker}: {e}")
                continue
        
        if not all_X_train:
            print(f"  No valid data for group {group_id}, skipping")
            continue
        
        # Concatenate
        X_train = np.concatenate(all_X_train)
        y_train = np.concatenate(all_y_train)
        regime_train = np.concatenate(all_regime_train)
        X_val = np.concatenate(all_X_val)
        y_val = np.concatenate(all_y_val)
        regime_val = np.concatenate(all_regime_val)
        
        # HYBRID: Decide which specialists to train based on PRIMARY regime
        # Skip specialists that don't match the primary regime context
        MIN_SAMPLES_THRESHOLD = 100
        specialists_to_train = []
        
        # BULL_JOINT always trains (acts as fallback)
        specialists_to_train.append("BULL_JOINT")
        
        # Only train regime-specific specialists if primary regime matches
        if primary_regime == "BEAR":
            specialists_to_train.append("BEAR_ISOLATED")
        elif primary_regime == "CRISIS":
            specialists_to_train.append("CRISIS_ISOLATED")
        
        print(f"[Hybrid] Training specialists: {specialists_to_train} (primary={primary_regime})")
        
        # Train each specialist type
        for specialist_name in specialists_to_train:
            specialist_config = SPECIALIST_CONFIGS[specialist_name]
            print(f"\n [{specialist_name}] Training...")

            try:
                # Create specialist
                specialist = RegimeSpecialistLSTM(
                    specialist_name=specialist_name,
                    base_model_dir=str(specialists_dir / specialist_name / f"group_{group_id}"),
                )
                specialist.build_model(input_shape=(24, X_train.shape[2]))
                
                # Initialize from SAML if available
                if saml_learner is not None:
                    try:
                        saml_learner.initialize_specialist(specialist.model)
                        print(f"    Initialized from SAML meta-weights")
                    except Exception as e:
                        print(f"    SAML init failed: {e}, using random init")
                
                # Train with regime labels
                specialist.train(
                    X_train=X_train,
                    y_train_classification=y_train,
                    X_val=X_val,
                    y_val_classification=y_val,
                    regime_train=regime_train,
                    regime_val=regime_val,
                    epochs=50,
                    batch_size=32,
                    early_stopping_patience=5,
                )
                
                # Save
                specialist.save()
                specialist_paths[specialist_name][group_id] = str(specialist.model_path)
                print(f"    [OK] Complete: {specialist.model_path}")
                
            except Exception as e:
                print(f"    [X] Training failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary
    print("\n" + "=" * 60)
    print("SPECIALIST TRAINING COMPLETE")
    print("=" * 60)
    for specialist_name in SPECIALIST_CONFIGS.keys():
        count = len(specialist_paths[specialist_name])
        print(f"{specialist_name}: {count} groups trained")
    
    return specialist_paths


# --------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical grouped training pipeline")
    parser.add_argument("--seed",            type=int,   default=42,   help="Global RNG seed (vary to test different stock draws)")
    parser.add_argument("--n_stocks",        type=int,   default=100)
    parser.add_argument("--group_size",      type=int,   default=5)
    parser.add_argument("--threshold",       type=float, default=0.0,  help="Min val Sharpe to survive")
    parser.add_argument("--data_path",       type=str,   default=DATA_PATH)
    parser.add_argument("--sentiment_alpha", type=float, default=0.0,  help="FinBERT sentiment gate weight (0=off, 0.25=recommended)")
    parser.add_argument("--bear",    action="store_true", help="Retrain and evaluate on 2018 Q4 bear market window")
    parser.add_argument("--covid",   action="store_true", help="Retrain and evaluate on 2020 Q1-Q2 COVID crash window")
    parser.add_argument("--full",    action="store_true", help="Use full dataset (60/20/20 proportional split)")
    parser.add_argument("--live",    action="store_true", help="Fetch recent data via yfinance (2yr hourly) and evaluate on present-day market (13/5/6 month split)")
    parser.add_argument("--custom",  type=str, default=None, metavar="DDMMYYYY",
                        help="Custom test end date (DDMMYYYY). Pipeline splits backwards from this date.")
    parser.add_argument("--compare", action="store_true", help="Run both LSTM and MTL on the same groups and print a side-by-side comparison")
    parser.add_argument(
        "--mtl",
        action="store_true",
        help="Use shared/private Multi-Task Learning LSTM (trains one model across all groups)",
    )
    parser.add_argument(
        "--glm",
        action="store_true",
        help="Enable GLM AI signal review layer (requires GLM_API_KEY env var)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep selector after backtest to auto-optimize parameters (saves best to models/optimized_config.json)",
    )
    parser.add_argument(
        "--oos_holdout_pct",
        type=float,
        default=0.0,
        help="Reserve last N%% of test as true OOS holdout (0=disabled, 15=recommended)",
    )
    parser.add_argument(
        "--train_all_specialists",
        action="store_true",
        help="Train complete regime-specialist pipeline: SAML -> Quantile Barriers -> Base LSTM -> 3 Specialists -> Backtest with routing",
    )
    args = parser.parse_args()

    # Mutual-exclusion guards
    if args.bear and args.covid:
        parser.error("Use --bear or --covid, not both.")
    if args.full and (args.bear or args.covid):
        parser.error("--full cannot be combined with --bear or --covid.")
    if args.live and (args.bear or args.covid or args.full):
        parser.error("--live cannot be combined with --bear, --covid, or --full.")
    if args.custom and (args.bear or args.covid or args.full or args.live):
        parser.error("--custom cannot be combined with --bear, --covid, --full, or --live.")
    if args.compare and args.mtl:
        parser.error("--compare already runs MTL internally; do not combine with --mtl.")

    # Set anchor dates and run-mode labels
    split_months = (12, 2, 3)  # default: DataPreprocessor defaults (train/val/test months)

    if args.live:
        train_anchor   = None
        test_anchor    = None
        curve_name     = "equity_curve_live.png"
        use_full_split = False
        split_months   = (12, 4, 6)

        # Download recent data via yfinance
        from yfinance_loader import download_yfinance_tickers, get_yfinance_data_path

        print("\n>>> LIVE MODE: Fetching recent data via yfinance")
        print("    Split: 12-month train / 4-month val / 6-month test")
        print("    Data source: Yahoo Finance (~2 years hourly OHLCV)\n")

        # Check if yfinance data already cached from a previous run
        yf_cache_dir = Path(get_yfinance_data_path()) / "yfinance_1hour"
        from data_loader import DataLoader
        if yf_cache_dir.exists() and len(list(yf_cache_dir.glob("*_1hour.txt"))) >= args.n_stocks:
            cached_count = len(list(yf_cache_dir.glob("*_1hour.txt")))
            print(f"Reusing {cached_count} cached yfinance tickers from {yf_cache_dir}")
            args.data_path = get_yfinance_data_path()
        else:
            # Step 1: Volume filter on local data to get the liquid universe
            print("Volume-filtering tickers from local data...")
            available_symbols = DataLoader.list_available_stocks(args.data_path)
            vol_pool = []
            for sym in available_symbols:
                try:
                    loader = DataLoader(args.data_path, sym)
                    fp = loader._find_data_file()
                    df = pd.read_csv(fp, header=None, usecols=[5], names=["Volume"])
                    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                    avg_vol = df["Volume"].mean(skipna=True)
                    if pd.notna(avg_vol) and avg_vol >= 300_000:
                        vol_pool.append(sym)
                except Exception:
                    pass
            print(f"Found {len(vol_pool)} liquid stocks (avg volume >= 300K)")

            # Step 2: Download yfinance data for all liquid tickers
            successful = download_yfinance_tickers(vol_pool, delay=0.2)
            if len(successful) < args.group_size:
                raise ValueError(
                    f"Only {len(successful)} tickers downloaded successfully; "
                    f"need at least {args.group_size} for one group."
                )
            print(f"Downloaded {len(successful)} tickers via yfinance")

            # Point data_path to yfinance cache
            args.data_path = get_yfinance_data_path()

            # Clear scan cache so beta is calculated on live data
            scan_cache = Path("models/stock_scan_cache.json")
            if scan_cache.exists():
                scan_cache.unlink()

        print(f"Data path: {args.data_path}")

    elif args.bear:
        train_anchor   = BEAR_TRAIN_END
        test_anchor    = BEAR_TEST_END
        curve_name     = "equity_curve_bear.png"
        use_full_split = False
        split_months   = (12, 2, 3)   # anchor-date split; sets min_test_hours correctly
        print(f"\n>>> BEAR MARKET RUN (train \u2264 {train_anchor}, test \u2264 {test_anchor})")
    elif args.covid:
        train_anchor   = COVID_TRAIN_END
        test_anchor    = COVID_TEST_END
        curve_name     = "equity_curve_covid.png"
        use_full_split = False
        split_months   = (12, 2, 6)   # anchor-date split; sets min_test_hours correctly
        print(f"\n>>> COVID CRASH RUN (train \u2264 {train_anchor}, test \u2264 {test_anchor})")
    elif args.custom:
        from datetime import datetime as _dt
        try:
            custom_date = _dt.strptime(args.custom, "%d%m%Y")
        except ValueError:
            parser.error(f"Invalid date format '{args.custom}'. Use DDMMYYYY (e.g. 31012020).")

        test_anchor  = custom_date.strftime("%Y-%m-%d")
        split_months = (12, 4, 6)
        from dateutil.relativedelta import relativedelta
        train_anchor_dt = custom_date - relativedelta(months=split_months[2])
        train_anchor    = train_anchor_dt.strftime("%Y-%m-%d")

        curve_name     = "equity_curve_custom.png"
        use_full_split = False
        print(f"\n>>> CUSTOM RUN (train <= {train_anchor}, test <= {test_anchor})")
        print(f"    Split: {split_months[0]}mo train / {split_months[1]}mo val / {split_months[2]}mo test")

    elif args.full:
        train_anchor   = None
        test_anchor    = None
        curve_name     = "equity_curve_full.png"
        use_full_split = True
        print(
            "\n>>> FULL DATA RUN (60/20/20 proportional split across all available years)\n"
            "    Train  : ~60% of data (~9.6 years for 2005-2021 datasets)\n"
            "    Val    : ~20% of data (~3.2 years)\n"
            "    Test   : ~20% of data (~3.2 years)"
        )
    else:
        train_anchor   = None
        test_anchor    = None
        curve_name     = "equity_curve.png"
        use_full_split = False

    if args.compare:
        print("\n>>> COMPARE MODE: running LSTM then MTL on identical groups")
    elif args.mtl:
        print("\n>>> MTL MODE: shared/private Multi-Task Learning LSTM")

    # ---- Stage 1: Cluster -----------------------------------------------
    groups = cluster_stocks(
        data_path=args.data_path,
        n_stocks=args.n_stocks,
        group_size=args.group_size,
        anchor_end_date=train_anchor,
    )

    # ---- Compare mode (LSTM + MTL, same groups) -------------------------
    if args.compare:
        run_compare(
            groups=groups,
            selection_threshold=args.threshold,
            data_path=args.data_path,
            sentiment_alpha=args.sentiment_alpha,
            train_anchor=train_anchor,
            test_anchor=test_anchor,
            use_full_split=use_full_split,
        )
        import sys; sys.exit(0)

# ---- Train All Specialists Pipeline (if enabled) -----------------------
# This mode replaces the standard tournament with full regime-specialist training
saml_learner_for_backtest = None
qbarrier_learner_for_backtest = None
mtl_model = None

if args.train_all_specialists:
    print("\n" + "=" * 60)
    print(">>> TRAIN ALL SPECIALISTS MODE")
    print("=" * 60)
    print("Pipeline: SAML -> Quantile -> Base LSTM -> 3 Specialists -> Backtest")

    # Step 1: Train/Load SAML
    print("\n--- Step 1: Training SAML Meta-Learner ---")
    saml_learner = train_or_load_saml(
        groups=groups,
        data_path=args.data_path,
        anchor_end_date=train_anchor,
        use_full_split=use_full_split,
        split_months=split_months,
    )
    saml_learner_for_backtest = saml_learner

    # Step 2: Train/Load Quantile Barriers
    print("\n--- Step 2: Training Quantile Barrier Learner ---")
    quantile_learner = train_or_load_quantile_barriers(
        groups=groups,
        data_path=args.data_path,
        anchor_end_date=train_anchor,
        use_full_split=use_full_split,
        split_months=split_months,
    )
    qbarrier_learner_for_backtest = quantile_learner

    # Step 3: Train Base LSTM with SAML initialization
    print("\n--- Step 3: Training Base LSTM (initialized from SAML) ---")
    survivors = run_tournament(
        groups=groups,
        selection_threshold=args.threshold,
        data_path=args.data_path,
        sentiment_alpha=args.sentiment_alpha,
        anchor_end_date=train_anchor,
        use_full_split=use_full_split,
        split_months=split_months,
        use_min_trades=args.live or bool(args.custom),
        use_walk_forward=args.live or bool(args.custom),
        saml_learner=saml_learner,  # NEW: Initialize from SAML
        quantile_learner=quantile_learner,  # NEW: Use quantile barriers
    )

    # Step 4: Train 3 Specialists for each surviving group
    if survivors:
        print("\n--- Step 4: Training 3 Regime Specialists per group ---")
        specialist_paths = run_specialist_training(
            groups=groups,
            survivors=survivors,
            data_path=args.data_path,
            train_anchor=train_anchor,
            test_anchor=test_anchor,
            use_full_split=use_full_split,
            split_months=split_months,
            saml_learner=saml_learner,
            quantile_learner=quantile_learner,
        )
        print("\n" + "=" * 60)
        print("SPECIALIST TRAINING COMPLETE")
        print("=" * 60)
        for specialist_name, paths in specialist_paths.items():
            print(f"{specialist_name}: {len(paths)} groups trained")
    else:
        print("\n[Warning] No survivors - skipping specialist training")

    # Step 5: Run backtest with routing
    args.saml = True  # Enable SAML for backtest
    args.learn_barriers = True  # Enable quantile barriers for backtest

else:
    # ---- Standard Stage 2: Tournament (standard or MTL) -----------------
    mtl_model = None
    if args.mtl:
        survivors, mtl_model = run_tournament_mtl(
            groups=groups,
            selection_threshold=args.threshold,
            data_path=args.data_path,
            sentiment_alpha=args.sentiment_alpha,
            anchor_end_date=train_anchor,
            use_full_split=use_full_split,
        )
        # Only reload from disk if the live model somehow wasn't returned
        if survivors and mtl_model is None:
            from mtl_lstm_model import MTLLSTMModel
            from data_loader import DataLoader
            from feature_engineer import FeatureEngineer
            from ensemble_model import _make_preprocessor
            first_ticker = groups[0][0] if groups else None
            if first_ticker:
                try:
                    loader = DataLoader(args.data_path, first_ticker)
                    enriched = FeatureEngineer().compute_indicators(loader.load_data())
                    _, splits, _ = _make_preprocessor(LOOKBACK, HORIZON, enriched, train_anchor, use_full_split)
                    n_feats = splits["X_train"].shape[2]
                    group_ids = [r["group_id"] for r in survivors]
                    mtl_model = MTLLSTMModel.load(
                        save_dir="models/mtl",
                        input_shape=(LOOKBACK, n_feats),
                        group_ids=group_ids,
                    )
                except Exception as e:
                    print(f"[MTL] Could not reload model for Stage 3: {e}")
    else:
        survivors = run_tournament(
            groups=groups,
            selection_threshold=args.threshold,
            data_path=args.data_path,
            sentiment_alpha=args.sentiment_alpha,
            anchor_end_date=train_anchor,
            use_full_split=use_full_split,
            split_months=split_months,
            use_min_trades=args.live or bool(args.custom),
            use_walk_forward=args.live or bool(args.custom),
        )

# ---- GLM guard (optional) -----------------------------------------------
# This runs for BOTH specialist and standard pipelines
glm_guard = None
if args.glm:
    from llm_guard import GLMGuard
    try:
        glm_guard = GLMGuard()
        print("\n[GLM] Guard enabled -- signals will be reviewed before backtest execution.")
    except ValueError as e:
        print(f"\n[GLM] Could not initialise guard: {e}")
        print("[GLM] Continuing without GLM review. Set GLM_API_KEY env var.")

# ---- Stage 3: Combined backtest -------------------------------------
# This runs for BOTH specialist and standard pipelines
if survivors:
    # Enable regime routing if --train_all_specialists was used
    use_routing = args.train_all_specialists

    metrics = run_combined_backtest(
        survivors=survivors,
        data_path=args.data_path,
        anchor_end_date=test_anchor,
        equity_curve_name=curve_name,
        sentiment_alpha=args.sentiment_alpha,
        train_anchor_date=train_anchor,
        use_full_split=use_full_split,
        mtl_model=mtl_model,
        split_months=split_months,
        glm_guard=glm_guard,
        oos_holdout_pct=args.oos_holdout_pct,
        use_regime_routing=use_routing,
        saml_meta_learner=saml_learner_for_backtest,
        quantile_learner=qbarrier_learner_for_backtest,
    )

    # ---- Optional: Sweep selector for auto-optimization ------------
    if args.sweep and metrics.get("sharpe_ratio", 0) > 0:
        print("\n" + "=" * 60)
        print("AUTO-SWEEP: Running parameter optimization...")
        print("=" * 60)
        try:
            import subprocess
            sweep_args = ["python", "src/sweep_selector.py"]
            if args.bear:
                sweep_args.append("--bear")
            if args.live:
                sweep_args.append("--live")
            if args.threshold > 0:
                sweep_args += ["--threshold", str(args.threshold)]
            if args.oos_holdout_pct > 0:
                sweep_args += ["--oos_holdout_pct", str(args.oos_holdout_pct)]
            if args.sentiment_alpha > 0:
                sweep_args += ["--sentiment_alpha", str(args.sentiment_alpha)]
            result = subprocess.run(
                sweep_args,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for full sweep
            )
            print(result.stdout)
            if result.returncode == 0:
                print("\n[Sweep] Auto-optimization complete. Best config saved to models/optimized_config.json")
            else:
                print(f"\n[Sweep] Sweep completed with warnings: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("\n[Sweep] Sweep timed out after 1 hour -- continuing without auto-optimization")
        except Exception as e:
            print(f"\n[Sweep] Sweep failed: {e} -- continuing without auto-optimization")
else:
    print("\nNo groups survived selection. Adjust --threshold or train more groups.")