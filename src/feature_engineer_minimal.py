"""
Feature Engineering Pipeline for Time-Series Financial Data

This module processes raw Open, High, Low, Close, Volume (OHLCV) market data and extracts 
statistically significant predictive features. The feature selection methodology relies on 
empirical findings from regularized logistic regression (LASSO) and Random Forest feature 
importance tests.

The pipeline outputs two distinct categories of features:
1. Top-ranked Technical Analysis (TA-Lib) indicators that capture momentum, volume, and volatility.
2. Custom momentum and regime-context metrics based on continuous logarithmic returns, 
   which academic literature frequently identifies as highly effective inputs for LSTM networks.
"""

import numpy as np
import pandas as pd
import talib
from pathlib import Path
from typing import List, Dict

# Bump this if compute_indicators logic changes — invalidates cached parquet files.
_FE_CACHE_VERSION = "v2"

class FeatureEngineer:
    """
    Computes a streamlined subset of predictive indicators, systematically avoiding 
    multicollinearity and dimensional noise by dropping historically weak features 
    (e.g., simple moving averages, lagging stochastic oscillators, and redundant OHLC inputs).
    """

    # Top-tier TA-Lib indicators isolated via empirical feature ablation testing
    INDICATOR_COLUMNS = [
        'RSI_14',
        'MACD', 'MACD_signal', 'MACD_hist',
        'ATR_14',
        'BB_Width',
        'BB_PctB'
    ]
    
    # Custom statistical features designed to provide multi-scale momentum and regime context
    RETURN_VOL_COLUMNS = [
        'Close_to_SMA20'
    ]
    
    # Volatility regime features to handle market regime shifts
    REGIME_COLUMNS = [
        'vol_percentile_60',
    ]

    # EXP1: Additional predictive signals
    SIGNAL_COLUMNS = [
        'vol_surge',
        'price_accel',
        'rel_strength',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
    ]

    def __init__(self):
        self._feature_columns: List[str] = []

    def compute_frac_diff(self, series: pd.Series, d: float = 0.4, thres: float = 1e-4) -> pd.Series:
        """
        Fractional Differencing: Makes the price series stationary while preserving long-term memory.
        Calculates an expanding window of weights based on the binomial series.
        """
        w = [1.]
        for k in range(1, len(series)):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thres:
                break
            w.append(w_k)
        
        w = np.array(w[::-1])
        L = len(w)
        
        res = np.full_like(series.values, np.nan, dtype=float)
        if len(series) >= L:
            out = np.convolve(series.values, w, mode='valid')
            res[L-1:] = out
            
        return pd.Series(res, index=series.index)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw OHLCV pricing data into a dense, normalized feature matrix.

        Uses an on-disk parquet cache when the input DataFrame carries
        DataLoader provenance attrs (``_dl_ticker``, ``_dl_raw_mtime``,
        ``_dl_cache_root``).  Cache key = ticker + raw-CSV mtime + FE version.
        Output is byte-identical to a fresh computation.

        Args:
            df: Raw DataFrame containing ['Open', 'High', 'Low', 'Close', 'Volume']

        Returns:
            pd.DataFrame: Enriched DataFrame stripped of NaN warmup rows.
        """
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Input DataFrame must contain: {required}")

        # ---- cache lookup --------------------------------------------------
        cache_path = None
        try:
            ticker     = df.attrs.get('_dl_ticker')
            raw_mtime  = df.attrs.get('_dl_raw_mtime')
            cache_root = df.attrs.get('_dl_cache_root')
            if ticker and raw_mtime and cache_root:
                cache_path = (
                    Path(cache_root) / f"enriched_{_FE_CACHE_VERSION}" / f"{ticker}.parquet"
                )
                if cache_path.exists() and cache_path.stat().st_mtime >= raw_mtime:
                    cached = pd.read_parquet(cache_path)
                    self._feature_columns = cached.columns.tolist()
                    print(f"[FE cache hit] {ticker}  ({len(cached)} rows)")
                    return cached
        except Exception:
            cache_path = None  # any failure → recompute below

        out = pd.DataFrame(index=df.index)

        # Retain foundational vectors
        out['Close'] = df['Close'].astype(float)
        out['Volume'] = df['Volume'].astype(float)

        # Convert to contiguous NumPy arrays for C-optimized TA-Lib execution
        high   = df['High'].values.astype(float)
        low    = df['Low'].values.astype(float)
        close  = df['Close'].values.astype(float)
        volume = df['Volume'].values.astype(float)

        # ---------------------------------------------------------------------
        # 1. TECHNICAL INDICATORS (Momentum, Volatility, Volume, Statistical)
        # ---------------------------------------------------------------------
        
        # Relative Strength Index (Momentum)
        out['RSI_14'] = talib.RSI(close, timeperiod=14)

        # Moving Average Convergence Divergence (Trend/Momentum)
        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        out['MACD']        = macd
        out['MACD_signal'] = macd_signal
        out['MACD_hist']   = macd_hist


        # On-Balance Volume — differenced (1-period) to remove cumulative non-stationarity.
        # Raw OBV grows unboundedly; differencing yields stationary volume-flow signal.
        raw_obv = talib.OBV(close, volume)
        out['OBV_diff'] = pd.Series(raw_obv, index=df.index).diff()
        
        # Average True Range (Absolute Volatility) - used for adaptive barriers
        out['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

        # Bollinger Bands (Volatility and Breakout Context)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        out['BB_Width'] = (upper - lower) / middle
        diff = upper - lower
        diff[diff == 0] = 1e-8
        # ---------------------------------------------------------------------
        # 2. CONTINUOUS LOG RETURNS & CONTEXTUAL FEATURES
        # ---------------------------------------------------------------------
        
        # Close-to-SMA20 ratio: normalised distance from 20-period mean
        sma20 = out['Close'].rolling(window=20).mean()
        out['Close_to_SMA20'] = (out['Close'] - sma20) / sma20

        # ---------------------------------------------------------------------
        # 3. VOLATILITY REGIME FEATURES
        # ---------------------------------------------------------------------
        
        returns = out['Close'].pct_change()
        rolling_vol = returns.rolling(window=60).std()
        
        # Volatility percentile rank over 60 periods (bounded 0-1, stationary)
        out['vol_percentile_60'] = rolling_vol.rolling(window=60).rank(pct=True)

        # -------------------------------------------------------------
        # 4. ADDITIONAL PREDICTIVE SIGNALS (EXP1/EXP2)
        # -------------------------------------------------------------
        out['vol_surge'] = out['Volume'] / out['Volume'].rolling(100).mean().clip(lower=1e-8)
        out['price_accel'] = out['Close'].pct_change(3) - out['Close'].pct_change(6)
        out['hl_spread'] = (high - low) / close.clip(min=1e-8)
        ret_1h = out['Close'].pct_change()
        out['rel_strength'] = (ret_1h - ret_1h.rolling(50).mean()) / ret_1h.rolling(50).std().clip(lower=1e-8)
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour.values.astype(float)
            dow = df.index.dayofweek.values.astype(float)
            out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            out['dow_sin'] = np.sin(2 * np.pi * dow / 5)
            out['dow_cos'] = np.cos(2 * np.pi * dow / 5)
        else:
            for f in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
                out[f] = 0.0
        close_series = pd.Series(close, index=df.index)
        prev_close = close_series.shift(1).fillna(close[0])
        out['overnight_gap'] = (df['Open'].astype(float) - prev_close.values) / prev_close.values.clip(min=1e-8)
        out['macd_divergence'] = out['MACD_hist'].diff(3)

        # EXP9: Feature interactions for non-linear patterns
        out['hour_x_macd'] = out['hour_sin'] * out['MACD_hist']
        out['vol_x_price'] = out['vol_surge'] * out['Close_to_SMA20']
        out['accel_x_rel'] = out['price_accel'] * out['rel_strength']
        out['gap_x_vol'] = out['overnight_gap'] * out['vol_surge']

        # Non-linear transformations
        out['rsi_squared'] = out['RSI_14'] ** 2
        out['volpct_squared'] = out['vol_percentile_60'] ** 2
        out['macd_trend_x'] = out['MACD_signal'] * out['MACD_hist']

        # Truncate preliminary rows containing NaNs
        rows_before = len(out)
        out = out.dropna()
        rows_dropped = rows_before - len(out)

        self._feature_columns = out.columns.tolist()

        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING COMPLETE (TOP INDICATORS + RETURNS/VOL + REGIME)")
        print(f"{'='*60}")
        print(f"  Total features      : {len(self._feature_columns)}")
        print(f"    Base              : Close, Volume (2)")
        print(f"    TA Indicators     : {len(self.INDICATOR_COLUMNS)}")
        print(f"    Returns/Vol       : {len(self.RETURN_VOL_COLUMNS)}")
        print(f"    Regime Features   : {len(self.REGIME_COLUMNS)}")
        print(f"    Signal Features   : {len(self.SIGNAL_COLUMNS)}")
        print(f"  Warmup rows dropped : {rows_dropped}")
        print(f"  Remaining rows      : {len(out)}")
        print(f"\n  Feature columns: {self._feature_columns}")

        # ---- cache write (best-effort) ------------------------------------
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(cache_path)
            except Exception as e:
                print(f"  (FE cache write failed: {e})")

        return out

    def get_feature_columns(self) -> List[str]:
        """Returns a stable snapshot of the engineered column index."""
        if not self._feature_columns:
            raise ValueError("No features computed yet. Call compute_indicators() first.")
        return self._feature_columns.copy()

    def get_close_index(self) -> int:
        """Returns the dimensional index of the Close price column in the feature matrix."""
        return self._feature_columns.index('Close')

    def get_atr_index(self) -> int:
        """Returns the dimensional index of the ATR_14 column in the feature matrix."""
        if not self._feature_columns:
            raise ValueError("No features computed yet. Call compute_indicators() first.")
        return self._feature_columns.index('ATR_14')

    def get_indicator_info(self) -> Dict[str, dict]:
        """Provides metadata mappings for all configured features."""
        return {
            'RSI_14':            {'category': 'Momentum',   'talib_fn': 'RSI',    'params': 'period=14'},
            'MACD':              {'category': 'Momentum',   'talib_fn': 'MACD',   'params': '12,26,9 line'},
            'MACD_signal':       {'category': 'Momentum',   'talib_fn': 'MACD',   'params': '12,26,9 signal'},
            'MACD_hist':         {'category': 'Momentum',   'talib_fn': 'MACD',   'params': '12,26,9 hist'},
            'ROC_10':            {'category': 'Momentum',   'talib_fn': 'ROC',    'params': 'period=10'},
            'OBV_diff':          {'category': 'Volume',     'talib_fn': 'OBV',    'params': '1-period diff'},
            'ATR_14':            {'category': 'Volatility', 'talib_fn': 'ATR',    'params': 'period=14'},
            'CORREL_30':         {'category': 'Statistical','talib_fn': 'CORREL', 'params': 'period=30'},
            'BB_Width':          {'category': 'Volatility', 'talib_fn': 'BBANDS', 'params': 'period=20'},
            'BB_PctB':           {'category': 'Volatility', 'talib_fn': 'BBANDS', 'params': '%B period=20'},
            'return_24h_lagged': {'category': 'Returns',    'method': 'Pandas',   'params': 'log(close/close.shift(24))'},
            'Close_to_SMA20':    {'category': 'Statistical','method': 'Pandas',   'params': '(close - SMA20) / SMA20'},
            'vol_percentile_60': {'category': 'Regime',     'method': 'Pandas',   'params': 'Rolling vol rank pct (60-period)'},
            'sharpe_ratio_20':   {'category': 'Regime',     'method': 'Pandas',   'params': 'Rolling Sharpe ratio (20-period)'},
        }

if __name__ == "__main__":
    from data_loader import DataLoader

    try:
        loader = DataLoader("data/raw", "AAPL")
        raw_df = loader.load_data()
        loader.validate_data()

        fe = FeatureEngineer()
        enriched_df = fe.compute_indicators(raw_df)

        print(f"\nEnriched DataFrame shape: {enriched_df.shape}")
        print(f"\nFirst row after warmup:\n{enriched_df.iloc[0].to_string()}")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
