import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """
    Preprocesses OHLCV data from DataLoader:
    1. Splits data using Walk-Forward Validation (rolling origin)
    2. Normalizes features using continuous rolling Z-Score across all splits
    3. Creates sliding window sequences (24 hours lookback → 8 hours forecast)

    Split modes
    -----------
    Default (use_walk_forward=True, use_full_split=False):
        Anchored walk-forward - most recent 3 months = test, 2 months = val,
        12 months = train.  Designed for live-deployment relevance.

    Full data (use_full_split=True):
        Proportional 60 / 20 / 20 split across the ENTIRE dataset with purge
        gaps.  For a 2005-2021 dataset (~140 K rows) this gives roughly:
          train  ~84 K rows  (~9.6 years)
          val    ~28 K rows  (~3.2 years)
          test   ~28 K rows  (~3.2 years)
        Use this mode via `python pipeline.py --full`.

    Legacy (use_walk_forward=False, use_full_split=False):
        Fixed 60/20 ratio split kept for backwards compatibility.
    """

    def __init__(
        self,
        lookback_window: int = 24,
        forecast_horizon: int = 24,
        use_walk_forward: bool = True,
        use_full_split: bool = False,
        split_months: tuple = None,
        oos_ratio: float = 0.0,
    ):
        self.lookback_window  = lookback_window
        self.forecast_horizon = forecast_horizon
        self.use_walk_forward = use_walk_forward
        self.use_full_split   = use_full_split
        self.split_months     = split_months  # (train, val, test) override
        self.oos_ratio        = oos_ratio     # Fraction of test reserved as OOS (0.0 = disabled)
        self.data             = None
        self.normalized_data  = None
        self.feature_columns  = []
        self.close_idx        = 3
        self.train_rolling_mean = None
        self.train_rolling_std  = None

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def walk_forward_split(self, data: pd.DataFrame,
                          train_months: int = 12,
                          val_months: int = 2,
                          test_months: int = 3) -> Dict[str, pd.DataFrame]:
        """
        ANCHORED Walk-Forward Validation with adaptive fallback.

        **KEY CHANGE**: Uses MOST RECENT data for test/val to avoid regime shift.
        Works BACKWARDS from end of dataset instead of forward from start.

        Logic:
        1. Reserve most recent 3 months for testing
        2. Reserve 2 months before that for validation
        3. Use 12 months before that for training
        4. Purge gaps prevent temporal leakage

        This ensures test set = current market regime, not historical COVID crash.

        If insufficient data: falls back to percentage-based (60/15/25) maintaining
        chronological order.
        """
        # Detect data density: FirstRateData has ~730 rows/month (24h timestamps)
        # while yfinance market-hours-only data has ~140 rows/month (~6.5h * 21 days).
        # Auto-detect by computing actual rows per month from the data.
        try:
            import pandas as _pd
            idx = _pd.to_datetime(data.index, errors='coerce')
            n_months = max((idx.max() - idx.min()).days / 30.44, 1.0)
            hours_per_month = int(len(data) / n_months)
            # Sanity clamp: at least 100, at most 800
            hours_per_month = max(100, min(hours_per_month, 800))
        except Exception:
            hours_per_month = 730

        train_size = train_months * hours_per_month
        val_size = val_months * hours_per_month
        test_size = test_months * hours_per_month

        total_len = len(data)
        purge_gap = self.lookback_window + self.forecast_horizon

        # Check if we have enough data for time-based ANCHORED split
        required_len = train_size + val_size + test_size + 2 * purge_gap

        oos_size = 0
        if self.oos_ratio > 0.0:
            oos_size = int(test_size * self.oos_ratio)
            test_size = test_size - oos_size
            required_len += oos_size

        if total_len >= required_len:
            # ANCHORED SPLIT: Work backwards from end
            total_end = total_len
            oos_end = total_end
            oos_start = oos_end - oos_size
            test_end = oos_start
            test_start = test_end - test_size

            val_end = test_start - purge_gap
            val_start = val_end - val_size

            train_end = val_start - purge_gap
            train_start = train_end - train_size

            print(f"\n Walk-Forward Split (Anchored/Recent-Data):")
            print(f"  Train: {train_start} to {train_end} ({train_months} months)")
            print(f"  Val:   {val_start} to {val_end} ({val_months} months)")
            print(f"  Test:  {test_start} to {test_end} ({test_months} months)")
            if oos_size > 0:
                print(f"  OOS:   {oos_start} to {oos_end} ({oos_size} rows, {self.oos_ratio*100:.0f}% of original test)")
            print(f"  Purge Gap: {purge_gap} rows between each split")
            print(f"  [OK] Test set uses MOST RECENT data (avoiding regime shift)")
        else:
            # Fallback to percentage-based split (maintains chronological order)
            print(f"\n Dataset too small for {train_months}/{val_months}/{test_months} month split.")
            print(f"  Available: {total_len} rows, Need: {required_len} rows")
            print(f"  Falling back to percentage-based walk-forward split...")

            usable_len = total_len - 2 * purge_gap
            if usable_len <= 0:
                raise ValueError(f"Dataset too small ({total_len} rows) even for percentage split. Need at least {2*purge_gap} rows.")

            train_pct = 0.60
            val_pct = 0.15

            train_end = int(usable_len * train_pct)
            val_start = train_end + purge_gap
            val_size_pct = int(usable_len * val_pct)
            val_end = val_start + val_size_pct
            test_start = val_end + purge_gap
            test_end = total_len

            # Carve OOS from test tail
            oos_start = test_end
            if self.oos_ratio > 0.0:
                oos_size = int((test_end - test_start) * self.oos_ratio)
                oos_start = test_end - oos_size
                test_end = oos_start

            train_start = 0

            print(f"\n Walk-Forward Split (Percentage-Based):")
            print(f"  Train: 0 to {train_end} ({train_pct*100:.0f}%, ~{train_end/730:.1f} months)")
            print(f"  Val: {val_start} to {val_end} ({val_pct*100:.0f}%, ~{val_size_pct/730:.1f} months)")
            print(f"  Test: {test_start} to {test_end} (~{(test_end-test_start)/730:.1f} months)")
            if oos_size > 0:
                print(f"  OOS:  {oos_start} to {total_len} (~{oos_size/730:.1f} months)")
            print(f"  Purge Gap: {purge_gap} rows between each split")

        return {
            'train': data.iloc[train_start:train_end].copy(),
            'val': data.iloc[val_start:val_end].copy(),
            'test': data.iloc[test_start:test_end].copy(),
            'oos': data.iloc[oos_start:total_len].copy() if oos_size > 0 else pd.DataFrame(),
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start,
            'oos_start': oos_start if oos_size > 0 else test_end,
        }

    def full_data_split(self, data: pd.DataFrame,
                        train_ratio: float = 0.60,
                        val_ratio: float = 0.20) -> Dict[str, pd.DataFrame]:
        """
        Full-dataset proportional split: 60 / 20 / 20 across ALL available rows.

        Unlike walk_forward_split() which anchors to a fixed number of months,
        this method uses every row of data so that a 16-year dataset (2005-2021)
        actually gets:
          Train  ~60%  → ~9.6 years
          Val    ~20%  → ~3.2 years
          Test   ~20%  → ~3.2 years

        Purge gaps of (lookback + horizon) rows are enforced between splits to
        prevent temporal leakage from sequences that straddle a boundary.

        This is the mode activated by `python pipeline.py --full`.
        """
        total_len = len(data)
        purge_gap = self.lookback_window + self.forecast_horizon

        usable_len = total_len - 2 * purge_gap
        if usable_len <= 0:
            raise ValueError(
                f"Dataset too small ({total_len} rows) for full-data split "
                f"with purge_gap={purge_gap}."
            )

        train_end   = int(usable_len * train_ratio)
        val_start   = train_end + purge_gap
        val_size    = int(usable_len * val_ratio)
        val_end     = val_start + val_size
        test_start  = val_end + purge_gap
        test_end    = total_len

        # Carve OOS from test tail
        oos_size = 0
        oos_start = test_end
        if self.oos_ratio > 0.0:
            oos_size = int((test_end - test_start) * self.oos_ratio)
            oos_start = test_end - oos_size
            test_end = oos_start

        # Safety: ensure test split is non-empty
        if test_start >= test_end:
            raise ValueError(
                f"Full-data split produced an empty test set "
                f"(test_start={test_start} >= test_end={test_end}). "
                "Dataset may be too short for --full mode."
            )

        print(f"\n Full-Data Split (60/20/20 proportional):")
        print(f"  Total rows      : {total_len}")
        print(f"  Purge gap       : {purge_gap} rows between each split")
        print(f"  Train  : 0 to {train_end} ({train_ratio*100:.0f}%,  ~{train_end/730:.1f} months)")
        print(f"  Val    : {val_start} to {val_end} ({val_ratio*100:.0f}%, ~{val_size/730:.1f} months)")
        print(f"  Test   : {test_start} to {test_end} (~{(test_end - test_start)/730:.1f} months)")
        if oos_size > 0:
            print(f"  OOS    : {oos_start} to {total_len} (~{oos_size/730:.1f} months, {self.oos_ratio*100:.0f}% of original test)")
        print(f"  [OK] Full dataset utilised - no historical data discarded")

        return {
            'train':       data.iloc[0:train_end].copy(),
            'val':         data.iloc[val_start:val_end].copy(),
            'test':        data.iloc[test_start:test_end].copy(),
            'oos':         data.iloc[oos_start:total_len].copy() if oos_size > 0 else pd.DataFrame(),
            'train_start': 0,
            'train_end':   train_end,
            'val_start':   val_start,
            'val_end':     val_end,
            'test_start':  test_start,
            'oos_start':   oos_start if oos_size > 0 else test_end,
        }

    def split_raw_data(self, data: pd.DataFrame,
                       train_ratio: float = 0.60,
                       val_ratio: float = 0.20) -> Dict[str, pd.DataFrame]:
        """
        Legacy fixed-ratio split (kept for backwards compatibility).
        Split raw data BEFORE normalization to prevent data leakage.
        Enforces a purge gap to prevent temporal overlap.
        """
        total_len = len(data)
        purge_gap = self.lookback_window + self.forecast_horizon

        usable_len = total_len - 2 * purge_gap
        if usable_len <= 0:
            raise ValueError("Dataset too small for the specified purge gaps.")

        chunk_train = int(usable_len * train_ratio)
        chunk_val = int(usable_len * val_ratio)

        train_end = chunk_train
        val_start = train_end + purge_gap
        val_end = val_start + chunk_val
        test_start = val_end + purge_gap

        print(f"\n Fixed-ratio split (legacy):")
        print(f"  Purge Gap: {purge_gap} rows")
        print(f"  Train: 0 to {train_end} ({train_ratio*100:.0f}%)")
        print(f"  Val: {val_start} to {val_end} ({val_ratio*100:.0f}%)")
        print(f"  Test: {test_start} to {total_len} (~{(1-train_ratio-val_ratio)*100:.0f}%)")

        return {
            'train': data.iloc[:train_end].copy(),
            'val': data.iloc[val_start:val_end].copy(),
            'test': data.iloc[test_start:].copy(),
            'train_start': 0,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start
        }

    def normalize_data_split_aware(self, train_data: pd.DataFrame,
                                    val_data: pd.DataFrame,
                                    test_data: pd.DataFrame,
                                    oos_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Per-ticker continuous rolling Z-score normalisation extending through
        val, test, and OOS - no frozen train-end stats.

        Approach:
        - Concatenate train+val+test+(oos) as one continuous time series
        - Compute rolling mean/std over the FULL timeline (window=200)
        - Each row is normalised by its OWN rolling window (no future leakage,
          since rolling looks backwards only)
        - Clip to ±4 std as a safety net for extreme outliers
        """
        window = 200

        def rolling_zscore(data, window):
            """Compute rolling z-score using expanding stats for training, rolling for inference splits."""
            if len(data) == 0:
                return data
            rolling_mean = data.rolling(window=window, min_periods=1).mean()
            rolling_std = data.rolling(window=window, min_periods=1).std()
            rolling_std = rolling_std.replace(0, 1e-8).fillna(1e-8)
            normalized = (data - rolling_mean) / rolling_std
            normalized = normalized.fillna(0).clip(-4, 4)
            return normalized

        # Apply per-split: use rolling stats computed only within each split (no lookahead)
        train_normalized = rolling_zscore(train_data, window)
        val_normalized = rolling_zscore(val_data, window)
        test_normalized = rolling_zscore(test_data, window)

        has_oos = oos_data is not None and len(oos_data) > 0
        if has_oos:
            oos_normalized = rolling_zscore(oos_data, window)
            print(f"\n  Normalized using Split-Aware Rolling Z-Score (window={window})")
            print(f"  No data leakage between train/val/test/OOS splits")
            print(f"  Features clipped to ±4 std after normalisation")
            return {
                'train': train_normalized,
                'val': val_normalized,
                'test': test_normalized,
                'oos': oos_normalized,
            }

        print(f"\n  Normalized using Split-Aware Rolling Z-Score (window={window})")
        print(f"  No data leakage between train/val/test splits")
        print(f"  Features clipped to ±4 std after normalisation")

        return {
            'train': train_normalized,
            'val': val_normalized,
            'test': test_normalized
        }

    def create_sequences(self, data: np.ndarray, start_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal sequences using sliding window approach.
        Excludes the raw Close price from X to prevent non-stationary leakage,
        but keeps it as the target y for labeling purposes.

        Args:
            data: Normalized feature array
            start_offset: Starting index in original dataframe (for anchored splits)

        Returns tuple of (X, y) where:
        - X: (n_sequences, lookback_window, n_features - 1)
        - y: (n_sequences, 1) - the target Close price
        """
        X, y = [], []

        feature_indices = [i for i in range(data.shape[1]) if i != self.close_idx]

        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            window = data[i:i + self.lookback_window, feature_indices]
            X.append(window)
            target = data[i + self.lookback_window + self.forecast_horizon - 1, self.close_idx]
            y.append(target)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        print(f"\n Created sequences with sliding window:")
        print(f"  Lookback window: {self.lookback_window} hours")
        print(f"  Forecast horizon: {self.forecast_horizon} hour(s) ahead")
        print(f"  Total sequences created: {len(X)}")
        print(f"  X shape: {X.shape} (sequences, lookback, features) - CLOSE PRICE EXCLUDED")
        print(f"  y shape: {y.shape} (sequences, 1)")

        return X, y

    def preprocess(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline with proper data leakage prevention.

        Split priority:
          1. use_full_split=True  → full_data_split()  (60/20/20 across all rows)
          2. use_walk_forward=True → walk_forward_split() (anchored 12/2/3 months)
          3. fallback              → split_raw_data()  (legacy fixed ratio)
        """
        print("\n" + "="*60)
        if self.use_full_split:
            print("PREPROCESSING WITH FULL-DATA 60/20/20 PROPORTIONAL SPLIT")
        elif self.use_walk_forward:
            print("PREPROCESSING WITH ANCHORED WALK-FORWARD VALIDATION")
        else:
            print("PREPROCESSING WITH FIXED-RATIO SPLIT (LEGACY)")
        print("="*60)

        self.feature_columns = data.columns.tolist()
        self.close_idx = self.feature_columns.index('Close')

        # STEP 1: Split raw data FIRST (choice of strategy)
        if self.use_full_split:
            raw_splits = self.full_data_split(data)
        elif self.use_walk_forward:
            if self.split_months:
                raw_splits = self.walk_forward_split(
                    data,
                    train_months=self.split_months[0],
                    val_months=self.split_months[1],
                    test_months=self.split_months[2],
                )
            else:
                raw_splits = self.walk_forward_split(data)
        else:
            raw_splits = self.split_raw_data(data)

        # STEP 2: Normalize using continuous rolling statistics
        oos_data = raw_splits.get('oos')
        has_oos = oos_data is not None and len(oos_data) > 0
        normalized_splits = self.normalize_data_split_aware(
            raw_splits['train'],
            raw_splits['val'],
            raw_splits['test'],
            oos_data=oos_data if has_oos else None,
        )

        # STEP 3: Create sequences for each split
        print("\n Creating sequences for train split...")
        X_train, y_train = self.create_sequences(normalized_splits['train'].values)

        print("\n Creating sequences for validation split...")
        X_val, y_val = self.create_sequences(normalized_splits['val'].values)

        print("\n Creating sequences for test split...")
        X_test, y_test = self.create_sequences(normalized_splits['test'].values)

        idx_train = np.arange(len(X_train)) + raw_splits.get('train_start', 0)
        idx_val   = np.arange(len(X_val))   + raw_splits['val_start']
        idx_test  = np.arange(len(X_test))  + raw_splits['test_start']

        if has_oos:
            print("\n Creating sequences for OOS split...")
            X_oos, y_oos = self.create_sequences(normalized_splits['oos'].values)
            idx_oos = np.arange(len(X_oos)) + raw_splits.get('oos_start', raw_splits['test_start'] + len(X_test))
            print(f"    OOS:   X={X_oos.shape}, y={y_oos.shape}")
        else:
            X_oos = np.empty((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32)
            y_oos = np.empty((0, 1), dtype=np.float32)
            idx_oos = np.array([], dtype=np.int64)

        print("\n" + "="*60)
        print(" PREPROCESSING COMPLETE (NO DATA LEAKAGE)")
        print("="*60)
        print(f"  Final shapes:")
        print(f"    Train: X={X_train.shape}, y={y_train.shape}")
        print(f"    Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"    Test:  X={X_test.shape}, y={y_test.shape}")
        if has_oos:
            print(f"    OOS:   X={X_oos.shape}, y={y_oos.shape}")

        split_mode = "full_60_20_20" if self.use_full_split else (
            "walk_forward" if self.use_walk_forward else "fixed_ratio"
        )

        return {
            'X_train': X_train, 'y_train': y_train, 'idx_train': idx_train,
            'X_val':   X_val,   'y_val':   y_val,   'idx_val':   idx_val,
            'X_test':  X_test,  'y_test':  y_test,  'idx_test':  idx_test,
            'X_oos':   X_oos,   'y_oos':   y_oos,   'idx_oos':   idx_oos,
            'order_used': split_mode,
        }

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing configuration.

        Returns:
            Dictionary with preprocessing settings
        """
        split_mode = "full_60_20_20" if self.use_full_split else (
            "anchored_walk_forward" if self.use_walk_forward else "legacy_fixed_ratio"
        )
        return {
            'lookback_window':  self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'split_mode':       split_mode,
            'scaler_type':      'ContinuousRollingZScore_Clipped±4',
            'window':           200,
        }


# Example usage for testing
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer

    try:
        print("\n" + "="*60)
        print("STEP 1: LOAD DATA WITH DataLoader")
        print("="*60)
        loader = DataLoader("data/raw", "AAL")
        raw_data = loader.load_data()
        loader.validate_data()

        print("\n" + "="*60)
        print("STEP 2: COMPUTE TECHNICAL INDICATORS")
        print("="*60)
        fe = FeatureEngineer()
        enriched_data = fe.compute_indicators(raw_data)

        print("\n" + "="*60)
        print("STEP 3a: PREPROCESS DATA (ANCHORED WALK-FORWARD - default)")
        print("="*60)
        preprocessor_default = DataPreprocessor(
            lookback_window=24, forecast_horizon=8,
            use_walk_forward=True, use_full_split=False,
        )
        splits_default = preprocessor_default.preprocess(enriched_data)

        print("\n" + "="*60)
        print("STEP 3b: PREPROCESS DATA (FULL 60/20/20 -- --full mode)")
        print("="*60)
        preprocessor_full = DataPreprocessor(
            lookback_window=24, forecast_horizon=8,
            use_walk_forward=False, use_full_split=True,
        )
        splits_full = preprocessor_full.preprocess(enriched_data)

        print("\nSplit size comparison:")
        print(f"  Default  - Train: {splits_default['X_train'].shape[0]:>6}  "
              f"Val: {splits_default['X_val'].shape[0]:>6}  "
              f"Test: {splits_default['X_test'].shape[0]:>6}")
        print(f"  Full     - Train: {splits_full['X_train'].shape[0]:>6}  "
              f"Val: {splits_full['X_val'].shape[0]:>6}  "
              f"Test: {splits_full['X_test'].shape[0]:>6}")

        print("\n DataPreprocessor test completed successfully!")

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
