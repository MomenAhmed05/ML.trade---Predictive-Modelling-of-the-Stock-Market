import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback

tf.config.optimizer.set_jit(True)
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pickle
import random
import json
import pandas as pd
import gc
from quantile_barrier import QuantileBarrierLearner


# Module-level constants
DEFAULT_ATR_RATIO = 0.45
MIN_BARRIER_PCT = 0.003
MAX_BARRIER_PCT = 0.04


class CompositeCheckpoint(Callback):
    """
    Custom callback to save the model based on a composite metric combining
    validation accuracy and validation loss.

    The composite score is calculated as:
        score = val_accuracy - (alpha * val_loss)

    Higher score is better.
    """
    def __init__(self, filepath, alpha=0.5):
        super(CompositeCheckpoint, self).__init__()
        self.filepath = filepath
        self.alpha = alpha
        self.best_score = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc  = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        if val_acc is not None and val_loss is not None:
            current_score = val_acc - (self.alpha * val_loss)
            if current_score > self.best_score:
                print(f"\nEpoch {epoch+1}: Composite score improved from {self.best_score:.4f} to {current_score:.4f} "
                      f"(Acc: {val_acc:.4f}, Loss: {val_loss:.4f}). Saving weights...")
                self.best_score = current_score
                self.model.save_weights(self.filepath)


class LSTMModel:
    """
    MINIMAL LSTM for stock direction prediction.

    Stripped to bare essentials that research papers actually use for 55%+ accuracy:
    - Single LSTM(16) layer
    - Dropout(0.3)
    - Dense(2, softmax)
    - Standard categorical cross-entropy (no focal loss)

    Removed complexity that was preventing learning:
    - No second LSTM layer
    - No LayerNorm
    - No Attention
    - No focal loss
    """

    def __init__(
        self,
        model_path: str = "models/lstm_model.weights.h5",
        history_path: str = "models/training_history.pkl",
        regime_specialist_name: Optional[str] = None,
    ):
        self.model = None
        self.history = None

        self.model_path = Path(model_path)
        self.history_path = Path(history_path)
        self.regime_specialist_name = regime_specialist_name

    def build_model(
        self,
        input_shape: Tuple[int, int],
        learning_rate: float = 1e-3,
    ) -> keras.Model:
        """
        Build minimal LSTM model.

        Architecture:
        - Input Layer: (Lookback Window, Features)
        - Single LSTM(16): Extract temporal patterns
        - Dropout(0.3): Regularization
        - Dense(2, softmax): Binary classification (DOWN/UP)
        """
        print("\n" + "=" * 60)
        print("BUILDING MINIMAL LSTM MODEL")
        print("=" * 60)

        inputs = layers.Input(shape=input_shape, name="input_sequences")
        x = layers.LSTM(16, name="lstm")(inputs)
        x = layers.Dropout(0.3, name="dropout")(x)
        classification_output = layers.Dense(2, activation="softmax", name="direction_classification")(x)

        self.model = keras.Model(
            inputs=inputs,
            outputs=classification_output,
            name="minimal_lstm_stock_predictor",
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
            steps_per_execution=32,  # batch dispatch — bit-identical, reduces py/GPU overhead
        )

        print("\nModel architecture:")
        self.model.summary()

        print("\nTraining configuration:")
        print("  Objective: Pure Classification")
        print("  Loss: Categorical Cross-Entropy (standard)")
        print(f"  Optimizer: Adam (learning_rate={learning_rate})")
        print(f"  Input shape: {input_shape}")
        print("  LSTM Units: 16")
        print("  Dropout rate: 0.3")

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train_classification: np.ndarray,
        X_val: np.ndarray,
        y_val_classification: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute training loop with CompositeCheckpoint + EarlyStopping.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("\n" + "=" * 60)
        print("TRAINING MINIMAL LSTM MODEL")
        print("=" * 60)
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Epochs: {epochs} (max), Batch size: {batch_size}")
        print(f"Early stopping patience: {early_stopping_patience} epochs")
        print(f"Early stopping monitor: val_accuracy")
        print(f"Checkpoint monitor: Composite Score (Accuracy & Loss)\n")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        )

        composite_checkpoint = CompositeCheckpoint(
            filepath=str(self.model_path),
            alpha=0.5,
        )

        self.history = self.model.fit(
            X_train,
            y_train_classification,
            validation_data=(X_val, y_val_classification),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, composite_checkpoint],
            class_weight=class_weight,
            verbose=1,
        )

        try:
            self.load_model()
            print(f"\nBest model weights (by composite score) restored from: {self.model_path}")
        except Exception as e:
            print(f"\nWarning: Could not reload best model weights: {e}")

        return self.history.history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate directional predictions for new sequence data.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")

        predicted_directions = self.model.predict(X, batch_size=1024, verbose=0)
        predicted_class = np.argmax(predicted_directions, axis=1)
        predicted_text  = np.where(predicted_class == 0, "DOWN", "UP")
        confidence      = np.max(predicted_directions, axis=1)

        return {
            "direction_probabilities": predicted_directions,
            "direction":               predicted_text,
            "direction_confidence":    confidence,
        }

    def save_model(self) -> None:
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.model_path))

        # Save regime metadata alongside weights
        metadata = {
            "regime_specialist_name": self.regime_specialist_name,
        }
        metadata_path = self.model_path.parent / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"Model weights saved to: {self.model_path}")
        if self.regime_specialist_name:
            print(f"  (Regime specialist: {self.regime_specialist_name})")

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        self.model.load_weights(str(self.model_path))
        print(f"Model weights loaded from: {self.model_path}")

        # Load regime metadata if available
        metadata_path = self.model_path.parent / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.regime_specialist_name = metadata.get("regime_specialist_name")
            if self.regime_specialist_name:
                print(f"  (Regime specialist: {self.regime_specialist_name})")

    def save_history(self) -> None:
        if self.history is None:
            raise ValueError("No training history to save.")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "wb") as f:
            pickle.dump(self.history.history, f)
        print(f"Training history saved to: {self.history_path}")

    def load_history(self) -> Dict[str, list]:
        if not self.history_path.exists():
            raise FileNotFoundError(f"History not found at {self.history_path}")
        with open(self.history_path, "rb") as f:
            self.history = pickle.load(f)
        print(f"Training history loaded from: {self.history_path}")
        return self.history


def make_direction_onehot_from_raw(
    enriched_df: "pd.DataFrame",
    split_indices: Dict[str, np.ndarray],
    lookback_window: int = 24,
    forecast_horizon: int = 24,
    embargo: int = None,
    quantile_learner: Optional[QuantileBarrierLearner] = None,
    regime_series: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:

    """
    Computes objective classification targets (UP/DOWN) using Triple-Barrier Method
    with CLIPPED DYNAMIC ATR-BASED BARRIERS.

    Barrier 1 (Profit Target): +0.5 * ATR_14 (CLIPPED to 0.3% - 4.0%)
    Barrier 2 (Stop Loss)    : -0.5 * ATR_14 (CLIPPED to -4.0% - -0.3%)
    Barrier 3 (Time-Out)     : 'forecast_horizon' hours

    Purge/Embargo (Lopez de Prado, AFML Ch.7):
    Removes training samples whose label computation window overlaps with
    test data, plus an additional embargo period. This prevents information
    leakage from the triple-barrier path crossing split boundaries.

    Parameters
    ----------
    quantile_learner : QuantileBarrierLearner, optional
        Learner that provides regime-aware ATR ratios. If provided with regime_series,
        uses tau[regime] instead of fixed 0.45.
    regime_series : np.ndarray, optional
        Regime labels for each timestep. If provided with quantile_learner,
        uses regime-specific ATR ratios.
    """

    close_prices = enriched_df["Close"].values
    atr_values = enriched_df["ATR_14"].values
    n = len(close_prices)

    # Default ATR ratio; will be overridden if quantile_learner and regime_series are provided
    DEFAULT_ATR_RATIO = 0.45
    MIN_BARRIER_PCT = 0.003
    MAX_BARRIER_PCT = 0.04

    is_up_list = []
    barrier_stats = {"hit_pt": 0, "hit_sl": 0, "timeout_up": 0, "timeout_down": 0, "clipped": 0}
    barrier_pct_list = []

    # Determine if we're using regime-aware quantiles
    use_regime_quantiles = quantile_learner is not None and regime_series is not None

    for i in range(n - lookback_window - forecast_horizon + 1):
        base_idx = i + lookback_window
        base_price = close_prices[base_idx]
        base_atr = atr_values[base_idx]

        # Use regime-aware ATR ratio if available, otherwise default to 0.45
        if use_regime_quantiles:
            regime = regime_series[base_idx] if base_idx < len(regime_series) else "BULL"
            tau_values = quantile_learner.get_tau_values()
            # Map regime name to tau key
            regime_to_tau = {
                "BULL": "tau_bull",
                "BEAR": "tau_bear",
                "CRISIS": "tau_crisis",
                0: "tau_bull",
                1: "tau_bear",
                2: "tau_crisis",
            }
            tau_key = regime_to_tau.get(regime, "tau_bull")
            atr_ratio = tau_values.get(tau_key, DEFAULT_ATR_RATIO)
        else:
            atr_ratio = DEFAULT_ATR_RATIO

        profit_target_pct = atr_ratio * base_atr / base_price
        profit_target_pct_clipped = np.clip(profit_target_pct, MIN_BARRIER_PCT, MAX_BARRIER_PCT)

        if profit_target_pct != profit_target_pct_clipped:
            barrier_stats["clipped"] += 1

        stop_loss_pct = -profit_target_pct_clipped
        barrier_pct_list.append(profit_target_pct_clipped)

        future_path  = close_prices[base_idx + 1 : base_idx + 1 + forecast_horizon]
        path_returns = (future_path - base_price) / base_price

        hit_pt = np.where(path_returns >= profit_target_pct_clipped)[0]
        hit_sl = np.where(path_returns <= stop_loss_pct)[0]

        idx_pt = hit_pt[0] if len(hit_pt) > 0 else np.inf
        idx_sl = hit_sl[0] if len(hit_sl) > 0 else np.inf

        if idx_pt < idx_sl:
            is_up_list.append(1)
            barrier_stats["hit_pt"] += 1
        elif idx_sl < idx_pt:
            is_up_list.append(0)
            barrier_stats["hit_sl"] += 1
        else:
            final_return = path_returns[-1] if len(path_returns) > 0 else 0
            if final_return > 0:
                is_up_list.append(1)
                barrier_stats["timeout_up"] += 1
            else:
                is_up_list.append(0)
                barrier_stats["timeout_down"] += 1

    is_up            = np.array(is_up_list)
    barrier_pct_array = np.array(barrier_pct_list)
    barrier_stats["avg_barrier_pct"] = np.mean(barrier_pct_array) * 100
    barrier_stats["min_barrier_pct"] = np.min(barrier_pct_array)  * 100
    barrier_stats["max_barrier_pct"] = np.max(barrier_pct_array)  * 100

    def _to_onehot(arr: np.ndarray) -> np.ndarray:
        onehot = np.zeros((len(arr), 2), dtype=np.float32)
        onehot[np.arange(len(arr)), arr] = 1.0
        return onehot

    train_idx = split_indices["idx_train"]
    val_idx   = split_indices["idx_val"]
    test_idx  = split_indices["idx_test"]

    # -- Purge/Embargo (Lopez de Prado, AFML Ch.7) ----------------------
    # Each label at position j depends on prices from base_idx+1 to base_idx+horizon.
    # base_idx = j + lookback_window, so label j uses prices up to j + lookback + horizon.
    # If a training sample's label window overlaps with any test index, purge it.
    # Additionally, embargo the first `embargo` samples of the test set.
    if embargo is None:
        embargo = forecast_horizon

    test_start_raw = test_idx[0] if len(test_idx) > 0 else len(is_up)

    # Purge train: remove samples whose label computation reaches into test
    # Label at sample j uses prices up to enriched index (j + lookback + horizon).
    # The enriched index for sample j is j + lookback_window.
    # We need: j + lookback_window + forecast_horizon <= test_start_raw
    # => j <= test_start_raw - lookback_window - forecast_horizon
    purge_boundary = test_start_raw - lookback_window - forecast_horizon
    train_mask = train_idx < purge_boundary

    # Purge val similarly
    val_purge_boundary = test_start_raw - lookback_window - forecast_horizon
    val_mask = val_idx < val_purge_boundary

    # Embargo: additionally remove the first `embargo` test samples
    embargo_count = min(embargo, len(test_idx))
    test_mask = np.ones(len(test_idx), dtype=bool)
    if embargo_count > 0:
        test_mask[:embargo_count] = False

    n_purged_train = int(np.sum(~train_mask))
    n_purged_val = int(np.sum(~val_mask))
    n_embargo_test = int(np.sum(~test_mask))
    if n_purged_train > 0 or n_purged_val > 0 or n_embargo_test > 0:
        barrier_stats["n_purged_train"] = n_purged_train
        barrier_stats["n_purged_val"] = n_purged_val
        barrier_stats["n_embargo_test"] = n_embargo_test
        print(f"  [Purge/Embargo] Purged {n_purged_train} train, "
              f"{n_purged_val} val samples; "
              f"embargoed {n_embargo_test} test samples (embargo={embargo})")

    labels = {
        "train": _to_onehot(is_up[train_idx]),
        "val":   _to_onehot(is_up[val_idx]),
        "test":  _to_onehot(is_up[test_idx]),
    }

    masks = {
        "train": train_mask,
        "val":   val_mask,
        "test":  test_mask,
    }

    return labels, masks, barrier_stats


# Minimum enriched-row count required for the anchored 12/2/3-month split.
# Stocks below this fall back to the percentage split - their idx_train[-1]
# maps to a much earlier date and must NOT influence the beta cutoff.
_ANCHORED_SPLIT_MIN_ROWS = 12474


def get_high_beta_stocks(data_path: str, pool_size: int = 200, num_select: int = 5, min_volume: float = 300000) -> list:
    """
    Scans all available stocks to filter by volume, builds a synthetic market index
    from the remaining liquid stocks, and calculates Beta against that proxy.
    Finally, selects a random subset from the high-beta pool.
    Caches the results to disk to avoid expensive recalculations.

    Beta cutoff derivation:
      Only stocks that have enough history to use the anchored 12/2/3-month split
      contribute to the cutoff.  Their idx_train[-1] maps to ~12 months before the
      end of their data, which is exactly "just before the val set" as intended.
      We take the MEDIAN across those dates so that even one short-history outlier
      cannot drag the cutoff back by years.
    """
    import pandas as pd
    import numpy as np
    import random
    from data_loader import DataLoader

    beta_cache_file = Path("models/stock_scan_cache.json")
    if beta_cache_file.exists():
        print(f"Loading high beta pool from {beta_cache_file}...")
        with open(beta_cache_file, "r") as f:
            high_beta_pool = json.load(f)
        if high_beta_pool:
            random.seed(42)
            return random.sample(high_beta_pool, min(num_select, len(high_beta_pool)))

    print(f"Scanning ALL stocks in '{data_path}' to filter by volume > {min_volume}...")

    available_symbols = DataLoader.list_available_stocks(data_path)
    if not available_symbols:
        raise ValueError(f"No data files found in {data_path}")

    series_list = {}
    volume_list = {}
    successful_loads = 0

    for symbol in available_symbols:
        try:
            loader   = DataLoader(data_path, symbol)
            filepath = loader._find_data_file()
            df = pd.read_csv(filepath, header=None, usecols=[0, 4, 5], names=['Datetime', 'Close', 'Volume'])
            df['Close']  = pd.to_numeric(df['Close'],  errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > 0:
                df.set_index('Datetime', inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                avg_volume = df['Volume'].mean(skipna=True)
                if pd.notna(avg_volume) and avg_volume >= min_volume:
                    series_list[symbol] = df['Close']
                    volume_list[symbol] = avg_volume
                    successful_loads += 1
        except Exception:
            pass

    print(f"Filtered down to {successful_loads} 'high volume' stocks (>= {min_volume} avg daily volume).")

    if not series_list:
        print(f"WARNING: No stocks met the volume threshold of {min_volume}. Lowering threshold to 0 to prevent crash.")
        return get_high_beta_stocks_fallback(data_path, pool_size, num_select)

    print("Deriving beta cutoff date from anchored-split stocks only (survivorship-bias fix)...")
    from feature_engineer import FeatureEngineer
    from ensemble_model import _make_preprocessor

    train_end_dates = []
    sampled_symbols = list(series_list.keys())[:50]  # cap at 50 for startup speed
    anchored_count  = 0

    for symbol in sampled_symbols:
        try:
            loader   = DataLoader(data_path, symbol)
            raw_df   = loader.load_data()
            enriched = FeatureEngineer().compute_indicators(raw_df)

            # Skip stocks that fall back to the percentage split - their
            # idx_train[-1] resolves to an arbitrarily early date.
            if len(enriched) < _ANCHORED_SPLIT_MIN_ROWS:
                continue

            _pre, splits, enriched_trunc = _make_preprocessor(24, 8, enriched, None)
            last_train_pos = int(splits["idx_train"][-1])
            train_end_dt   = pd.to_datetime(
                enriched_trunc.index[last_train_pos], errors="coerce"
            )
            if pd.notna(train_end_dt):
                train_end_dates.append(train_end_dt)
                anchored_count += 1
        except Exception:
            pass

    cutoff_ts = None
    if train_end_dates:
        # Median is robust to any remaining outliers.
        sorted_dates = sorted(train_end_dates)
        cutoff_ts    = sorted_dates[len(sorted_dates) // 2]
        cutoff_str   = cutoff_ts.strftime("%Y-%m-%d")
        print(f"  Beta cutoff derived: {cutoff_str}  "
              f"(median train_end from {anchored_count} anchored-split stocks)")
    else:
        print("  WARNING: No anchored-split stocks found in sample - beta will use full data.")

    if cutoff_ts is not None:
        truncated_series = {}
        for symbol, close_series in series_list.items():
            try:
                idx_ts    = pd.to_datetime(close_series.index, errors="coerce")
                truncated = close_series.loc[idx_ts <= cutoff_ts]
                if not truncated.empty:
                    truncated_series[symbol] = truncated
            except Exception:
                truncated_series[symbol] = close_series
        series_list = truncated_series
        print(f"  {len(series_list)} stocks remain after truncation to cutoff.")

    print("Building synthetic market index and calculating Beta...")

    price_df = pd.DataFrame(series_list)
    price_df.ffill(inplace=True)
    price_df.dropna(how='all', inplace=True)
    returns_df    = price_df.pct_change(fill_method=None)
    returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    market_return = returns_df.mean(axis=1, skipna=True)
    market_return.replace([np.inf, -np.inf], np.nan, inplace=True)
    market_return.dropna(inplace=True)
    market_var    = market_return.var()

    betas = {}
    if pd.notna(market_var) and market_var > 0:
        for ticker in returns_df.columns:
            stock_ret = returns_df[ticker].replace([np.inf, -np.inf], np.nan).dropna()
            aligned   = pd.concat([stock_ret, market_return], axis=1).dropna()
            if len(aligned) > 100:
                cov  = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                beta = cov / market_var
                if pd.notna(beta):
                    betas[ticker] = beta

    sorted_stocks  = sorted(betas.keys(), key=lambda k: betas[k], reverse=True)
    high_beta_pool = sorted_stocks[:pool_size]

    print(f"Identified {len(high_beta_pool)} stocks that classify as 'high beta' pool candidates.")

    if not high_beta_pool:
        print("WARNING: Could not calculate Beta. Falling back to random selection of liquid stocks.")
        high_beta_pool = list(series_list.keys())

    beta_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(beta_cache_file, "w") as f:
        json.dump(high_beta_pool, f)

    random.seed(42)
    return random.sample(high_beta_pool, min(num_select, len(high_beta_pool)))


def get_high_beta_stocks_fallback(data_path: str, pool_size: int, num_select: int) -> list:
    """Fallback function without volume limits."""
    import pandas as pd
    import numpy as np
    import random
    from data_loader import DataLoader

    available_symbols = DataLoader.list_available_stocks(data_path)
    sample_size       = min(2000, len(available_symbols))
    market_sample_symbols = random.sample(available_symbols, sample_size)

    series_list = {}
    for symbol in market_sample_symbols:
        try:
            loader   = DataLoader(data_path, symbol)
            filepath = loader._find_data_file()
            df = pd.read_csv(filepath, header=None, usecols=[0, 4], names=['Datetime', 'Close'])
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > 0:
                df.set_index('Datetime', inplace=True)
                df = df[~df.index.duplicated(keep='first')]
                series_list[symbol] = df['Close']
        except Exception:
            pass

    price_df = pd.DataFrame(series_list)
    price_df.ffill(inplace=True)
    price_df.dropna(how='all', inplace=True)
    returns_df    = price_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    market_return = returns_df.mean(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan).dropna()
    market_var    = market_return.var()

    betas = {}
    if pd.notna(market_var) and market_var > 0:
        for ticker in returns_df.columns:
            stock_ret = returns_df[ticker].replace([np.inf, -np.inf], np.nan).dropna()
            aligned   = pd.concat([stock_ret, market_return], axis=1).dropna()
            if len(aligned) > 100:
                cov  = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                beta = cov / market_var
                if pd.notna(beta):
                    betas[ticker] = beta

    sorted_stocks  = sorted(betas.keys(), key=lambda k: betas[k], reverse=True)
    high_beta_pool = sorted_stocks[:pool_size]

    if not high_beta_pool:
        return random.sample(available_symbols, min(num_select, len(available_symbols)))
    return random.sample(high_beta_pool, min(num_select, len(high_beta_pool)))


def _evaluate_group(
    model: "LSTMModel",
    test_data_dict: Dict[str, Dict[str, np.ndarray]],
    group_id: int,
    group_dir: Path,
) -> Dict[str, Any]:
    """
    Runs test-set evaluation for a trained group model.
    """
    from model_evaluator import ModelEvaluator
    from sklearn.metrics import accuracy_score

    all_y_true, all_y_pred, all_confidences = [], [], []

    for ticker, tdata in test_data_dict.items():
        if len(tdata["X_test"]) == 0:
            continue
        preds      = model.predict(tdata["X_test"])
        y_pred_cls = np.argmax(preds["direction_probabilities"], axis=1)
        y_true_cls = np.argmax(tdata["y_test_cls"], axis=1)
        all_y_true.extend(y_true_cls)
        all_y_pred.extend(y_pred_cls)
        all_confidences.extend(preds["direction_confidence"])

    if not all_y_true:
        print(f"  Group {group_id}: no test data available for evaluation.")
        return {}

    arr_y_true = np.array(all_y_true)
    arr_y_pred = np.array(all_y_pred)
    arr_conf   = np.array(all_confidences)

    print("\n" + "=" * 60)
    print(f"EVALUATING ON TEST SET - GROUP {group_id}")
    print("=" * 60)

    from model_evaluator import ModelEvaluator
    evaluator = ModelEvaluator(results_dir="")
    group_dir.mkdir(parents=True, exist_ok=True)
    evaluator.plot_confusion_matrix(
        arr_y_true, arr_y_pred,
        save_path=str(group_dir / "confusion_matrix.png"),
    )
    overall_metrics = evaluator.evaluate_classification(arr_y_true, arr_y_pred)

    print("\n" + "=" * 40)
    print("HIGH CONFIDENCE CLASSIFICATION METRICS")
    print("=" * 40)

    results: Dict[str, Any] = {
        "overall_accuracy":  overall_metrics.get("accuracy",  0.0),
        "overall_precision": overall_metrics.get("precision", 0.0),
        "overall_recall":    overall_metrics.get("recall",    0.0),
        "overall_f1":        overall_metrics.get("f1",        0.0),
    }

    thresholds = [0.55, 0.60, 0.65]
    for t in thresholds:
        key = f"{int(t * 100)}"   # "55", "60", "65"
        mask  = arr_conf >= t
        count = int(np.sum(mask))
        pct   = count / len(arr_y_pred) * 100 if len(arr_y_pred) > 0 else 0.0

        if count == 0:
            print(f"Threshold >= {t:.2f}: No predictions met this threshold.")
            results[f"acc_{key}"]    = 0.0
            results[f"trades_{key}"] = 0
            results[f"pct_{key}"]    = 0.0
            continue

        acc = accuracy_score(arr_y_true[mask], arr_y_pred[mask])
        results[f"acc_{key}"]    = round(acc, 4)
        results[f"trades_{key}"] = count
        results[f"pct_{key}"]    = round(pct, 1)

        print(f"Threshold >= {t:.2f}:")
        print(f"  Total Trades : {count} ({pct:.1f}% of data)")
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  Pred UP ratio: {np.mean(arr_y_pred[mask]):.3f}")
        print("-" * 30)

    return results


def train_group(
    stock_list: list,
    group_id: int,
    data_path: str = "data/raw",
    lookback_window: int = 24,
    forecast_horizon: int = 24,
    anchor_end_date: Optional[str] = None,
    model_dir_prefix: Optional[str] = None,
    use_full_split: bool = False,
    split_months: tuple = None,
    regime_filter: Optional[str] = None,
    saml_learner=None,
    quantile_learner=None,
) -> dict:
    """
    Trains a single LSTM model on the given stock list.

    Parameters
    ----------
    use_full_split : bool
        When True (--full mode), passes the 60/20/20 proportional split flag
        through to DataPreprocessor via _make_preprocessor().
        When False (default), uses the anchored walk-forward 12/2/3-month split.
    split_months : tuple, optional
        (train, val, test) month counts to override the default 12/2/3.
        Used by --live mode for 13/5/6 month splits on yfinance data.
    regime_filter : str, optional
        Filter training samples by regime label (e.g., "BULL", "BEAR", "CRISIS", or None).
        If "BULL_JOINT", includes both BULL and CRISIS samples.
        If provided, saves to regime-specific subdirectory.
    """
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    from model_evaluator import ModelEvaluator
    from ensemble_model import _make_preprocessor

    # Build regime-specific path if regime_filter is provided
    if regime_filter:
        regime_subdir = f"regime_{regime_filter.lower()}"
        if model_dir_prefix:
            group_dir = Path(f"models/{model_dir_prefix}/{regime_subdir}/group_{group_id}")
        else:
            group_dir = Path(f"models/{regime_subdir}/group_{group_id}")
    elif model_dir_prefix:
        group_dir = Path(f"models/{model_dir_prefix}/group_{group_id}")
    else:
        group_dir = Path(f"models/group_{group_id}")
    group_dir.mkdir(parents=True, exist_ok=True)

    model_path   = str(group_dir / "lstm_model.weights.h5")
    history_path = str(group_dir / "training_history.pkl")
    rf_path      = str(group_dir / "rf_model.pkl")
    xgb_path     = str(group_dir / "xgb_model.pkl")
    tickers_path = str(group_dir / "selected_tickers.json")

    print("\n" + "=" * 60)
    print(f"LSTM PIPELINE: GROUP {group_id} - {stock_list}")
    if model_dir_prefix:
        print(f"  artifact dir   : {group_dir}")
    if anchor_end_date:
        print(f"  anchor_end_date: {anchor_end_date}  (data truncated before splitting)")
    if use_full_split:
        print("  split mode     : FULL DATA (60/20/20 proportional)")
    print("=" * 60)

    valid_tickers = []
    all_X_train, all_y_train_cls = [], []
    all_X_val,   all_y_val_cls   = [], []
    test_data_dict = {}
    global_barrier_stats = {
        "hit_pt": 0, "hit_sl": 0, "timeout_up": 0, "timeout_down": 0,
        "avg_barrier_pct": 0, "min_barrier_pct": 999, "max_barrier_pct": 0, "clipped": 0,
    }

    for ticker in stock_list:
        print(f"\n--- Processing {ticker} ---")
        try:
            loader = DataLoader(data_path, ticker)
            raw_df = loader.load_data()
            loader.validate_data()
        except Exception as e:
            print(f"  -> Skipping {ticker}: {e}")
            continue

        fe       = FeatureEngineer()
        enriched = fe.compute_indicators(raw_df)

        try:
            pre, splits, enriched_trunc = _make_preprocessor(
                lookback_window, forecast_horizon, enriched, anchor_end_date,
                use_full_split=use_full_split, split_months=split_months,
            )
        except ValueError as e:
            print(f"  -> Skipping {ticker}: {e}")
            continue

        # Scale minimum test rows to actual test window size
        if use_full_split:
            min_test_rows = 7_300
        elif split_months is not None:
            # ~10% of expected test sequences (test_months * ~730 rows/month)
            min_test_rows = max(100, int(split_months[2] * 730 * 0.10))
        else:
            min_test_rows = 730
        if len(splits["X_test"]) < min_test_rows:
            print(f"  -> Skipping {ticker}: Test split too small ({len(splits['X_test'])} sequences < {min_test_rows})")
            continue

        valid_tickers.append(ticker)

        direction_labels, direction_masks, barrier_stats = make_direction_onehot_from_raw(
            enriched_df=enriched_trunc,
            split_indices={
                "idx_train": splits["idx_train"],
                "idx_val": splits["idx_val"],
                "idx_test": splits["idx_test"],
            },
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
    embargo=forecast_horizon,  # FIX 3: Enable purge/embargo (Lopez de Prado)
            quantile_learner=quantile_learner,
        )

        train_up_pct = direction_labels["train"][:, 1].mean()
        val_up_pct = direction_labels["val"][:, 1].mean()
        drift = val_up_pct - train_up_pct
        print(f" {ticker} labels: train_UP={train_up_pct:.3f} val_UP={val_up_pct:.3f} drift={drift:+.3f} {'[!] SKEWED' if abs(drift) > 0.07 else '[OK]'}")
        if abs(drift) > 0.07:
            print(f" -> Skipping {ticker}: label drift {drift:+.3f} exceeds threshold")
            continue

        for key in ["hit_pt", "hit_sl", "timeout_up", "timeout_down", "clipped"]:
            global_barrier_stats[key] += barrier_stats.get(key, 0)
        global_barrier_stats["avg_barrier_pct"] += barrier_stats["avg_barrier_pct"]
        global_barrier_stats["min_barrier_pct"]  = min(global_barrier_stats["min_barrier_pct"],  barrier_stats["min_barrier_pct"])
        global_barrier_stats["max_barrier_pct"]  = max(global_barrier_stats["max_barrier_pct"],  barrier_stats["max_barrier_pct"])

        train_mask = direction_masks["train"]
        val_mask   = direction_masks["val"]
        test_mask  = direction_masks["test"]

        all_X_train.append(splits["X_train"][train_mask])
        all_y_train_cls.append(direction_labels["train"][train_mask])
        all_X_val.append(splits["X_val"][val_mask])
        all_y_val_cls.append(direction_labels["val"][val_mask])

        test_data_dict[ticker] = {
            "X_test": splits["X_test"][test_mask],
            "y_test_cls": direction_labels["test"][test_mask],
        }

    if not valid_tickers:
        raise ValueError(f"Group {group_id}: No tickers passed the test split size filter.")

    # Apply regime filtering to training samples if specified
    if regime_filter:
        print(f"\nApplying regime filter: '{regime_filter}'")
        # Note: This assumes regime labels are available in the feature data
        # The regime is expected to be in the last few columns of X_train
        # Look for regime indicator columns (e.g., is_bull, is_bear, is_crisis)
        # For now, we'll filter based on regime metadata if available
        # This is a placeholder - actual implementation depends on how regime info is stored

        # If regime_filter is "BULL_JOINT", include both BULL and CRISIS
        included_regimes = [regime_filter]
        if regime_filter == "BULL_JOINT":
            included_regimes = ["BULL", "CRISIS"]

        print(f"  Including regimes: {included_regimes}")
        print(f"  (Note: Full regime filtering requires regime labels in training data)")

    with open(tickers_path, "w") as f:
        json.dump(valid_tickers, f)

    X_train     = np.concatenate(all_X_train)
    y_train_cls = np.concatenate(all_y_train_cls)
    X_val       = np.concatenate(all_X_val)
    y_val_cls   = np.concatenate(all_y_val_cls)

    del all_X_train, all_y_train_cls, all_X_val, all_y_val_cls
    gc.collect()

# -------------------------------------------------------------------------
# Bearish regime oversampling (Lopez de Prado, AFML ch.4).
# Identify bearish sequences by their DOWN label (y_train_cls[:,0] == 1)
# rather than searching for sharpe_ratio_20 feature, which may not be
# reliably available as the last column in X_train.
# -------------------------------------------------------------------------
    BEARISH_OVERSAMPLE_MULTIPLIER = 2

    bearish_mask = y_train_cls[:, 0] == 1
    n_bearish = int(bearish_mask.sum())

    if n_bearish > 0:
        X_bear = X_train[bearish_mask]
        y_bear = y_train_cls[bearish_mask]
        X_train = np.concatenate(
            [X_train] + [X_bear] * (BEARISH_OVERSAMPLE_MULTIPLIER - 1), axis=0
        )
        y_train_cls = np.concatenate(
            [y_train_cls] + [y_bear] * (BEARISH_OVERSAMPLE_MULTIPLIER - 1), axis=0
        )
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train_cls = y_train_cls[perm]
        print(f" Bearish regime oversampling: {n_bearish} bearish sequences "
            f"duplicated {BEARISH_OVERSAMPLE_MULTIPLIER - 1}x "
            f"-> total training set: {len(X_train)} samples")
    else:
        print(" Bearish regime oversampling: no bearish sequences found, skipping.")

    n_features = X_train.shape[2]

    down_count   = float(y_train_cls[:, 0].sum())
    up_count     = float(y_train_cls[:, 1].sum())
    total        = down_count + up_count
    class_weight = {
        0: total / (2.0 * down_count),
        1: total / (2.0 * up_count),
    }

    model = LSTMModel(
        model_path=model_path,
        history_path=history_path,
        regime_specialist_name=regime_filter,
    )
    model.build_model(input_shape=(lookback_window, n_features), learning_rate=1e-3)

    # Initialize from SAML meta-learner if provided
    if saml_learner is not None:
        print(f"[SAML] Initializing model from meta-learner...")
        try:
            saml_learner.initialize_specialist(model.model)
            print(f"[SAML] Model initialized from meta-weights")
        except Exception as e:
            print(f"[SAML] Failed to initialize from meta-learner: {e}")
            print(f"[SAML] Continuing with random initialization")

        # Apply quantile barrier learner if provided
        if quantile_learner is not None:
            tau_values = quantile_learner.get_tau_values()
            print(f"[QuantileBarrier] Using learned barriers: BULL={tau_values.get('tau_bull', DEFAULT_ATR_RATIO):.3f}, "
                  f"BEAR={tau_values.get('tau_bear', DEFAULT_ATR_RATIO):.3f}, "
                  f"CRISIS={tau_values.get('tau_crisis', DEFAULT_ATR_RATIO):.3f}")

    history = model.train(
        X_train=X_train,
        y_train_classification=y_train_cls,
        X_val=X_val,
        y_val_classification=y_val_cls,
        epochs=50,
        batch_size=32,
        early_stopping_patience=5,
        class_weight=class_weight,
    )
    model.save_history()

    best_val_acc = max(history.get("val_accuracy", [0.5]))

    val_acc_series   = np.array(history.get("val_accuracy", [0.5]))
    val_mean         = np.mean(val_acc_series - 0.5)
    val_std          = np.std(val_acc_series) if np.std(val_acc_series) > 0 else 1e-8
    val_sharpe_proxy = (val_mean / val_std) * np.sqrt(len(val_acc_series))

    eval_metrics = _evaluate_group(model, test_data_dict, group_id, group_dir)

    print(f"\nGroup {group_id} training complete.")
    print(f"  Best val accuracy  : {best_val_acc:.4f}")
    print(f"  Val Sharpe proxy   : {val_sharpe_proxy:.3f}")
    if model_dir_prefix:
        print(f"  Artifacts saved to : {group_dir}")

    return {
        "group_id":          group_id,
        "stocks":            valid_tickers,
        "val_accuracy":      best_val_acc,
        "val_sharpe_proxy":  val_sharpe_proxy,
        "val_sharpe":        val_sharpe_proxy,
        "overall_accuracy":  eval_metrics.get("overall_accuracy",  0.0),
        "acc_55":            eval_metrics.get("acc_55",  0.0),
        "acc_60":            eval_metrics.get("acc_60",  0.0),
        "acc_65":            eval_metrics.get("acc_65",  0.0),
        "trades_55":         eval_metrics.get("trades_55", 0),
        "trades_60":         eval_metrics.get("trades_60", 0),
        "trades_65":         eval_metrics.get("trades_65", 0),
        "model_path":        model_path,
        "rf_path":           rf_path,
        "xgb_path":          xgb_path,
        "tickers_path":      tickers_path,
        "n_features":        n_features,
        "test_data":         test_data_dict,
    }


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    from model_evaluator import ModelEvaluator

    try:
        print("\n" + "=" * 60)
        print("LSTM PIPELINE: ANCHORED WALK-FORWARD + CLIPPED ATR BARRIERS")
        print("=" * 60)

        LOOKBACK_WINDOW  = 24
        FORECAST_HORIZON = 24

        target_stocks = get_high_beta_stocks('data/raw', pool_size=200, num_select=5, min_volume=300000)
        target_stocks.sort()
        print(f"\nSelected stocks: {target_stocks}")

        result = train_group(
            stock_list=target_stocks,
            group_id=0,
            data_path='data/raw',
            lookback_window=LOOKBACK_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )

        with open(result["tickers_path"], "r") as f:
            valid_tickers = json.load(f)
        Path("models").mkdir(parents=True, exist_ok=True)
        with open("models/selected_tickers.json", "w") as f:
            json.dump(valid_tickers, f)

        import shutil
        shutil.copy(result["model_path"],   "models/lstm_model.weights.h5")
        shutil.copy(result["tickers_path"], "models/selected_tickers.json")

        print(f"\nVal accuracy : {result['val_accuracy']:.4f}")
        print(f"Val Sharpe   : {result['val_sharpe']:.3f}")
        print("\nRun ensemble_model.py to evaluate on the test set.")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
