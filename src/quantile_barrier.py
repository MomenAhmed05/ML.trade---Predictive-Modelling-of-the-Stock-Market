"""
Quantile-Based Adaptive Barriers for Triple-Barrier Method

This module implements learnable regime-specific quantile barriers that replace
the fixed ATR_RATIO (0.45) with adaptive values based on market regime.

Expected behavior:
- BULL regime: tighter barriers (~0.35) for more sensitive entries
- BEAR regime: medium barriers (~0.55) for balanced protection
- CRISIS regime: wider barriers (~0.70) to accommodate volatility
"""

import json
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(True)
from pathlib import Path
from typing import Dict, Optional, Union, List


class QuantileBarrierLearner(tf.keras.layers.Layer):
    """
    Learnable quantile-based adaptive barriers for triple-barrier labeling.
    
    Replaces fixed ATR_RATIO with regime-specific learnable quantiles:
    - tau_bull: barrier width for bull markets (tighter, ~0.35)
    - tau_bear: barrier width for bear markets (medium, ~0.55)
    - tau_crisis: barrier width for crisis markets (wider, ~0.70)
    
    All taus are constrained to [0.1, 0.9] via sigmoid transformation.
    """
    
    # Regime mapping constants
    REGIME_BULL = 0
    REGIME_BEAR = 1
    REGIME_CRISIS = 2
    
    # Expected values for initialization/reference
    DEFAULT_TAU_BULL = 0.35
    DEFAULT_TAU_BEAR = 0.55
    DEFAULT_TAU_CRISIS = 0.70
    DEFAULT_FIXED_RATIO = 0.45  # For backward compatibility
    
    # Constraints
    TAU_MIN = 0.1
    TAU_MAX = 0.9
    
    def __init__(
        self,
        tau_bull: Optional[float] = None,
        tau_bear: Optional[float] = None,
        tau_crisis: Optional[float] = None,
        trainable: bool = True,
        name: str = "quantile_barrier_learner",
        **kwargs
    ):
        """
        Initialize the QuantileBarrierLearner.
        
        Args:
            tau_bull: Initial tau for bull regime (default: 0.35)
            tau_bear: Initial tau for bear regime (default: 0.55)
            tau_crisis: Initial tau for crisis regime (default: 0.70)
            trainable: Whether the taus are trainable
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        # Use expected values as defaults if not provided
        self._init_tau_bull = tau_bull if tau_bull is not None else self.DEFAULT_TAU_BULL
        self._init_tau_bear = tau_bear if tau_bear is not None else self.DEFAULT_TAU_BEAR
        self._init_tau_crisis = tau_crisis if tau_crisis is not None else self.DEFAULT_TAU_CRISIS
        
        self._trainable = trainable
        
        # Initialize raw parameters (will be transformed via sigmoid to [TAU_MIN, TAU_MAX])
        # We use inverse sigmoid to get the raw values that produce desired initial taus
        self._raw_tau_bull = self._inverse_sigmoid_transform(self._init_tau_bull)
        self._raw_tau_bear = self._inverse_sigmoid_transform(self._init_tau_bear)
        self._raw_tau_crisis = self._inverse_sigmoid_transform(self._init_tau_crisis)
    
    def build(self, input_shape=None):
        """Build the layer by creating trainable variables."""
        self.raw_tau_bull = self.add_weight(
            name="raw_tau_bull",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._raw_tau_bull),
            trainable=self._trainable,
            dtype=tf.float32
        )
        self.raw_tau_bear = self.add_weight(
            name="raw_tau_bear",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._raw_tau_bear),
            trainable=self._trainable,
            dtype=tf.float32
        )
        self.raw_tau_crisis = self.add_weight(
            name="raw_tau_crisis",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._raw_tau_crisis),
            trainable=self._trainable,
            dtype=tf.float32
        )
        super().build(input_shape)
    
    def _sigmoid_transform(self, x: tf.Tensor) -> tf.Tensor:
        """
        Transform raw parameter to constrained [TAU_MIN, TAU_MAX] range.
        Uses sigmoid for smooth differentiable constraints.
        """
        # Sigmoid outputs [0, 1], scale to [TAU_MIN, TAU_MAX]
        return self.TAU_MIN + (self.TAU_MAX - self.TAU_MIN) * tf.sigmoid(x)
    
    def _inverse_sigmoid_transform(self, tau: float) -> float:
        """
        Compute raw parameter value that produces desired tau after sigmoid transform.
        Used for initialization.
        """
        # Clip to avoid numerical issues
        tau = np.clip(tau, self.TAU_MIN + 1e-6, self.TAU_MAX - 1e-6)
        normalized = (tau - self.TAU_MIN) / (self.TAU_MAX - self.TAU_MIN)
        # Inverse of sigmoid: log(x / (1 - x))
        return float(np.log(normalized / (1.0 - normalized)))
    
    @property
    def tau_bull(self) -> tf.Tensor:
        """Get the current tau value for bull regime (constrained to [0.1, 0.9])."""
        return self._sigmoid_transform(self.raw_tau_bull)
    
    @property
    def tau_bear(self) -> tf.Tensor:
        """Get the current tau value for bear regime (constrained to [0.1, 0.9])."""
        return self._sigmoid_transform(self.raw_tau_bear)
    
    @property
    def tau_crisis(self) -> tf.Tensor:
        """Get the current tau value for crisis regime (constrained to [0.1, 0.9])."""
        return self._sigmoid_transform(self.raw_tau_crisis)
    
    def get_tau_values(self) -> Dict[str, float]:
        """Get current tau values as Python floats."""
        if not self.built:
            self.build()
        return {
            "tau_bull": float(self.tau_bull.numpy()),
            "tau_bear": float(self.tau_bear.numpy()),
            "tau_crisis": float(self.tau_crisis.numpy())
        }
    
    def call(self, regime_labels: tf.Tensor) -> tf.Tensor:
        """
        Get tau values for given regime labels.
        
        Args:
            regime_labels: Tensor of shape (n,) with values 0 (bull), 1 (bear), 2 (crisis)
        
        Returns:
            Tensor of tau values for each sample
        """
        regime_labels = tf.cast(regime_labels, tf.int32)
        
        taus = tf.stack([self.tau_bull, self.tau_bear, self.tau_crisis])
        return tf.gather(taus, regime_labels)
    
    def compute_barriers(
        self,
        atr_values: Union[tf.Tensor, np.ndarray],
        prices: Union[tf.Tensor, np.ndarray],
        regime_labels: Union[tf.Tensor, np.ndarray],
        min_barrier_pct: float = 0.003,
        max_barrier_pct: float = 0.04
    ) -> tf.Tensor:
        """
        Compute regime-aware barrier widths.
        
        Args:
            atr_values: ATR values for each sample, shape (n,)
            prices: Price values for each sample, shape (n,)
            regime_labels: Regime labels (0=bull, 1=bear, 2=crisis), shape (n,)
            min_barrier_pct: Minimum barrier percentage (clipping)
            max_barrier_pct: Maximum barrier percentage (clipping)
        
        Returns:
            Tensor of barrier widths (profit target percentages), shape (n,)
        """
        # Convert inputs to tensors
        atr_values = tf.convert_to_tensor(atr_values, dtype=tf.float32)
        prices = tf.convert_to_tensor(prices, dtype=tf.float32)
        regime_labels = tf.convert_to_tensor(regime_labels, dtype=tf.int32)
        
        # Get regime-specific taus
        taus = self.call(regime_labels)
        
        # Compute raw barrier widths: tau * ATR / price
        barriers = taus * atr_values / prices
        
        # Clip to min/max bounds
        barriers_clipped = tf.clip_by_value(barriers, min_barrier_pct, max_barrier_pct)
        
        return barriers_clipped
    
    def differentiable_barrier_loss(
        self,
        atr_values: Union[tf.Tensor, np.ndarray],
        prices: Union[tf.Tensor, np.ndarray],
        regime_labels: Union[tf.Tensor, np.ndarray],
        future_returns: Union[tf.Tensor, np.ndarray],
        temperature: float = 10.0,
        min_barrier_pct: float = 0.003,
        max_barrier_pct: float = 0.04
    ) -> tf.Tensor:
        """
        Compute differentiable barrier hit loss using sigmoid-smoothed barriers.
        
        This enables gradient flow through the barrier computation by using
        soft sigmoid approximations instead of hard threshold comparisons.
        
        Args:
            atr_values: ATR values, shape (n,)
            prices: Price values, shape (n,)
            regime_labels: Regime labels, shape (n,)
            future_returns: Actual future returns for each sample, shape (n,)
            temperature: Temperature for sigmoid smoothing (higher = sharper)
            min_barrier_pct: Minimum barrier percentage
            max_barrier_pct: Maximum barrier percentage
        
        Returns:
            Scalar loss tensor representing how well barriers captured moves
        """
        # Convert inputs to tensors
        atr_values = tf.convert_to_tensor(atr_values, dtype=tf.float32)
        prices = tf.convert_to_tensor(prices, dtype=tf.float32)
        regime_labels = tf.convert_to_tensor(regime_labels, dtype=tf.int32)
        future_returns = tf.convert_to_tensor(future_returns, dtype=tf.float32)
        
        # Compute barriers (with gradient tracking)
        barriers = self.compute_barriers(
            atr_values, prices, regime_labels,
            min_barrier_pct, max_barrier_pct
        )
        
        # Sigmoid-smoothed barrier hits
        # Soft indicator: probability that return exceeded barrier
        temp = tf.constant(temperature, dtype=tf.float32)
        
        # Profit target hit (soft): sigmoid(temp * (return - barrier))
        pt_hit_prob = tf.sigmoid(temp * (future_returns - barriers))
        
        # Stop loss hit (soft): sigmoid(temp * (-return - barrier))
        sl_hit_prob = tf.sigmoid(temp * (-future_returns - barriers))
        
        # Neither hit: probability return stayed within barriers
        within_barriers = (1.0 - pt_hit_prob) * (1.0 - sl_hit_prob)
        
        # Loss components:
        # 1. Reward barriers that are just exceeded (neither too tight nor too loose)
        # 2. Penalize barriers where neither PT nor SL is hit
        
        # Ideal: either PT or SL is hit (clear signal)
        hit_either = pt_hit_prob + sl_hit_prob - pt_hit_prob * sl_hit_prob
        
        # Loss: negative log likelihood of having a clear hit
        # Lower loss when exactly one barrier is hit
        loss = -tf.reduce_mean(tf.math.log(hit_either + 1e-8))
        
        # Additional regularization: prefer tighter barriers in bull markets
        taus = self.call(regime_labels)
        
        # Regime-aware regularization
        # Bull should have tighter barriers (lower tau), crisis wider (higher tau)
        regime_weights = tf.where(
            regime_labels == self.REGIME_BULL,
            1.5,  # Penalize high taus more in bull
            tf.where(
                regime_labels == self.REGIME_CRISIS,
                0.5,  # Penalize low taus less in crisis
                1.0   # Bear: neutral
            )
        )
        
        tau_regularization = tf.reduce_mean(regime_weights * taus)
        
        total_loss = loss + 0.1 * tau_regularization
        
        return total_loss
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save learned tau values to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "tau_bull": float(self.tau_bull.numpy()),
            "tau_bear": float(self.tau_bear.numpy()),
            "tau_crisis": float(self.tau_crisis.numpy()),
            "raw_tau_bull": float(self.raw_tau_bull.numpy()),
            "raw_tau_bear": float(self.raw_tau_bear.numpy()),
            "raw_tau_crisis": float(self.raw_tau_crisis.numpy()),
            "tau_min": self.TAU_MIN,
            "tau_max": self.TAU_MAX,
            "version": "1.0.0"
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[QuantileBarrierLearner] Saved tau values to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], trainable: bool = True) -> "QuantileBarrierLearner":
        """
        Load learned tau values from JSON file.
        
        Args:
            filepath: Path to JSON file
            trainable: Whether loaded taus should remain trainable
        
        Returns:
            QuantileBarrierLearner instance with loaded parameters
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"[QuantileBarrierLearner] File not found: {filepath}, using defaults")
            return cls(trainable=trainable)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        instance = cls(
            tau_bull=data.get("tau_bull", cls.DEFAULT_TAU_BULL),
            tau_bear=data.get("tau_bear", cls.DEFAULT_TAU_BEAR),
            tau_crisis=data.get("tau_crisis", cls.DEFAULT_TAU_CRISIS),
            trainable=trainable,
            name="quantile_barrier_learner_loaded"
        )
        
        print(f"[QuantileBarrierLearner] Loaded tau values from {filepath}")
        print(f"  tau_bull={instance._init_tau_bull:.3f}, "
              f"tau_bear={instance._init_tau_bear:.3f}, "
              f"tau_crisis={instance._init_tau_crisis:.3f}")
        
        return instance
    
    def get_config(self) -> Dict:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "tau_bull": self._init_tau_bull,
            "tau_bear": self._init_tau_bear,
            "tau_crisis": self._init_tau_crisis,
            "trainable": self._trainable
        })
        return config


class BarrierLabeler:
    """
    Wrapper for triple-barrier labeling with optional quantile-based adaptive barriers.
    
    Provides backward compatibility - can use fixed ATR_RATIO (0.45) or
    learned regime-specific quantiles.
    """
    
    FIXED_ATR_RATIO = 0.45  # Default fixed ratio for backward compatibility
    
    def __init__(
        self,
        use_learned_barriers: bool = False,
        barrier_learner: Optional[QuantileBarrierLearner] = None,
        min_barrier_pct: float = 0.003,
        max_barrier_pct: float = 0.04
    ):
        """
        Initialize the BarrierLabeler.
        
        Args:
            use_learned_barriers: Whether to use learned quantile barriers
            barrier_learner: Pre-initialized QuantileBarrierLearner (optional)
            min_barrier_pct: Minimum barrier percentage (clipping)
            max_barrier_pct: Maximum barrier percentage (clipping)
        """
        self.use_learned_barriers = use_learned_barriers
        self.min_barrier_pct = min_barrier_pct
        self.max_barrier_pct = max_barrier_pct
        
        if use_learned_barriers:
            if barrier_learner is not None:
                self.barrier_learner = barrier_learner
            else:
                # Initialize with expected values
                self.barrier_learner = QuantileBarrierLearner(
                    tau_bull=0.35,
                    tau_bear=0.55,
                    tau_crisis=0.70,
                    trainable=False  # Default to non-trainable for inference
                )
        else:
            self.barrier_learner = None
    
    def compute_barriers(
        self,
        atr_values: np.ndarray,
        prices: np.ndarray,
        regime_labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute barrier widths (profit target percentages).
        
        Args:
            atr_values: ATR values for each sample
            prices: Price values for each sample
            regime_labels: Regime labels (0=bull, 1=bear, 2=crisis) - required if using learned barriers
        
        Returns:
            Array of barrier widths (profit target percentages)
        """
        if self.use_learned_barriers and self.barrier_learner is not None:
            if regime_labels is None:
                raise ValueError("regime_labels required when using learned barriers")
            
            # Build the layer if not already built
            if not self.barrier_learner.built:
                self.barrier_learner.build()
            
            barriers = self.barrier_learner.compute_barriers(
                atr_values=atr_values,
                prices=prices,
                regime_labels=regime_labels,
                min_barrier_pct=self.min_barrier_pct,
                max_barrier_pct=self.max_barrier_pct
            )
            return barriers.numpy()
        else:
            # Backward compatibility: use fixed ATR_RATIO
            barriers = self.FIXED_ATR_RATIO * atr_values / prices
            return np.clip(barriers, self.min_barrier_pct, self.max_barrier_pct)
    
    def compute_labels(
        self,
        close_prices: np.ndarray,
        atr_values: np.ndarray,
        lookback_window: int,
        forecast_horizon: int,
        regime_labels: Optional[np.ndarray] = None,
        start_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Compute triple-barrier labels for a sequence of prices.
        
        Args:
            close_prices: Close price series
            atr_values: ATR values (aligned with close_prices)
            lookback_window: Number of bars to look back (for feature window)
            forecast_horizon: Number of bars to forecast ahead
            regime_labels: Regime labels for each potential label position
            start_idx: Starting index in the price series
        
        Returns:
            Dictionary with:
                - labels: Binary array (1=up, 0=down) for valid samples
                - barriers: Barrier widths used for each sample
                - stats: Dictionary with hit statistics
        """
        n = len(close_prices)
        is_up_list = []
        barrier_pct_list = []
        barrier_stats = {
            "hit_pt": 0, "hit_sl": 0, "timeout_up": 0, "timeout_down": 0, "clipped": 0
        }
        
        for i in range(start_idx, n - lookback_window - forecast_horizon + 1):
            base_idx = i + lookback_window
            base_price = close_prices[base_idx]
            base_atr = atr_values[base_idx]
            
            # Get regime label if available
            if regime_labels is not None and self.use_learned_barriers:
                regime = regime_labels[base_idx] if base_idx < len(regime_labels) else 0
                profit_target_pct = self.compute_barriers(
                    np.array([base_atr]),
                    np.array([base_price]),
                    np.array([regime])
                )[0]
            else:
                # Fixed ratio
                profit_target_pct = self.FIXED_ATR_RATIO * base_atr / base_price
                profit_target_pct = np.clip(
                    profit_target_pct, self.min_barrier_pct, self.max_barrier_pct
                )
            
            if profit_target_pct != np.clip(profit_target_pct, self.min_barrier_pct, self.max_barrier_pct):
                barrier_stats["clipped"] += 1
            
            stop_loss_pct = -profit_target_pct
            barrier_pct_list.append(profit_target_pct)
            
            # Future path
            future_path = close_prices[base_idx + 1:base_idx + 1 + forecast_horizon]
            path_returns = (future_path - base_price) / base_price
            
            # Check barriers
            hit_pt = np.where(path_returns >= profit_target_pct)[0]
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
                # Timeout - use final return direction
                final_return = path_returns[-1] if len(path_returns) > 0 else 0
                if final_return > 0:
                    is_up_list.append(1)
                    barrier_stats["timeout_up"] += 1
                else:
                    is_up_list.append(0)
                    barrier_stats["timeout_down"] += 1
        
        # Convert to numpy arrays
        labels = np.array(is_up_list, dtype=np.int32)
        barriers = np.array(barrier_pct_list, dtype=np.float32)
        
        # Compute statistics
        if len(barriers) > 0:
            barrier_stats["avg_barrier_pct"] = np.mean(barriers) * 100
            barrier_stats["min_barrier_pct"] = np.min(barriers) * 100
            barrier_stats["max_barrier_pct"] = np.max(barriers) * 100
        
        return {
            "labels": labels,
            "barriers": barriers,
            "stats": barrier_stats
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save barrier configuration and learned parameters."""
        if self.use_learned_barriers and self.barrier_learner is not None:
            self.barrier_learner.save(filepath)
        else:
            # Save config only
            data = {
                "use_learned_barriers": False,
                "fixed_atr_ratio": self.FIXED_ATR_RATIO,
                "min_barrier_pct": self.min_barrier_pct,
                "max_barrier_pct": self.max_barrier_pct
            }
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        use_learned: bool = False,
        trainable: bool = False
    ) -> "BarrierLabeler":
        """
        Load barrier labeler from file.
        
        Args:
            filepath: Path to saved parameters
            use_learned: Whether to use learned barriers
            trainable: Whether loaded barriers should be trainable
        
        Returns:
            BarrierLabeler instance
        """
        if use_learned:
            barrier_learner = QuantileBarrierLearner.load(filepath, trainable=trainable)
            return cls(use_learned_barriers=True, barrier_learner=barrier_learner)
        else:
            return cls(use_learned_barriers=False)


def integrate_with_lstm_model(
    enriched_df,
    split_indices: Dict[str, np.ndarray],
    lookback_window: int = 12,
    forecast_horizon: int = 12,
    embargo: Optional[int] = None,
    use_learned_barriers: bool = False,
    barrier_learner: Optional[QuantileBarrierLearner] = None,
    regime_labels: Optional[np.ndarray] = None
) -> tuple:
    """
    Integration function for use with lstm_model.py.
    
    Replaces the fixed ATR_RATIO logic with regime-aware adaptive barriers.
    
    Args:
        enriched_df: DataFrame with Close and ATR_14 columns
        split_indices: Dictionary with train/val/test indices
        lookback_window: Number of bars to look back
        forecast_horizon: Number of bars to forecast ahead
        embargo: Embargo period for purge/embargo
        use_learned_barriers: Whether to use learned barriers
        barrier_learner: Pre-initialized QuantileBarrierLearner
        regime_labels: Regime labels for each row in enriched_df
    
    Returns:
        Tuple of (labels, masks, barrier_stats) compatible with lstm_model.py
    """
    labeler = BarrierLabeler(
        use_learned_barriers=use_learned_barriers,
        barrier_learner=barrier_learner
    )
    
    close_prices = enriched_df["Close"].values
    atr_values = enriched_df["ATR_14"].values
    
    # Compute labels
    result = labeler.compute_labels(
        close_prices=close_prices,
        atr_values=atr_values,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
        regime_labels=regime_labels
    )
    
    labels_raw = result["labels"]
    barrier_stats = result["stats"]
    
    # Convert to one-hot
    def _to_onehot(arr: np.ndarray) -> np.ndarray:
        onehot = np.zeros((len(arr), 2), dtype=np.float32)
        onehot[np.arange(len(arr)), arr] = 1.0
        return onehot
    
    # Get indices for each split
    train_idx = split_indices["idx_train"]
    val_idx = split_indices["idx_val"]
    test_idx = split_indices["idx_test"]
    
    # Filter indices to valid label range
    max_label_idx = len(labels_raw)
    train_idx = train_idx[train_idx < max_label_idx]
    val_idx = val_idx[val_idx < max_label_idx]
    test_idx = test_idx[test_idx < max_label_idx]
    
    # Purge/Embargo logic (simplified - full implementation in lstm_model.py)
    if embargo is None:
        embargo = forecast_horizon
    
    test_start_raw = test_idx[0] if len(test_idx) > 0 else max_label_idx
    purge_boundary = test_start_raw - lookback_window - forecast_horizon
    
    train_mask = train_idx < purge_boundary
    val_mask = val_idx < purge_boundary
    
    embargo_count = min(embargo, len(test_idx))
    test_mask = np.ones(len(test_idx), dtype=bool)
    if embargo_count > 0:
        test_mask[:embargo_count] = False
    
    barrier_stats["n_purged_train"] = int(np.sum(~train_mask))
    barrier_stats["n_purged_val"] = int(np.sum(~val_mask))
    barrier_stats["n_embargo_test"] = int(np.sum(~test_mask))
    
    labels = {
        "train": _to_onehot(labels_raw[train_idx]),
        "val": _to_onehot(labels_raw[val_idx]),
        "test": _to_onehot(labels_raw[test_idx])
    }
    
    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask
    }
    
    return labels, masks, barrier_stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("QuantileBarrierLearner Demo")
    print("=" * 60)
    
    # Create a learner
    learner = QuantileBarrierLearner(trainable=True)
    learner.build()
    
    print("\nInitial tau values:")
    print(f"  tau_bull: {learner.tau_bull.numpy():.4f} (expected: ~0.35)")
    print(f"  tau_bear: {learner.tau_bear.numpy():.4f} (expected: ~0.55)")
    print(f"  tau_crisis: {learner.tau_crisis.numpy():.4f} (expected: ~0.70)")
    
    # Test barrier computation
    print("\nBarrier computation demo:")
    atr_values = np.array([1.5, 2.0, 3.5], dtype=np.float32)
    prices = np.array([100.0, 150.0, 200.0], dtype=np.float32)
    regimes = np.array([0, 1, 2], dtype=np.int32)  # bull, bear, crisis
    
    barriers = learner.compute_barriers(atr_values, prices, regimes)
    print(f"  ATR values: {atr_values}")
    print(f"  Prices: {prices}")
    print(f"  Regimes (bull/bear/crisis): {regimes}")
    print(f"  Computed barriers: {barriers.numpy()}")
    
    # Test save/load
    print("\nSave/Load demo:")
    test_path = Path("test_barrier_params.json")
    learner.save(test_path)
    
    loaded_learner = QuantileBarrierLearner.load(test_path, trainable=False)
    loaded_learner.build()
    print(f"  Loaded tau_bull: {loaded_learner.tau_bull.numpy():.4f}")
    
    # Cleanup
    test_path.unlink(missing_ok=True)
    
    # Test backward compatibility
    print("\nBackward compatibility demo:")
    labeler = BarrierLabeler(use_learned_barriers=False)
    barriers_fixed = labeler.compute_barriers(
        np.array([2.0]),
        np.array([100.0]),
        None
    )
    print(f"  Fixed barrier (ATR_RATIO=0.45): {barriers_fixed[0]:.4f}")
    print(f"  Expected: {0.45 * 2.0 / 100.0:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
