"""
State-Aware Meta-Learning (SAML) for Regime-Adaptive Initialization
=====================================================================
Implements MAML/Reptile-style meta-learning to learn initial weights
that enable fast adaptation to market regime changes.

Key Concepts:
- Meta-learning initialization using MAML/Reptile-style updates
- Inner loop: fast adaptation to regime-specific data (5 steps)
- Outer loop: meta-update to improve initial weights
- Regime persistence tracking (weight by regime duration)

Usage:
------
# Create SAML meta-learner
saml = SAMLMetaLearner(
    input_shape=(24, n_features),
    inner_lr=0.01,
    meta_lr=0.001,
    inner_steps=5,
)

# Meta-train across regimes
regime_data = {
    "BULL": (X_bull_train, y_bull_train, X_bull_val, y_bull_val),
    "BEAR": (X_bear_train, y_bear_train, X_bear_val, y_bear_val),
    "CRISIS": (X_crisis_train, y_crisis_train, X_crisis_val, y_crisis_val),
}
saml.meta_train_step(regime_data, epochs=50)

# Adapt meta-weights to specific regime
specialist_model = saml.adapt_to_regime("BEAR", X_bear_train, y_bear_train)

# Save/load meta-learner
saml.save("models/saml")
saml = SAMLMetaLearner.load("models/saml", input_shape=(24, n_features))

# Initialize specialist from SAML weights
from lstm_model import LSTMModel
lstm = LSTMModel()
lstm.build_model(input_shape=(24, n_features))
saml.initialize_specialist(lstm.model)
"""

from __future__ import annotations

import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback

tf.config.optimizer.set_jit(True)


class RegimePersistenceTracker:
    """
    Tracks regime duration and persistence for weighted meta-learning.
    Longer-lasting regimes contribute more to meta-updates.
    """

    def __init__(self, regime_names: List[str] = None):
        self.regime_names = regime_names or ["BULL", "BEAR", "CRISIS"]
        self.regime_stats: Dict[str, Dict[str, float]] = {
            name: {"total_duration": 0.0, "episode_count": 0, "avg_loss": 0.0}
            for name in self.regime_names
        }
        self.current_regime: Optional[str] = None
        self.current_start_idx: int = 0
        self.persistence_weights: Dict[str, float] = {
            name: 1.0 for name in self.regime_names
        }

    def update_regime_duration(self, regime: str, duration: int, avg_loss: float):
        """Update statistics for a completed regime episode."""
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                "total_duration": 0.0, "episode_count": 0, "avg_loss": 0.0
            }

        self.regime_stats[regime]["total_duration"] += duration
        self.regime_stats[regime]["episode_count"] += 1

        # Running average of loss
        n = self.regime_stats[regime]["episode_count"]
        old_avg = self.regime_stats[regime]["avg_loss"]
        self.regime_stats[regime]["avg_loss"] = (old_avg * (n - 1) + avg_loss) / n

        self._recalculate_weights()

    def _recalculate_weights(self):
        """Recalculate persistence weights based on regime statistics."""
        total_duration = sum(
            stats["total_duration"] for stats in self.regime_stats.values()
        )

        if total_duration > 0:
            for regime, stats in self.regime_stats.items():
                # Weight proportional to duration, normalized
                duration_weight = stats["total_duration"] / total_duration

                # Inverse loss weight (lower loss = higher weight)
                loss_weight = 1.0 / (1.0 + stats["avg_loss"])

                # Combined weight: duration * performance
                self.persistence_weights[regime] = duration_weight * loss_weight

        # Normalize to sum to number of regimes
        total_weight = sum(self.persistence_weights.values())
        if total_weight > 0:
            n_regimes = len(self.regime_names)
            for regime in self.persistence_weights:
                self.persistence_weights[regime] *= n_regimes / total_weight

    def get_weight(self, regime: str) -> float:
        """Get persistence weight for a regime."""
        return self.persistence_weights.get(regime, 1.0)

    def get_all_weights(self) -> Dict[str, float]:
        """Get all persistence weights."""
        return self.persistence_weights.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "regime_stats": self.regime_stats,
            "persistence_weights": self.persistence_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimePersistenceTracker":
        """Deserialize from dictionary."""
        tracker = cls()
        tracker.regime_stats = data.get("regime_stats", tracker.regime_stats)
        tracker.persistence_weights = data.get(
            "persistence_weights", tracker.persistence_weights
        )
        return tracker


class SAMLMetaLearner:
    """
    State-Aware Meta-Learning (SAML) for regime-adaptive initialization.

    Implements MAML/Reptile-style meta-learning where:
    - Inner loop: fast adaptation to regime-specific data
    - Outer loop: meta-update to learn better initial weights
    - Regime persistence: weight regimes by duration/performance

    Parameters
    ----------
    input_shape : tuple
        (lookback_window, n_features) shape of input sequences
    inner_lr : float
        Learning rate for inner loop (task-specific) adaptation
    meta_lr : float
        Learning rate for outer loop (meta) updates
    inner_steps : int
        Number of gradient steps in inner loop
    lstm_units : int
        Number of LSTM units in base architecture
    dropout_rate : float
        Dropout rate for regularization
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
        lstm_units: int = 16,
        dropout_rate: float = 0.3,
    ):
        self.input_shape = input_shape
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        # Meta-model (theta): the initialization we want to learn
        self.meta_model: Optional[keras.Model] = None

        # Persistence tracker for regime weighting
        self.persistence_tracker = RegimePersistenceTracker()

        # Training history
        self.meta_history: Dict[str, List[float]] = {
            "meta_loss": [],
            "avg_inner_loss": [],
        }

        self._build_meta_model()

    def _build_meta_model(self) -> keras.Model:
        """
        Build the base LSTM architecture that will serve as meta-initialization.
        Matches the architecture from lstm_model.py for compatibility.
        """
        inputs = layers.Input(shape=self.input_shape, name="saml_input")
        x = layers.LSTM(self.lstm_units, name="saml_lstm")(inputs)
        x = layers.Dropout(self.dropout_rate, name="saml_dropout")(x)
        outputs = layers.Dense(2, activation="softmax", name="saml_output")(x)

        self.meta_model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="saml_meta_learner",
        )

        self.meta_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.meta_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.meta_model

    def _clone_model(self) -> keras.Model:
        """Create a fresh clone of the meta-model architecture."""
        inputs = layers.Input(shape=self.input_shape, name="saml_input")
        x = layers.LSTM(self.lstm_units, name="saml_lstm")(inputs)
        x = layers.Dropout(self.dropout_rate, name="saml_dropout")(x)
        outputs = layers.Dense(2, activation="softmax", name="saml_output")(x)

        clone = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="saml_task_model",
        )
        clone.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.inner_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return clone

    def _inner_loop(
        self,
        task_model: keras.Model,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_query: np.ndarray,
        y_query: np.ndarray,
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Inner loop: Fast adaptation to task-specific data.

        Performs `inner_steps` gradient updates on support set,
        then evaluates on query set.

        Returns
        -------
        query_loss : float
            Loss after adaptation on query set
        adapted_weights : list of arrays
            Final weights after inner loop adaptation
        """
        # Train on support set for inner_steps
        task_model.fit(
            X_support,
            y_support,
            epochs=self.inner_steps,
            batch_size=min(32, len(X_support)),
            verbose=0,
        )

        # Evaluate on query set
        query_loss, query_acc = task_model.evaluate(
            X_query, y_query, verbose=0
        )

        # Get adapted weights
        adapted_weights = task_model.get_weights()

        return query_loss, adapted_weights

    def meta_train_step(
        self,
        regime_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        epochs: int = 50,
        meta_batch_size: int = 2,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Meta-training across all regimes using Reptile-style updates.

        Parameters
        ----------
        regime_data : dict
            {regime_name: (X_train, y_train, X_val, y_val)} for each regime
        epochs : int
            Number of meta-training epochs
        meta_batch_size : int
            Number of regimes to sample per meta-batch
        early_stopping_patience : int
            Early stopping patience for meta-training

        Returns
        -------
        history : dict
            Meta-training history with meta_loss and avg_inner_loss
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not built. Call _build_meta_model() first.")

        print("\n" + "=" * 60)
        print("SAML META-TRAINING")
        print("=" * 60)
        print(f"Regimes: {list(regime_data.keys())}")
        print(f"Inner steps: {self.inner_steps}, Inner LR: {self.inner_lr}")
        print(f"Meta LR: {self.meta_lr}, Meta epochs: {epochs}")
        print(f"Meta batch size: {meta_batch_size}")

        regime_names = list(regime_data.keys())
        best_meta_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_meta_losses = []
            epoch_inner_losses = []

            # Sample regimes for this meta-batch
            sampled_regimes = np.random.choice(
                regime_names,
                size=min(meta_batch_size, len(regime_names)),
                replace=False,
            )

            # Store adapted weights from each task
            task_adapted_weights = []
            task_weights = []

            for regime in sampled_regimes:
                X_train, y_train, X_val, y_val = regime_data[regime]

                # Clone meta-model for this task
                task_model = self._clone_model()
                task_model.set_weights(self.meta_model.get_weights())

                # Inner loop adaptation
                inner_loss, adapted_weights = self._inner_loop(
                    task_model, X_train, y_train, X_val, y_val
                )

                epoch_inner_losses.append(inner_loss)
                task_adapted_weights.append(adapted_weights)

                # Get persistence weight for this regime
                persistence_weight = self.persistence_tracker.get_weight(regime)
                task_weights.append(persistence_weight)

                del task_model
                gc.collect()

            # Reptile-style meta-update: move meta-weights toward mean of adapted weights
            if task_adapted_weights:
                meta_weights = self.meta_model.get_weights()
                new_meta_weights = []

                # Normalize weights
                total_weight = sum(task_weights)
                if total_weight > 0:
                    task_weights = [w / total_weight for w in task_weights]

                for i in range(len(meta_weights)):
                    # Weighted average of adapted weights
                    avg_adapted = sum(
                        w[i] * tw for w, tw in zip(task_adapted_weights, task_weights)
                    )

                    # Reptile update: theta = theta + meta_lr * (avg_adapted - theta)
                    new_weight = meta_weights[i] + self.meta_lr * (
                        avg_adapted - meta_weights[i]
                    )
                    new_meta_weights.append(new_weight)

                self.meta_model.set_weights(new_meta_weights)

                # Calculate meta-loss as weighted average of task losses
                meta_loss = sum(l * w for l, w in zip(epoch_inner_losses, task_weights))
                epoch_meta_losses.append(meta_loss)

                # Update persistence tracker
                for regime, loss in zip(sampled_regimes, epoch_inner_losses):
                    duration = len(regime_data[regime][0])
                    self.persistence_tracker.update_regime_duration(
                        regime, duration, loss
                    )

            # Record history
            avg_meta_loss = np.mean(epoch_meta_losses) if epoch_meta_losses else 0.0
            avg_inner_loss = np.mean(epoch_inner_losses) if epoch_inner_losses else 0.0

            self.meta_history["meta_loss"].append(avg_meta_loss)
            self.meta_history["avg_inner_loss"].append(avg_inner_loss)

            if True:  # Print epoch info
                print(
                    f"Epoch {epoch+1}/{epochs}: meta_loss={avg_meta_loss:.4f}, "
                    f"avg_inner_loss={avg_inner_loss:.4f}"
                )
                print(
                    f"  Persistence weights: {self.persistence_tracker.get_all_weights()}"
                )

            # Early stopping
            if avg_meta_loss < best_meta_loss:
                best_meta_loss = avg_meta_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("\nMeta-training complete.")
        print(f"Best meta-loss: {best_meta_loss:.4f}")
        print(f"Final persistence weights: {self.persistence_tracker.get_all_weights()}")

        return self.meta_history

    def adapt_to_regime(
        self,
        regime: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        adaptation_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> keras.Model:
        """
        Adapt meta-weights to a specific regime (create a specialist).

        Parameters
        ----------
        regime : str
            Regime name (e.g., "BULL", "BEAR", "CRISIS")
        X_train : np.ndarray
            Training data for this regime
        y_train : np.ndarray
            Training labels for this regime
        adaptation_steps : int, optional
            Number of adaptation steps (default: self.inner_steps)
        verbose : bool
            Whether to print adaptation info

        Returns
        -------
        specialist_model : keras.Model
            Fine-tuned model for this regime
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized. Train or load first.")

        if verbose:
            print(f"\n[Adapt] Creating {regime} specialist from meta-weights...")

        # Clone and initialize from meta-weights
        specialist = self._clone_model()
        specialist.set_weights(self.meta_model.get_weights())

        steps = adaptation_steps or self.inner_steps

        # Fine-tune on regime-specific data
        history = specialist.fit(
            X_train,
            y_train,
            epochs=steps,
            batch_size=min(32, len(X_train)),
            verbose=1 if verbose else 0,
        )

        final_loss = history.history["loss"][-1] if history.history.get("loss") else 0.0

        if verbose:
            print(f"[Adapt] {regime} specialist adapted: final_loss={final_loss:.4f}")

        return specialist

    def initialize_specialist(self, model: keras.Model) -> None:
        """
        Initialize a specialist model (e.g., LSTMModel) from SAML meta-weights.

        Parameters
        ----------
        model : keras.Model
            Model to initialize from meta-weights. Must have compatible architecture.
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized. Train or load first.")

        # Transfer weights from meta-model to specialist
        meta_weights = self.meta_model.get_weights()

        try:
            model.set_weights(meta_weights)
            print("[SAML] Specialist initialized from meta-weights.")
        except ValueError as e:
            print(f"[SAML] Warning: Could not initialize specialist: {e}")
            print("[SAML] Architecture mismatch - specialist will use random initialization.")

    def create_regime_specialists(
        self,
        regime_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        save_dir: str = "models/saml/specialists",
    ) -> Dict[str, keras.Model]:
        """
        Create and save specialist models for all regimes.

        Parameters
        ----------
        regime_data : dict
            {regime_name: (X_train, y_train, X_val, y_val)} for each regime
        save_dir : str
            Directory to save specialist models

        Returns
        -------
        specialists : dict
            {regime_name: specialist_model} for each regime
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        specialists = {}

        print("\n" + "=" * 60)
        print("CREATING REGIME SPECIALISTS")
        print("=" * 60)

        for regime, (X_train, y_train, X_val, y_val) in regime_data.items():
            print(f"\nCreating {regime} specialist...")

            # Adapt to regime
            specialist = self.adapt_to_regime(regime, X_train, y_train)

            # Validate on validation set
            val_loss, val_acc = specialist.evaluate(X_val, y_val, verbose=0)
            print(f"  {regime} specialist: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Save specialist
            regime_dir = save_path / regime.lower()
            regime_dir.mkdir(parents=True, exist_ok=True)
            specialist.save_weights(str(regime_dir / "specialist.weights.h5"))

            specialists[regime] = specialist

        # Save metadata
        meta = {
            "regimes": list(regime_data.keys()),
            "input_shape": self.input_shape,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "inner_lr": self.inner_lr,
            "inner_steps": self.inner_steps,
        }
        with open(save_path / "specialists_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nAll specialists saved to: {save_path}")
        return specialists

    def save(self, save_dir: str = "models/saml") -> None:
        """
        Save the meta-learner state.

        Parameters
        ----------
        save_dir : str
            Directory to save meta-learner state
        """
        if self.meta_model is None:
            raise ValueError("No meta-model to save.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save meta-model weights
        self.meta_model.save_weights(str(save_path / "saml_meta.weights.h5"))

        # Save configuration
        config = {
            "input_shape": self.input_shape,
            "inner_lr": self.inner_lr,
            "meta_lr": self.meta_lr,
            "inner_steps": self.inner_steps,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
        }
        with open(save_path / "saml_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save persistence tracker
        with open(save_path / "persistence_tracker.pkl", "wb") as f:
            pickle.dump(self.persistence_tracker.to_dict(), f)

        # Save training history
        with open(save_path / "saml_history.pkl", "wb") as f:
            pickle.dump(self.meta_history, f)

        print(f"[SAML] Saved to {save_path}")

    @classmethod
    def load(
        cls,
        save_dir: str = "models/saml",
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> "SAMLMetaLearner":
        """
        Load a saved meta-learner.

        Parameters
        ----------
        save_dir : str
            Directory containing saved meta-learner
        input_shape : tuple, optional
            Override input shape from config

        Returns
        -------
        saml : SAMLMetaLearner
            Loaded meta-learner instance
        """
        save_path = Path(save_dir)

        # Load configuration
        with open(save_path / "saml_config.json", "r") as f:
            config = json.load(f)

        if input_shape is not None:
            config["input_shape"] = input_shape

        # Create instance
        saml = cls(**config)

        # Load meta-model weights
        saml.meta_model.load_weights(str(save_path / "saml_meta.weights.h5"))

        # Load persistence tracker
        tracker_path = save_path / "persistence_tracker.pkl"
        if tracker_path.exists():
            with open(tracker_path, "rb") as f:
                tracker_dict = pickle.load(f)
                saml.persistence_tracker = RegimePersistenceTracker.from_dict(tracker_dict)

        # Load history
        history_path = save_path / "saml_history.pkl"
        if history_path.exists():
            with open(history_path, "rb") as f:
                saml.meta_history = pickle.load(f)

        print(f"[SAML] Loaded from {save_path}")
        return saml

    def get_meta_weights(self) -> List[np.ndarray]:
        """Get current meta-weights (initialization parameters)."""
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized.")
        return self.meta_model.get_weights()

    def set_meta_weights(self, weights: List[np.ndarray]) -> None:
        """Set meta-weights directly."""
        if self.meta_model is None:
            raise ValueError("Meta-model not initialized.")
        self.meta_model.set_weights(weights)


class SAMLIntegrationHelper:
    """
    Integration helpers for using SAML with existing LSTMModel and pipeline.

    Provides utilities to:
    - Initialize specialists from SAML weights
    - Create regime-specific models
    - Integrate with the existing ensemble pipeline
    """

    @staticmethod
    def initialize_lstm_from_saml(
        lstm_model: "LSTMModel",
        saml_learner: SAMLMetaLearner,
    ) -> None:
        """
        Initialize an LSTMModel from SAML meta-weights.

        Parameters
        ----------
        lstm_model : LSTMModel
            The LSTM model to initialize
        saml_learner : SAMLMetaLearner
            Trained SAML meta-learner
        """
        if lstm_model.model is None:
            raise ValueError("LSTMModel must be built before initialization.")

        saml_learner.initialize_specialist(lstm_model.model)

    @staticmethod
    def create_regime_specialist_for_group(
        saml_learner: SAMLMetaLearner,
        regime: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_path: str,
    ) -> "LSTMModel":
        """
        Create a regime-specialized LSTMModel for a group.

        Parameters
        ----------
        saml_learner : SAMLMetaLearner
            Trained SAML meta-learner
        regime : str
            Regime name
        X_train : np.ndarray
            Training data for this regime
        y_train : np.ndarray
            Training labels
        model_path : str
            Path to save the specialist model

        Returns
        -------
        specialist : LSTMModel
            Adapted specialist model
        """
        from lstm_model import LSTMModel

        # Create LSTM model
        specialist = LSTMModel(model_path=model_path)
        specialist.build_model(input_shape=saml_learner.input_shape)

        # Initialize from SAML meta-weights
        saml_learner.initialize_specialist(specialist.model)

        # Further adapt to regime-specific data
        # (Optional: could call adapt_to_regime instead)

        return specialist

    @staticmethod
    def get_saml_ready_regime_data(
        enriched_df: "pd.DataFrame",
        regime_detector: "RegimeDetector",
        X_sequences: np.ndarray,
        y_labels: np.ndarray,
        split_indices: Dict[str, np.ndarray],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare regime-segregated data for SAML meta-training.

        Parameters
        ----------
        enriched_df : pd.DataFrame
            Enriched data with features
        regime_detector : RegimeDetector
            Fitted regime detector
        X_sequences : np.ndarray
            Sequence data
        y_labels : np.ndarray
            Labels (one-hot encoded)
        split_indices : dict
            Train/val/test indices

        Returns
        -------
        regime_data : dict
            {regime: (X_train, y_train, X_val, y_val)} for each regime
        """
        # Detect regimes for all data points
        regimes = regime_detector.predict_regime_series(enriched_df)

        # Map sequence indices to regime labels
        # Assume sequences are aligned: sequence i corresponds to regime at index i + lookback
        lookback = 24  # Default lookback

        regime_data = {}

        for regime_name in ["BULL", "BEAR", "CRISIS"]:
            # Find indices where regime matches
            regime_mask = regimes == regime_name

            # Map to sequence indices (account for lookback)
            # Sequence at position j uses data up to index j + lookback
            sequence_regimes = []
            for i in range(len(X_sequences)):
                data_idx = i + lookback
                if data_idx < len(regimes):
                    sequence_regimes.append(regime_mask[data_idx])
                else:
                    sequence_regimes.append(False)

            sequence_regimes = np.array(sequence_regimes)

            # Split into train/val using provided indices
            train_mask = np.isin(np.arange(len(X_sequences)), split_indices["idx_train"])
            val_mask = np.isin(np.arange(len(X_sequences)), split_indices["idx_val"])

            regime_train_mask = sequence_regimes & train_mask
            regime_val_mask = sequence_regimes & val_mask

            if regime_train_mask.sum() > 0 and regime_val_mask.sum() > 0:
                X_regime_train = X_sequences[regime_train_mask]
                y_regime_train = y_labels[regime_train_mask]
                X_regime_val = X_sequences[regime_val_mask]
                y_regime_val = y_labels[regime_val_mask]

                regime_data[regime_name] = (
                    X_regime_train,
                    y_regime_train,
                    X_regime_val,
                    y_regime_val,
                )

                print(
                    f"[SAML] {regime_name}: {len(X_regime_train)} train, "
                    f"{len(X_regime_val)} val sequences"
                )

        return regime_data


def demo_saml_training():
    """
    Demo function showing SAML usage with synthetic data.
    """
    print("\n" + "=" * 60)
    print("SAML DEMO: State-Aware Meta-Learning")
    print("=" * 60)

    # Generate synthetic data for 3 regimes
    np.random.seed(42)
    n_features = 10
    lookback = 24

    regime_data = {}

    for regime, n_samples in [("BULL", 1000), ("BEAR", 800), ("CRISIS", 400)]:
        X = np.random.randn(n_samples, lookback, n_features).astype(np.float32)

        # Different patterns per regime
        if regime == "BULL":
            y_bias = [0.6, 0.4]  # More UP
        elif regime == "BEAR":
            y_bias = [0.4, 0.6]  # More DOWN
        else:
            y_bias = [0.5, 0.5]  # Balanced

        y = np.random.choice([0, 1], size=n_samples, p=y_bias)
        y_onehot = np.zeros((n_samples, 2), dtype=np.float32)
        y_onehot[np.arange(n_samples), y] = 1.0

        # 80/20 split
        split = int(0.8 * n_samples)
        regime_data[regime] = (
            X[:split],
            y_onehot[:split],
            X[split:],
            y_onehot[split:],
        )

    # Create and train SAML
    saml = SAMLMetaLearner(
        input_shape=(lookback, n_features),
        inner_lr=0.01,
        meta_lr=0.001,
        inner_steps=5,
    )

    # Meta-train
    saml.meta_train_step(regime_data, epochs=20)

    # Create specialists
    specialists = saml.create_regime_specialists(regime_data)

    # Save
    saml.save("models/saml_demo")

    # Load and verify
    loaded_saml = SAMLMetaLearner.load("models/saml_demo")
    print(f"\n[SAML] Loaded meta-learner with persistence weights: "
          f"{loaded_saml.persistence_tracker.get_all_weights()}")

    print("\n" + "=" * 60)
    print("SAML DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_saml_training()
