import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback

tf.config.optimizer.set_jit(True)
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import pickle
import json
import pandas as pd
from enum import IntEnum


class Regime(IntEnum):
    """Enum for market regime states."""
    BULL = 0
    BEAR = 1
    CRISIS = 2


REGIME_NAMES = {Regime.BULL: "BULL", Regime.BEAR: "BEAR", Regime.CRISIS: "CRISIS"}
REGIME_TO_IDX = {"BULL": Regime.BULL, "BEAR": Regime.BEAR, "CRISIS": Regime.CRISIS}


class SpecialistConfig:
    """Configuration for a regime specialist model."""

    def __init__(
        self,
        name: str,
        train_regimes: List[Regime],
        dropout_rate: float = 0.3,
        lstm_units: int = 16,
        learning_rate: float = 1e-3,
    ):
        self.name = name
        self.train_regimes = train_regimes
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate


# Predefined specialist configurations
SPECIALIST_CONFIGS = {
    "BULL_JOINT": SpecialistConfig(
        name="BULL_JOINT",
        train_regimes=[Regime.BULL, Regime.CRISIS],
        dropout_rate=0.3,
        lstm_units=16,
        learning_rate=1e-3,
    ),
    "BEAR_ISOLATED": SpecialistConfig(
        name="BEAR_ISOLATED",
        train_regimes=[Regime.BEAR],
        dropout_rate=0.35,
        lstm_units=16,
        learning_rate=1e-3,
    ),
    "CRISIS_ISOLATED": SpecialistConfig(
        name="CRISIS_ISOLATED",
        train_regimes=[Regime.CRISIS],
        dropout_rate=0.5,  # Higher dropout for volatile crisis regime
        lstm_units=16,
        learning_rate=5e-4,  # Lower learning rate for stability
    ),
}


class RegimeCompositeCheckpoint(Callback):
    """
    Custom callback to save the model based on a composite metric.
    Score = val_accuracy - (alpha * val_loss)
    """

    def __init__(self, filepath: str, alpha: float = 0.5):
        super(RegimeCompositeCheckpoint, self).__init__()
        self.filepath = filepath
        self.alpha = alpha
        self.best_score = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")
        if val_acc is not None and val_loss is not None:
            current_score = val_acc - (self.alpha * val_loss)
            if current_score > self.best_score:
                print(
                    f"\nEpoch {epoch+1}: Composite score improved from {self.best_score:.4f} "
                    f"to {current_score:.4f} (Acc: {val_acc:.4f}, Loss: {val_loss:.4f}). Saving..."
                )
                self.best_score = current_score
                self.model.save_weights(self.filepath)


class RegimeSpecialistLSTM:
    """
    Single LSTM specialist model trained for specific market regimes.
    Extends the base LSTM architecture with regime-aware training.
    """

    def __init__(
        self,
        specialist_name: str,
        base_model_dir: str = "models/regime_specialists",
    ):
        self.specialist_name = specialist_name
        self.config = SPECIALIST_CONFIGS[specialist_name]
        self.base_model_dir = Path(base_model_dir)
        self.model_dir = self.base_model_dir / specialist_name.lower()
        self.model_path = self.model_dir / "model.weights.h5"
        self.history_path = self.model_dir / "training_history.pkl"
        self.config_path = self.model_dir / "config.json"

        self.model = None
        self.history = None
        self.input_shape = None

    def build_model(
        self,
        input_shape: Tuple[int, int],
        use_regime_embedding: bool = False,
    ) -> keras.Model:
        """
        Build LSTM model with optional regime embedding.

        Args:
            input_shape: (lookback_window, n_features)
            use_regime_embedding: Whether to include 3-state regime one-hot embedding
        """
        print("\n" + "=" * 60)
        print(f"BUILDING {self.specialist_name} SPECIALIST MODEL")
        print("=" * 60)
        print(f"Train regimes: {[REGIME_NAMES[r] for r in self.config.train_regimes]}")
        print(f"Dropout rate: {self.config.dropout_rate}")
        print(f"LSTM units: {self.config.lstm_units}")
        print(f"Learning rate: {self.config.learning_rate}")

        self.input_shape = input_shape
        lookback, n_features = input_shape

        # Main input sequence
        inputs = layers.Input(shape=input_shape, name="input_sequences")

        x = inputs

        # Optional regime embedding layer (3-state one-hot: BULL, BEAR, CRISIS)
        if use_regime_embedding:
            regime_input = layers.Input(shape=(3,), name="regime_embedding")
            regime_expanded = layers.RepeatVector(lookback)(regime_input)
            x = layers.Concatenate(axis=-1)([inputs, regime_expanded])
            n_features_with_regime = n_features + 3
            x = layers.LSTM(
                self.config.lstm_units,
                name=f"lstm_{self.specialist_name.lower()}",
            )(x)
        else:
            n_features_with_regime = n_features
            x = layers.LSTM(
                self.config.lstm_units,
                name=f"lstm_{self.specialist_name.lower()}",
            )(inputs)

        # Regime-specific dropout rate
        x = layers.Dropout(self.config.dropout_rate, name="dropout")(x)

        # Output layer
        classification_output = layers.Dense(
            2, activation="softmax", name="direction_classification"
        )(x)

        if use_regime_embedding:
            self.model = keras.Model(
                inputs=[inputs, regime_input],
                outputs=classification_output,
                name=f"regime_specialist_{self.specialist_name.lower()}",
            )
        else:
            self.model = keras.Model(
                inputs=inputs,
                outputs=classification_output,
                name=f"regime_specialist_{self.specialist_name.lower()}",
            )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("\nModel architecture:")
        self.model.summary()

        return self.model

    def filter_by_regime(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter training samples to only those matching the specialist's regimes.

        Args:
            X: Input sequences, shape (n_samples, lookback, n_features)
            y: Target labels, shape (n_samples, 2)
            regime_labels: Regime labels per sample, shape (n_samples,)

        Returns:
            Filtered X and y matching the specialist's training regimes
        """
        mask = np.isin(regime_labels, [r.value for r in self.config.train_regimes])
        n_filtered = int(np.sum(mask))
        print(
            f"[{self.specialist_name}] Filtered {n_filtered}/{len(X)} samples "
            f"({n_filtered/len(X)*100:.1f}%) matching regimes "
            f"{[REGIME_NAMES[r] for r in self.config.train_regimes]}"
        )
        return X[mask], y[mask]

    def train(
        self,
        X_train: np.ndarray,
        y_train_classification: np.ndarray,
        X_val: np.ndarray,
        y_val_classification: np.ndarray,
        regime_train: Optional[np.ndarray] = None,
        regime_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute training loop with regime-aware filtering.

        Args:
            X_train: Training sequences
            y_train_classification: Training labels (one-hot)
            X_val: Validation sequences
            y_val_classification: Validation labels (one-hot)
            regime_train: Optional regime labels for training samples
            regime_val: Optional regime labels for validation samples
            epochs: Max training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            class_weight: Optional class weights for imbalanced data
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Filter by regime if labels provided
        if regime_train is not None:
            X_train, y_train_classification = self.filter_by_regime(
                X_train, y_train_classification, regime_train
            )
        if regime_val is not None:
            X_val, y_val_classification = self.filter_by_regime(
                X_val, y_val_classification, regime_val
            )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError(
                f"[{self.specialist_name}] No training/validation samples after regime filtering"
            )

        print("\n" + "=" * 60)
        print(f"TRAINING {self.specialist_name} SPECIALIST")
        print("=" * 60)
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Early stopping patience: {early_stopping_patience}")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=False,
            verbose=1,
            mode="max",
        )

        composite_checkpoint = RegimeCompositeCheckpoint(
            filepath=str(self.model_path),
            alpha=0.5,
        )

        self.history = self.model.fit(
            X_train,
            y_train_classification,
            validation_data=(X_val, y_val_classification),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, composite_checkpoint],
            class_weight=class_weight,
            verbose=1,
        )

        # Load best weights
        try:
            self.load_model()
            print(f"\nBest model weights restored from: {self.model_path}")
        except Exception as e:
            print(f"\nWarning: Could not reload best model weights: {e}")

        return self.history.history

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate directional predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")

        predicted_directions = self.model.predict(X, verbose=0)
        predicted_class = np.argmax(predicted_directions, axis=1)
        predicted_text = np.where(predicted_class == 0, "DOWN", "UP")
        confidence = np.max(predicted_directions, axis=1)

        return {
            "direction_probabilities": predicted_directions,
            "direction": predicted_text,
            "direction_confidence": confidence,
            "specialist_name": self.specialist_name,
        }

    def save_model(self) -> None:
        """Save model weights and config."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.model_path))

        # Save config
        config_dict = {
            "specialist_name": self.specialist_name,
            "train_regimes": [REGIME_NAMES[r] for r in self.config.train_regimes],
            "dropout_rate": self.config.dropout_rate,
            "lstm_units": self.config.lstm_units,
            "learning_rate": self.config.learning_rate,
            "input_shape": self.input_shape,
        }
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"Model saved to: {self.model_dir}")

    def save(self) -> None:
        """Save model weights, config, and training history."""
        self.save_model()
        if self.history is not None:
            self.save_history()

    def load_model(self) -> None:
        """Load model weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        self.model.load_weights(str(self.model_path))

    def save_history(self) -> None:
        """Save training history."""
        if self.history is None:
            raise ValueError("No training history to save.")
        with open(self.history_path, "wb") as f:
            pickle.dump(self.history.history, f)

    def load_history(self) -> Dict[str, list]:
        """Load training history."""
        if not self.history_path.exists():
            raise FileNotFoundError(f"History not found at {self.history_path}")
        with open(self.history_path, "rb") as f:
            return pickle.load(f)


def route_to_specialist(
    regime: str,
    regime_confidence: Dict[str, float],
    confidence_threshold: float = 0.6,
    bull_crisis_overlap_threshold: float = 0.25,
) -> str:
    """
    Route to appropriate specialist based on regime and confidence scores.

    Implements Hard Model Routing logic:
    - BULL with high confidence → BULL_JOINT
    - BEAR with high confidence → BEAR_ISOLATED
    - CRISIS with high confidence → CRISIS_ISOLATED
    - Low confidence or ambiguous → BULL_JOINT (default/most robust)

    Args:
        regime: Predicted regime ("BULL", "BEAR", "CRISIS")
        regime_confidence: Dict with keys "BULL", "BEAR", "CRISIS" and probabilities
        confidence_threshold: Minimum confidence to use regime-specific specialist
        bull_crisis_overlap_threshold: Threshold for detecting BULL/CRISIS ambiguity

    Returns:
        Specialist name: "BULL_JOINT", "BEAR_ISOLATED", or "CRISIS_ISOLATED"
    """
    regime = regime.upper()
    bull_prob = regime_confidence.get("BULL", 0.0)
    bear_prob = regime_confidence.get("BEAR", 0.0)
    crisis_prob = regime_confidence.get("CRISIS", 0.0)

    max_prob = max(bull_prob, bear_prob, crisis_prob)

    # Low confidence: fall back to BULL_JOINT (most robust)
    if max_prob < confidence_threshold:
        return "BULL_JOINT"

    # Route based on regime with confidence check
    if regime == "BEAR" and bear_prob >= confidence_threshold:
        return "BEAR_ISOLATED"

    if regime == "CRISIS" and crisis_prob >= confidence_threshold:
        # Check if BULL is also probable (joint regime)
        if bull_prob > bull_crisis_overlap_threshold:
            return "BULL_JOINT"
        return "CRISIS_ISOLATED"

    if regime == "BULL" and bull_prob >= confidence_threshold:
        # Check for crisis overlap
        if crisis_prob > bull_crisis_overlap_threshold:
            return "BULL_JOINT"
        return "BULL_JOINT"

    # Default fallback
    return "BULL_JOINT"


class RegimeSpecialistEnsemble:
    """
    Hard Model Routing ensemble with 3 regime-specialist LSTMs.
    Routes inputs to the appropriate specialist based on regime detection.
    """

    def __init__(self, base_model_dir: str = "models/regime_specialists"):
        self.base_model_dir = Path(base_model_dir)
        self.specialists: Dict[str, RegimeSpecialistLSTM] = {}
        self._load_specialists()

    def _load_specialists(self):
        """Initialize all specialist models."""
        for name in SPECIALIST_CONFIGS.keys():
            specialist = RegimeSpecialistLSTM(name, str(self.base_model_dir))
            self.specialists[name] = specialist

    def build_all(
        self,
        input_shape: Tuple[int, int],
        use_regime_embedding: bool = False,
    ) -> None:
        """Build all specialist models."""
        for specialist in self.specialists.values():
            specialist.build_model(input_shape, use_regime_embedding)

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime_train: np.ndarray,
        regime_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all specialist models with their respective regime filters.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            regime_train: Regime labels for training (str or int)
            regime_val: Regime labels for validation (str or int)
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Dict mapping specialist name to training history
        """
        # Convert string regimes to indices if needed
        if isinstance(regime_train[0], str):
            regime_train = np.array([REGIME_TO_IDX.get(r, Regime.BULL) for r in regime_train])
        if isinstance(regime_val[0], str):
            regime_val = np.array([REGIME_TO_IDX.get(r, Regime.BULL) for r in regime_val])

        histories = {}
        for name, specialist in self.specialists.items():
            try:
                history = specialist.train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    regime_train,
                    regime_val,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                specialist.save_model()
                specialist.save_history()
                histories[name] = history
            except ValueError as e:
                print(f"[{name}] Skipping: {e}")
                histories[name] = None

        return histories

    def predict(
        self,
        X: np.ndarray,
        regime: str,
        regime_confidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Predict using the appropriate specialist based on regime.

        Args:
            X: Input sequences
            regime: Current regime ("BULL", "BEAR", "CRISIS")
            regime_confidence: Optional confidence scores for routing

        Returns:
            Prediction dict with direction, confidence, and specialist used
        """
        if regime_confidence is None:
            regime_confidence = {r: 1.0 if r == regime else 0.0 for r in ["BULL", "BEAR", "CRISIS"]}

        specialist_name = route_to_specialist(regime, regime_confidence)
        specialist = self.specialists[specialist_name]

        if specialist.model is None:
            specialist.load_model()
            if specialist.input_shape:
                specialist.build_model(specialist.input_shape)
                specialist.load_model()

        prediction = specialist.predict(X)
        prediction["specialist_used"] = specialist_name
        prediction["regime"] = regime

        return prediction

    def load_all(self) -> None:
        """Load all specialist models from disk."""
        for name, specialist in self.specialists.items():
            try:
                specialist.load_model()
                print(f"[{name}] Loaded successfully")
            except FileNotFoundError:
                print(f"[{name}] No saved model found")

    def save_all(self) -> None:
        """Save all specialist models."""
        for specialist in self.specialists.values():
            specialist.save_model()


def get_regime_onehot_embedding(regime: str) -> np.ndarray:
    """
    Convert regime label to 3-state one-hot embedding.

    Args:
        regime: "BULL", "BEAR", or "CRISIS"

    Returns:
        One-hot array: [BULL, BEAR, CRISIS]
    """
    embedding = np.zeros(3, dtype=np.float32)
    idx = REGIME_TO_IDX.get(regime.upper(), Regime.BULL)
    embedding[idx] = 1.0
    return embedding


def prepare_regime_training_data(
    X: np.ndarray,
    regime_labels: np.ndarray,
    regime_to_specialist: Dict[str, str] = None,
) -> Dict[str, np.ndarray]:
    """
    Prepare training data splits for each specialist.

    Args:
        X: Input sequences
        regime_labels: Regime label per sample
        regime_to_specialist: Optional mapping override

    Returns:
        Dict mapping specialist name to boolean mask
    """
    if regime_to_specialist is None:
        regime_to_specialist = {
            "BULL": "BULL_JOINT",
            "BEAR": "BEAR_ISOLATED",
            "CRISIS": "CRISIS_ISOLATED",
        }

    masks = {}
    for specialist_name in SPECIALIST_CONFIGS.keys():
        config = SPECIALIST_CONFIGS[specialist_name]
        allowed_regimes = [REGIME_NAMES[r] for r in config.train_regimes]
        mask = np.isin(regime_labels, allowed_regimes)
        masks[specialist_name] = mask

    return masks


if __name__ == "__main__":
    print("=" * 60)
    print("REGIME SPECIALIST MODEL - HARD MODEL ROUTING")
    print("=" * 60)
    print("\nSpecialist configurations:")
    for name, config in SPECIALIST_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Train regimes: {[REGIME_NAMES[r] for r in config.train_regimes]}")
        print(f"  Dropout: {config.dropout_rate}")
        print(f"  Learning rate: {config.learning_rate}")

    print("\n\nRouting examples:")
    test_cases = [
        ("BULL", {"BULL": 0.8, "BEAR": 0.1, "CRISIS": 0.1}),
        ("BEAR", {"BULL": 0.15, "BEAR": 0.75, "CRISIS": 0.1}),
        ("CRISIS", {"BULL": 0.1, "BEAR": 0.15, "CRISIS": 0.75}),
        ("CRISIS", {"BULL": 0.35, "BEAR": 0.1, "CRISIS": 0.55}),  # BULL/CRISIS overlap
        ("BULL", {"BULL": 0.45, "BEAR": 0.3, "CRISIS": 0.25}),  # Low confidence
    ]

    for regime, conf in test_cases:
        routed = route_to_specialist(regime, conf)
        print(f"  Regime={regime}, Conf={conf} → {routed}")

    print("\nOne-hot embeddings:")
    for regime in ["BULL", "BEAR", "CRISIS"]:
        emb = get_regime_onehot_embedding(regime)
        print(f"  {regime}: {emb}")
