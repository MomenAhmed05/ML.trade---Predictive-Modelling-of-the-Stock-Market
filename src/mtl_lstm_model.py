"""
Shared / Private Multi-Task Learning LSTM
==========================================
Architecture
------------
  Shared trunk  : LSTM(32)  -> Dropout(0.3)
  Private heads : one Dense(2, softmax) per group_id

The shared trunk learns universal market-structure patterns across all
groups simultaneously. Each group's private head specialises on its own
correlation-cluster. The combined loss is the mean of all per-group
cross-entropy terms, so every group contributes equal gradient to the
shared body.

Usage
-----
  # build once, pass list of group ids
  mtl = MTLLSTMModel(group_ids=[0, 1, 2])
  mtl.build_model(input_shape=(24, n_features))

  # train on all groups at once
  mtl.train(group_data)   # dict: {group_id: (X_train, y_train, X_val, y_val)}

  # save / load
  mtl.save("models/mtl")
  mtl.load("models/mtl", group_ids=[0, 1, 2], input_shape=(24, n_features))

  # predict for a specific group
  preds = mtl.predict(group_id=0, X=X_test)
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, Callback

tf.config.optimizer.set_jit(True)


# ---------------------------------------------------------------------------
# Composite checkpoint (same logic as in lstm_model.py)
# ---------------------------------------------------------------------------

class _CompositeCheckpoint(Callback):
    """Save best shared-trunk weights by composite score across all heads."""

    def __init__(self, filepath: str, alpha: float = 0.5):
        super().__init__()
        self.filepath   = filepath
        self.alpha      = alpha
        self.best_score = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs     = logs or {}
        val_acc  = logs.get("val_loss")       # proxy: use composite val_loss
        val_loss = logs.get("val_loss", 1.0)
        # For MTL we track val_loss only (lower = better)
        score = -val_loss
        if score > self.best_score:
            self.best_score = score
            self.model.save_weights(self.filepath)
            print(f"  [MTL] Epoch {epoch+1}: improved val_loss → saving weights.")


# ---------------------------------------------------------------------------
# MTL model
# ---------------------------------------------------------------------------

class MTLLSTMModel:
    """
    Multi-Task LSTM with one shared trunk and N private classification heads.

    Parameters
    ----------
    group_ids : list[int]
        The group IDs that will have private heads.
    shared_units : int
        Size of the shared LSTM layer.
    dropout_rate : float
        Dropout applied after the shared LSTM.
    """

    def __init__(
        self,
        group_ids: List[int],
        shared_units: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.group_ids    = list(group_ids)
        self.shared_units = shared_units
        self.dropout_rate = dropout_rate
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict[str, list]] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_model(
        self,
        input_shape: Tuple[int, int],
        learning_rate: float = 1e-3,
    ) -> keras.Model:
        """
        Construct the shared-trunk + private-heads Keras model.

        The model has:
          - one input  : (batch, lookback, features)
          - N outputs  : one softmax(2) per group in self.group_ids
        """
        print("\n" + "=" * 60)
        print("BUILDING SHARED/PRIVATE MTL LSTM")
        print(f"  groups        : {self.group_ids}")
        print(f"  shared_units  : {self.shared_units}")
        print(f"  dropout_rate  : {self.dropout_rate}")
        print(f"  input_shape   : {input_shape}")
        print("=" * 60)

        inp   = layers.Input(shape=input_shape, name="shared_input")
        x     = layers.LSTM(self.shared_units, name="shared_lstm")(inp)
        x     = layers.Dropout(self.dropout_rate, name="shared_dropout")(x)

        outputs = {}
        for gid in self.group_ids:
            head = layers.Dense(
                2,
                activation="softmax",
                name=f"head_group_{gid}",
            )(x)
            outputs[f"head_group_{gid}"] = head

        self.model = keras.Model(
            inputs=inp,
            outputs=outputs,
            name="mtl_lstm_stock_predictor",
        )

        # Equal-weight loss per head
        loss_dict   = {f"head_group_{gid}": "categorical_crossentropy" for gid in self.group_ids}
        loss_weights = {f"head_group_{gid}": 1.0 for gid in self.group_ids}
        metrics_dict = {f"head_group_{gid}": ["accuracy"]  for gid in self.group_ids}

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            loss_weights=loss_weights,
            metrics=metrics_dict,
            steps_per_execution=32,  # batch dispatch — bit-identical, reduces py/GPU overhead
        )

        self.model.summary()
        return self.model

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        group_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        save_dir: str = "models/mtl",
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 5,
        class_weights: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> Dict[str, list]:
        """
        Train the MTL model.

        Parameters
        ----------
        group_data : dict
            {group_id: (X_train, y_train_onehot, X_val, y_val_onehot)}
            All arrays must share the SAME number of rows (pad or truncate
            to the shortest group if sizes differ).
        save_dir : str
            Directory to store weights + history.
        class_weights : dict, optional
            {group_id: {0: w0, 1: w1}} - per-group class weights.
            Keras MTL does not natively support per-output class weights,
            so we apply them via sample_weight arrays.
        """
        if self.model is None:
            raise ValueError("Call build_model() first.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        weights_file = str(save_path / "mtl_model.weights.h5")

        # ---- Align lengths across groups ----------------------------------
        gids = [gid for gid in self.group_ids if gid in group_data]
        if not gids:
            raise ValueError("group_data contains no matching group_ids.")

        min_train = min(len(group_data[g][0]) for g in gids)
        min_val   = min(len(group_data[g][2]) for g in gids)

        X_train = group_data[gids[0]][0][:min_train]   # shared input
        X_val   = group_data[gids[0]][2][:min_val]

        y_train_dict: Dict[str, np.ndarray] = {}
        y_val_dict:   Dict[str, np.ndarray] = {}
        sw_train_dict: Dict[str, np.ndarray] = {}

        for gid in gids:
            key = f"head_group_{gid}"
            y_tr = group_data[gid][1][:min_train]
            y_vl = group_data[gid][3][:min_val]
            y_train_dict[key] = y_tr
            y_val_dict[key]   = y_vl

            # Build sample weights from class_weights if provided
            if class_weights and gid in class_weights:
                cw    = class_weights[gid]
                y_1d  = np.argmax(y_tr, axis=1)
                sw    = np.where(y_1d == 0, cw.get(0, 1.0), cw.get(1, 1.0)).astype(np.float32)
                sw_train_dict[key] = sw

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=False,
                verbose=1,
            ),
            _CompositeCheckpoint(filepath=weights_file),
        ]

        print(f"\n[MTL] Training on {min_train} samples | val on {min_val} samples")
        history = self.model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            sample_weight=sw_train_dict if sw_train_dict else None,
            callbacks=callbacks,
            verbose=1,
        )

        # Reload best weights
        try:
            self.model.load_weights(weights_file)
            print(f"[MTL] Best weights restored from {weights_file}")
        except Exception as e:
            print(f"[MTL] Warning: could not reload best weights: {e}")

        self.history = history.history

        # Save history
        hist_path = save_path / "mtl_history.pkl"
        with open(hist_path, "wb") as f:
            pickle.dump(self.history, f)
        print(f"[MTL] History saved to {hist_path}")

        gc.collect()
        return self.history

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, group_id: int, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference for a specific group head.

        Returns the same dict schema as LSTMModel.predict() for drop-in
        compatibility with EnsembleModel.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load().")
        if group_id not in self.group_ids:
            raise ValueError(f"group_id={group_id} not in {self.group_ids}")

        all_outputs = self.model.predict(X, batch_size=1024, verbose=0)

        # Keras returns a list (or dict) when there are multiple outputs
        key = f"head_group_{group_id}"
        if isinstance(all_outputs, dict):
            probs = all_outputs[key]
        else:
            idx   = self.group_ids.index(group_id)
            probs = all_outputs[idx]

        pred_class = np.argmax(probs, axis=1)
        return {
            "direction_probabilities": probs,
            "direction":               np.where(pred_class == 0, "DOWN", "UP"),
            "direction_confidence":    np.max(probs, axis=1),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        """Save weights + metadata to save_dir."""
        if self.model is None:
            raise ValueError("No model to save.")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(save_path / "mtl_model.weights.h5"))
        meta = {
            "group_ids":    self.group_ids,
            "shared_units": self.shared_units,
            "dropout_rate": self.dropout_rate,
        }
        with open(save_path / "mtl_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[MTL] Saved to {save_path}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        input_shape: Tuple[int, int],
        group_ids: Optional[List[int]] = None,
    ) -> "MTLLSTMModel":
        """
        Load a saved MTL model from save_dir.

        Parameters
        ----------
        save_dir    : directory produced by .save()
        input_shape : (lookback, n_features) needed to rebuild the graph
        group_ids   : override group list (must match what was saved)
        """
        save_path = Path(save_dir)
        meta_file = save_path / "mtl_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if group_ids is None:
                group_ids = meta["group_ids"]
            shared_units  = meta.get("shared_units", 32)
            dropout_rate  = meta.get("dropout_rate", 0.3)
        else:
            if group_ids is None:
                raise ValueError("group_ids required when mtl_meta.json is missing.")
            shared_units = 32
            dropout_rate = 0.3

        obj = cls(
            group_ids=group_ids,
            shared_units=shared_units,
            dropout_rate=dropout_rate,
        )
        obj.build_model(input_shape=input_shape)
        obj.model.load_weights(str(save_path / "mtl_model.weights.h5"))
        print(f"[MTL] Loaded from {save_path}")
        return obj


# ---------------------------------------------------------------------------
# Convenience: wrap a single MTL head as a duck-typed LSTMModel substitute
# so EnsembleModel can use it unchanged.
# ---------------------------------------------------------------------------

class MTLGroupAdapter:
    """
    Wraps one group-head of an MTLLSTMModel behind the same interface
    as LSTMModel so that EnsembleModel requires zero changes.
    """

    def __init__(self, mtl_model: MTLLSTMModel, group_id: int):
        self.mtl_model = mtl_model
        self.group_id  = group_id
        # expose .model so EnsembleModel's None-check passes
        self.model     = mtl_model.model

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return self.mtl_model.predict(self.group_id, X)
