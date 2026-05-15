"""
ONNX DirectML Bridge — GPU Inference for Keras/TF Models on AMD
================================================================

Exports trained Keras LSTM models to ONNX, then accelerates inference
using ONNX Runtime with the DirectML execution provider (AMD GPU native).

Usage:
    from onnx_directml import ONNXInferenceAdapter

    # Wrap any trained Keras model
    adapter = ONNXInferenceAdapter(lstm.model, input_shape=(24, n_features))
    adapter.export_and_warmup()

    # Predict with GPU acceleration
    probs = adapter.predict(X_test)  # np.ndarray → np.ndarray

    # Check if GPU is active
    print(adapter.status())

Design notes:
  - Falls back gracefully to Keras CPU inference if DirectML unavailable
  - Caches ONNX models on disk for instant reload
  - Uses float32 for best DirectML compatibility
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

# Lazy imports — DirectML is optional
_ORT_AVAILABLE = False
_DML_AVAILABLE = False
_ORT = None

try:
    import onnxruntime as _ORT

    _ORT_AVAILABLE = True
    if "DmlExecutionProvider" in _ORT.get_available_providers():
        _DML_AVAILABLE = True
except ImportError:
    pass


def _get_ort_session_options() -> Dict[str, Any]:
    """Build ONNX Runtime session options for DirectML on AMD GPU."""
    if not _DML_AVAILABLE:
        return {"providers": ["CPUExecutionProvider"]}

    provider_options = [
        {
            "device_id": "0",
            "performance_preference": "high_performance",
            # Disable memory arena to prevent OOM on 10GB VRAM
            "enable_memory_arena": "1",
        }
    ]

    return {
        "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "provider_options": provider_options,
    }


class ONNXInferenceAdapter:
    """
    Export Keras → ONNX → DirectML inference.

    Parameters
    ----------
    keras_model : tf.keras.Model
        Trained Keras model to export.
    input_shape : tuple
        (lookback, n_features) — input shape without batch dim.
    model_name : str
        Name for ONNX disk cache.
    cache_dir : str
        Directory to store exported ONNX models.
    force_cpu : bool
        Skip DirectML even if available.
    """

    def __init__(
        self,
        keras_model,
        input_shape: tuple,
        model_name: str = "lstm_model",
        cache_dir: str = "models/onnx_cache",
        force_cpu: bool = False,
    ):
        self.keras_model = keras_model
        self.input_shape = input_shape  # (lookback, n_features)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_cpu = force_cpu

        self._onnx_path = self.cache_dir / f"{model_name}.onnx"
        self._session: Optional[Any] = None  # onnxruntime.InferenceSession
        self._gpu_active = False
        self._ready = False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_and_warmup(self, force_rebuild: bool = False) -> bool:
        """
        Export Keras model to ONNX and create inference session.
        Returns True if GPU (DirectML) is active.
        """
        if not _ORT_AVAILABLE:
            print("[ONNX/DML] onnxruntime not installed — using Keras CPU inference")
            return False

        if self._session is not None and not force_rebuild:
            return self._gpu_active

        # Build or load ONNX
        if not self._onnx_path.exists() or force_rebuild:
            ok = self._export_keras_to_onnx()
            if not ok:
                return False

        # Create inference session
        return self._create_session()

    def _export_keras_to_onnx(self) -> bool:
        """Internal: Keras → ONNX conversion."""
        try:
            import tf2onnx
            import tensorflow as tf

            print(f"[ONNX/DML] Exporting {self.model_name} to ONNX...")
            spec = (
                tf.TensorSpec((None,) + self.input_shape, tf.float32, name="input"),
            )
            model_proto, _ = tf2onnx.convert.from_keras(
                self.keras_model,
                input_signature=spec,
                opset=13,
                output_path=str(self._onnx_path),
            )
            print(f"[ONNX/DML] Saved to {self._onnx_path}")
            return True

        except ImportError:
            print("[ONNX/DML] tf2onnx not installed — run: pip install tf2onnx")
            return False
        except Exception as e:
            print(f"[ONNX/DML] Export failed: {e}")
            return False

    def _create_session(self) -> bool:
        """Internal: create ONNX Runtime session."""
        if not self._onnx_path.exists():
            print(f"[ONNX/DML] ONNX file not found: {self._onnx_path}")
            return False

        options = _get_ort_session_options()
        use_dml = not self.force_cpu and _DML_AVAILABLE

        try:
            self._session = _ORT.InferenceSession(
                str(self._onnx_path),
                providers=options["providers"],
                provider_options=options.get("provider_options"),
            )
            actual_providers = self._session.get_providers()
            self._gpu_active = "DmlExecutionProvider" in actual_providers

            provider_str = "DirectML (GPU)" if self._gpu_active else "CPU"
            print(f"[ONNX/DML] Session ready — provider: {provider_str}")

            # Warmup with dummy data
            dummy = np.zeros((1,) + self.input_shape, dtype=np.float32)
            _ = self._session.run(None, {"input": dummy})
            self._ready = True
            return True

        except Exception as e:
            if use_dml:
                print(f"[ONNX/DML] DirectML failed ({e}), falling back to CPU...")
                # Retry with CPU only
                try:
                    self._session = _ORT.InferenceSession(
                        str(self._onnx_path),
                        providers=["CPUExecutionProvider"],
                    )
                    self._gpu_active = False
                    self._ready = True
                    return True
                except Exception as e2:
                    print(f"[ONNX/DML] CPU fallback also failed: {e2}")
            else:
                print(f"[ONNX/DML] Session creation failed: {e}")

        self._session = None
        return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Run inference. Falls back to Keras if ONNX isn't ready.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, lookback, n_features), float32.
        batch_size : int
            Batch size for ONNX inference.

        Returns
        -------
        probs : np.ndarray
            Shape (n_samples, 2) — softmax probabilities [DOWN, UP].
        """
        # Fallback to Keras
        if not self._ready or self._session is None:
            if self.keras_model is not None:
                return self.keras_model.predict(X, verbose=0)
            raise RuntimeError(
                "No inference backend available (neither ONNX nor Keras)."
            )

        # Ensure float32
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        # Batched ONNX inference
        input_name = self._session.get_inputs()[0].name
        n_samples = len(X)
        results = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X[start:end]
            output = self._session.run(None, {input_name: batch})
            results.append(output[0])

        return np.concatenate(results, axis=0)

    def predict_with_confidence(
        self, X: np.ndarray, batch_size: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Run inference returning direction + confidence dict (same schema as LSTMModel.predict()).
        """
        probs = self.predict(X, batch_size=batch_size)
        predicted_class = np.argmax(probs, axis=1)
        return {
            "direction_probabilities": probs,
            "direction": np.where(predicted_class == 0, "DOWN", "UP"),
            "direction_confidence": np.max(probs, axis=1),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return status dict for logging."""
        return {
            "model_name": self.model_name,
            "onnx_available": _ORT_AVAILABLE,
            "dml_available": _DML_AVAILABLE,
            "gpu_active": self._gpu_active,
            "ready": self._ready,
            "onnx_path": str(self._onnx_path) if self._onnx_path.exists() else None,
        }

    def is_gpu_active(self) -> bool:
        return self._gpu_active and self._ready


def install_instructions():
    """Print install instructions for ONNX DirectML stack."""
    print("""
    To enable ONNX DirectML GPU inference on AMD:

    1. Install ONNX Runtime with DirectML:
       pip install onnxruntime-directml

    2. Install tf2onnx for model conversion:
       pip install tf2onnx

    3. Verify:
       python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
       # Should show: ['DmlExecutionProvider', 'CPUExecutionProvider']

    Then use ONNXInferenceAdapter to wrap your Keras models.
    """)


if __name__ == "__main__":
    info = _get_ort_session_options()
    print(f"Providers: {info['providers']}")
    print(f"DML available: {_DML_AVAILABLE}")
    print(f"ORT available: {_ORT_AVAILABLE}")
    if not _DML_AVAILABLE:
        install_instructions()
