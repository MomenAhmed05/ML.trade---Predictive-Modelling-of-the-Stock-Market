"""
GPU Utilities for AMD Radeon RX 7700 XT
=======================================

Detects available GPU backends and provides helpers for:
  - TensorFlow: AMD DirectML (via tensorflow-directml on Windows)
  - XGBoost: OpenCL GPU acceleration (AMD-compatible, no CUDA needed)
  - scikit-learn: cuML support (via rapidsai-cuml on Linux) or joblib fallback
  - ONNX Runtime: DirectML for inference acceleration

Architecture
------------
  - detect_gpu()                  → returns dict of available GPU capabilities
  - print_gpu_report()            → prints formatted GPU availability
  - get_xgboost_device()          → returns {'device': 'gpu'} or {'device': 'cpu'}
  - get_tf_device_name()          → returns '/gpu:0' or '/cpu:0'
  - get_random_forest_config()    → returns cuML-compatible RF config or sklearn fallback
  - XGBoost GPU via OpenCL works on AMD: device='gpu', no CUDA required
"""

from __future__ import annotations

import platform
import warnings
from typing import Dict, Optional, Tuple, Any
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level cache — detects once
# ---------------------------------------------------------------------------

_gpu_info: Optional[Dict[str, Any]] = None


def _detect_amd_gpu_windows() -> Optional[Dict[str, str]]:
    """
    Detect AMD GPU on Windows via WMI.
    Returns dict with name, vendor, vram_mb or None.
    """
    try:
        import subprocess

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Where-Object { $_.Name -match 'AMD|Radeon' } | "
                "Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor | "
                "ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = json.loads(result.stdout)
            # If there's only one GPU, it comes back as a dict not a list
            devices = raw if isinstance(raw, list) else [raw]
            # Prefer dedicated GPU (highest VRAM) over iGPU
            best = None
            best_vram = 0
            for dev in devices:
                if "Radeon" in dev.get("Name", "") or "AMD" in dev.get("Name", ""):
                    ram_bytes = dev.get("AdapterRAM", 0)
                    if ram_bytes > best_vram:
                        best = dev
                        best_vram = ram_bytes
            if best:
                return {
                    "name": best["Name"],
                    "vendor": "AMD",
                    "vram_mb": round(best_vram / (1024 * 1024)) if best_vram else 0,
                    "driver": best.get("DriverVersion", "unknown"),
                }
    except Exception:
        pass
    return None


def _detect_vulkan() -> bool:
    """Check if Vulkan/OpenCL is available on Windows."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-ChildItem 'C:\\Windows\\System32' -Filter 'vulkan-1.dll' -ErrorAction SilentlyContinue",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "vulkan-1.dll" in result.stdout:
            return True
    except Exception:
        pass
    return False


def _check_xgboost_gpu_support() -> Dict[str, Any]:
    """Check if XGBoost can actually use the GPU.

    On AMD Windows: the pip wheel is CUDA-compiled only (USE_CUDA=True).
    Despite device='gpu' not crashing, XGBoost silently falls back to CPU
    with warning: "No visible GPU is found, setting device to CPU."
    We detect this by trying a tiny fit and checking if GPU actually engaged.
    """
    result = {"available": False, "backend": "cpu", "error": None}
    try:
        import xgboost as xgb
        import numpy as np
        from xgboost import XGBClassifier

        # Build info tells us whether CUDA or OpenCL is compiled in
        try:
            info = xgb.build_info()
            use_cuda = info.get("USE_CUDA", False)
            # Note: most Windows wheels have USE_CUDA=True but no OpenCL.
            # This means GPU only works on NVIDIA, not AMD.
        except Exception:
            use_cuda = False

        # Attempt GPU instantiation — may not crash even on AMD
        try:
            dummy = XGBClassifier(n_estimators=1, max_depth=2, device="gpu")
            # Fit a tiny dataset to see if GPU actually engages
            import warnings as _w
            import io

            _buf = io.StringIO()
            try:
                # Suppress stdout to catch only warnings
                import sys as _sys

                _old_stderr = _sys.stderr
                _sys.stderr = _buf
                dummy.fit(
                    np.random.randn(4, 2).astype(np.float32), np.random.randint(0, 2, 4)
                )
                _sys.stderr = _old_stderr
                warnings_log = _buf.getvalue()
                if (
                    "No visible GPU" in warnings_log
                    or "Device is changed from GPU to CPU" in warnings_log
                ):
                    result["backend"] = "cpu (GPU not usable on AMD Windows)"
                else:
                    result["available"] = True
                    result["backend"] = "cuda" if use_cuda else "opencl"
            except Exception:
                _sys.stderr = _old_stderr
                result["available"] = True
                result["backend"] = "cuda" if use_cuda else "opencl"
            del dummy
        except Exception as exc:
            result["error"] = str(exc)
            result["backend"] = "cpu"
    except ImportError:
        result["error"] = "xgboost not installed"
    return result


def _check_tf_gpu_support() -> Dict[str, Any]:
    """Check TensorFlow GPU capabilities."""
    result = {"available": False, "devices": [], "backend": None}
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        result["devices"] = [g.name for g in gpus]
        if gpus:
            result["available"] = True
            result["backend"] = "cuda"  # Only works on NVIDIA
        else:
            result["backend"] = "cpu"
    except ImportError:
        result["backend"] = "not_installed"
    return result


def _check_cuml_support() -> Dict[str, Any]:
    """Check if RAPIDS cuML is available (NVIDIA only, but check anyway)."""
    result = {"available": False, "error": None}
    try:
        import cuml

        result["available"] = True
    except ImportError:
        result["error"] = "cuml not installed"
    return result


def _check_onnx_directml() -> Dict[str, Any]:
    """Check ONNX Runtime DirectML availability for AMD GPU inference."""
    result = {"available": False, "error": None}
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        result["providers"] = providers
        if "DmlExecutionProvider" in providers:
            result["available"] = True
    except ImportError:
        result["error"] = "onnxruntime or onnxruntime-directml not installed"
    return result


def detect_gpu(force: bool = False) -> Dict[str, Any]:
    """
    Full GPU detection. Cached after first call unless force=True.

    Returns dict with keys:
      - amd_gpu       : dict or None  — GPU hardware info
      - vulkan        : bool          — Vulkan runtime available
      - xgboost       : dict          — XGBoost GPU support
      - tensorflow    : dict          — TF GPU support
      - cuml          : dict          — cuML/RAPIDS availability
      - onnx_directml : dict          — ONNX DirectML availability
      - summary       : str           — one-line summary
    """
    global _gpu_info
    if _gpu_info is not None and not force:
        return _gpu_info

    os_name = platform.system()

    amd_gpu = _detect_amd_gpu_windows() if os_name == "Windows" else None
    vulkan = _detect_vulkan() if os_name == "Windows" else False

    _gpu_info = {
        "amd_gpu": amd_gpu,
        "vulkan": vulkan,
        "os": os_name,
        "xgboost": _check_xgboost_gpu_support(),
        "tensorflow": _check_tf_gpu_support(),
        "cuml": _check_cuml_support(),
        "onnx_directml": _check_onnx_directml(),
    }

    # Build summary
    parts = []
    if amd_gpu:
        vram = amd_gpu.get("vram_mb", 0)
        parts.append(f"AMD GPU: {amd_gpu['name']} ({vram}MB)")
    else:
        parts.append("GPU: NOT DETECTED")

    if _gpu_info["xgboost"]["available"]:
        parts.append("XGBoost: GPU (OpenCL)")
    else:
        parts.append("XGBoost: CPU")

    if _gpu_info["tensorflow"]["available"]:
        parts.append("TensorFlow: GPU")
    else:
        parts.append("TensorFlow: CPU")

    if _gpu_info["cuml"]["available"]:
        parts.append("cuML: available")
    else:
        parts.append("cuML: N/A")

    if _gpu_info["onnx_directml"]["available"]:
        parts.append("ONNX/DirectML: ready")
    else:
        parts.append("ONNX/DirectML: N/A")

    _gpu_info["summary"] = " | ".join(parts)
    return _gpu_info


def print_gpu_report() -> None:
    """Print a formatted GPU capability report to stdout."""
    info = detect_gpu()

    header = "=" * 60
    print(f"\n{header}")
    print("  GPU CAPABILITY REPORT")
    print(f"{header}")

    print(f"\n  [Hardware]")
    amd = info["amd_gpu"]
    if amd:
        print(f"    GPU      : {amd['name']}")
        print(f"    VRAM     : {amd['vram_mb']} MB (WMI — may under-report)")
        print(f"    Driver   : {amd['driver']}")
        print(f"    Vendor   : {amd['vendor']} (no CUDA — uses OpenCL/DirectML)")
    else:
        print(f"    GPU      : NOT FOUND")
    print(f"    OS       : {info['os']}")
    print(f"    Vulkan   : {'YES' if info['vulkan'] else 'NO'}")

    print(f"\n  [Backends]")
    xgb = info["xgboost"]
    xgb_status = "GPU (CUDA)" if xgb["available"] else "CPU (multi-core)"
    print(f"    XGBoost         : {xgb_status}")
    if not xgb["available"] and xgb.get("error"):
        print(f"      -> Pip wheel is CUDA-only (no OpenCL for AMD)")
    tf = info["tensorflow"]
    print(
        f"    TensorFlow      : {tf['backend'].upper()}  devices={tf['devices'] or '[]'}"
    )
    if tf["backend"] == "cpu" or tf["backend"] == "not_installed":
        print(f"      -> For AMD GPU TF, use tensorflow-directml (experimental)")
    cuml = info["cuml"]
    print(
        f"    cuML/RAPIDS     : {'AVAILABLE' if cuml['available'] else 'N/A (NVIDIA only)'}"
    )
    onnx = info["onnx_directml"]
    onnx_status = "READY (for LARGE models)" if onnx["available"] else "NOT AVAILABLE"
    print(f"    ONNX DirectML   : {onnx_status}")
    if onnx["available"]:
        print(f"      -> Small models (LSTM<128): CPU is often faster")

    print(f"\n  [Recommendation]")
    print(f"    Best available: CPU multi-core (sklearn n_jobs=-1, xgb n_jobs=-1)")
    print(
        f"    For TF models:  XLA JIT is already enabled (tf.config.optimizer.set_jit)"
    )
    print(f"    ONNX DirectML:  Available for large-model inference if needed")

    print(f"\n  [Summary]")
    print(f"    {info['summary']}")
    print(f"{header}\n")


def get_xgboost_kwargs() -> Dict[str, Any]:
    """
    Returns optimal XGBoost parameters.

    On AMD Windows: pip wheels are CUDA-compiled only (no OpenCL).
    GPU='gpu' silently falls back to CPU with a warning.
    We use device='cpu' + tree_method='hist' + n_jobs=-1 (all cores).
    This is the practical best for AMD GPUs on Windows.

    On NVIDIA/Linux with CUDA: would use device='cuda' for real GPU.
    """
    # Real-world check: XGBoost 3.x Windows wheels are CUDA-only.
    # On AMD, 'gpu' silently falls back to CPU. Use CPU with all cores.
    return {
        "device": "cpu",
        "tree_method": "hist",
        "n_jobs": -1,
    }


def get_random_forest_kwargs() -> Dict[str, Any]:
    """
    Returns optimal RandomForest parameters.

    cuML (RAPIDS) GPU RF is NVIDIA-only, so on AMD we use scikit-learn
    with all CPU cores. This is still fast for our dataset sizes.
    """
    info = detect_gpu()

    if info["cuml"]["available"]:
        # cuML RF is GPU-accelerated (NVIDIA only)
        return {"model_type": "cuml", "n_estimators": 150, "max_depth": 5}
    else:
        # scikit-learn with all cores
        return {
            "model_type": "sklearn",
            "n_estimators": 150,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "random_state": 42,
            "n_jobs": -1,
        }


def get_tf_device_name() -> str:
    """Returns TensorFlow device string: '/gpu:0' or '/cpu:0'."""
    info = detect_gpu()
    if info["tensorflow"]["available"]:
        return "/gpu:0"
    return "/cpu:0"


def get_onnx_inference_options() -> Optional[Dict[str, Any]]:
    """
    Return ONNX Runtime session options for DirectML inference.
    Returns None if DirectML is not available.
    """
    info = detect_gpu()
    if info["onnx_directml"]["available"]:
        return {
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {"device_id": "0", "performance_preference": "high_performance"},
                {},
            ],
        }
    return None


def gpu_bar(label: str = "GPU DETECTION", enabled: bool = True) -> None:
    """Print a decorative GPU status line."""
    if enabled:
        info = detect_gpu()
        print(f"  [{label}] {info['summary']}")


if __name__ == "__main__":
    print_gpu_report()
