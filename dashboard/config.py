"""Shared configuration constants for the dashboard.

Paths are resolved relative to the repository root (two levels up from this file's parent).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SWEEP_RESULTS_DIR = RESULTS_DIR / "sweep"

# Files that /files/<rel> is allowed to serve from.
ALLOWED_STATIC_ROOTS = (RESULTS_DIR, MODELS_DIR)
