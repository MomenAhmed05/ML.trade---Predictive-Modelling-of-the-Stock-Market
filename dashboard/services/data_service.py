"""Disk-backed data loaders for the dashboard.

Every loader is defensive — a missing/malformed file returns an empty container,
never raises. Other dashboard units depend on these exact signatures.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import (
    DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SWEEP_RESULTS_DIR,
)

# Columns produced by the tournament stage — kept here so an empty frame still
# carries the right schema for downstream code.
TOURNAMENT_COLUMNS = [
    "group_id", "stocks", "val_accuracy", "val_sharpe",
    "overall_accuracy", "acc_55", "trades_55", "acc_60", "trades_60",
    "acc_65", "trades_65", "ticker_positive_ratio", "selected",
]

OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_tournament() -> pd.DataFrame:
    """Load results/group_tournament.csv, or an empty frame with the right columns."""
    path = RESULTS_DIR / "group_tournament.csv"
    if not path.exists():
        return pd.DataFrame(columns=TOURNAMENT_COLUMNS)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=TOURNAMENT_COLUMNS)


def load_manifest() -> list[dict]:
    """Load models/master_manifest.json. Returns [] if missing/malformed."""
    path = MODELS_DIR / "master_manifest.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_regime() -> list[str]:
    """Load results/portfolio_regime.npy as a list of strings."""
    path = RESULTS_DIR / "portfolio_regime.npy"
    if not path.exists():
        return []
    try:
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.tolist()]
    except Exception:
        return []


_EQUITY_RE = re.compile(r"^equity_curve_(?P<mode>.+)\.png$", re.IGNORECASE)


def _equity_curve_entry(png: Path) -> dict:
    m = _EQUITY_RE.match(png.name)
    mode = m.group("mode") if m else png.stem
    try:
        rel = str(png.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(png)
    return {"name": png.name, "path": rel, "mode": mode}


def list_equity_curves() -> list[dict]:
    """Scan results/ (non-recursive) for equity_curve_*.png files."""
    if not RESULTS_DIR.exists():
        return []
    return [
        _equity_curve_entry(p)
        for p in sorted(RESULTS_DIR.glob("equity_curve_*.png"))
    ]


def load_training_history(group_id: int) -> Optional[dict]:
    """Unpickle models/group_<id>/training_history.pkl. None if missing."""
    path = MODELS_DIR / f"group_{group_id}" / "training_history.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_group_tickers(group_id: int) -> list[str]:
    """Load models/group_<id>/selected_tickers.json."""
    path = MODELS_DIR / f"group_{group_id}" / "selected_tickers.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(t) for t in data] if isinstance(data, list) else []
    except Exception:
        return []


_LETTER_GROUP_RE = re.compile(r"^us3000_tickers_(?P<suffix>[^_]+)_1hour$")


def list_ticker_letter_groups() -> list[str]:
    """Return suffixes of data/raw/us3000_tickers_*_1hour/ dirs (e.g. 'A-B')."""
    if not RAW_DATA_DIR.exists():
        return []
    out: list[str] = []
    for child in sorted(RAW_DATA_DIR.iterdir()):
        if not child.is_dir():
            continue
        m = _LETTER_GROUP_RE.match(child.name)
        if m:
            out.append(m.group("suffix"))
    return out


def _letter_group_dir(letter_group: str) -> Path:
    return RAW_DATA_DIR / f"us3000_tickers_{letter_group}_1hour"


def list_tickers_in_letter_group(letter_group: str) -> list[str]:
    """List ticker symbols available in a letter-group dir."""
    d = _letter_group_dir(letter_group)
    if not d.exists():
        return []
    suffix = "_1hour.txt"
    out: list[str] = []
    for p in sorted(d.glob(f"*{suffix}")):
        name = p.name
        if name.endswith(suffix):
            out.append(name[: -len(suffix)])
    return out


def load_ohlcv(letter_group: str, ticker: str, limit: int = 500) -> pd.DataFrame:
    """Load the raw hourly bars for a ticker; returns last `limit` rows."""
    path = _letter_group_dir(letter_group) / f"{ticker}_1hour.txt"
    if not path.exists():
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    try:
        # Raw files are CSVs without a header row in most exports; handle both.
        df = pd.read_csv(path, header=None, names=OHLCV_COLUMNS)
        # If the first row parsed as a string header (open == "open"), reload.
        if str(df.iloc[0].get("open", "")).lower() == "open":
            df = pd.read_csv(path)
            df.columns = OHLCV_COLUMNS[: len(df.columns)]
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    if limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def list_sweep_equity_curves() -> list[dict]:
    """Scan results/sweep/ for equity_curve_*.png files. [] if absent."""
    if not SWEEP_RESULTS_DIR.exists():
        return []
    return [
        _equity_curve_entry(p)
        for p in sorted(SWEEP_RESULTS_DIR.glob("equity_curve_*.png"))
    ]


def load_backtest_metrics() -> dict:
    """Load results/backtest_metrics.json — the primary portfolio backtest results.

    Returns {} if missing/malformed.
    """
    path = RESULTS_DIR / "backtest_metrics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_sweep_metrics() -> list[dict]:
    """Load all sweep metrics JSONs from results/sweep/metrics_*.json."""
    if not SWEEP_RESULTS_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(SWEEP_RESULTS_DIR.glob("metrics_*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Inject the config name from the filename
                name = p.stem  # e.g. "metrics_aggressive" or "metrics_baseline"
                config = name.replace("metrics_", "", 1)
                data["config"] = config
                out.append(data)
        except Exception:
            continue
    return out


def load_comparison() -> Optional[pd.DataFrame]:
    """Read results/comparison.csv if it exists; otherwise None."""
    path = RESULTS_DIR / "comparison.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
