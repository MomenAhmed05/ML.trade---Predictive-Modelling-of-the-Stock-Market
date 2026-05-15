"""Derived statistics built on top of data_service loaders."""
from __future__ import annotations

from collections import Counter

from . import data_service


def survivor_count() -> tuple[int, int]:
    """Return (selected_count, total_count) from the tournament table."""
    df = data_service.load_tournament()
    if df.empty or "selected" not in df.columns:
        return (0, 0)
    total = len(df)
    selected = int((df["selected"].astype(str).str.upper() == "Y").sum())
    return (selected, total)


def regime_distribution() -> dict[str, float]:
    """Percent of each regime label (BULL / BEAR / CRISIS)."""
    regimes = data_service.load_regime()
    if not regimes:
        return {}
    counts = Counter(regimes)
    n = len(regimes)
    return {label: round(100.0 * c / n, 2) for label, c in counts.items()}


def current_regime() -> str:
    """Last regime observation, or 'UNKNOWN' if none."""
    regimes = data_service.load_regime()
    if not regimes:
        return "UNKNOWN"
    return str(regimes[-1])


def portfolio_summary() -> dict:
    """Averages across surviving entries in the master manifest."""
    manifest = data_service.load_manifest()
    if not manifest:
        return {"avg_val_sharpe": 0.0, "avg_val_accuracy": 0.0, "total_trades": 0, "survivors": 0}

    # A survivor is any manifest entry — the manifest is already filtered by
    # the tournament; we guard against missing fields.
    def _num(entry: dict, key: str) -> float:
        v = entry.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    sharpes = [_num(e, "val_sharpe") for e in manifest]
    accs = [_num(e, "val_accuracy") for e in manifest]
    trades = [int(_num(e, "val_trades")) for e in manifest]
    n = len(manifest)
    return {
        "avg_val_sharpe": round(sum(sharpes) / n, 4) if n else 0.0,
        "avg_val_accuracy": round(sum(accs) / n, 4) if n else 0.0,
        "total_trades": int(sum(trades)),
        "survivors": n,
    }
