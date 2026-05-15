"""Tournament leaderboard and per-group detail routes."""
from __future__ import annotations

from typing import Any

import math

import numpy as np
from flask import Blueprint, render_template

from ...services.data_service import (
    load_group_tickers,
    load_manifest,
    load_tournament,
    load_training_history,
)

bp = Blueprint(
    "tournament",
    __name__,
    url_prefix="/tournament",
    template_folder="../../templates",
)


def _clean_number(value: Any) -> Any:
    """Turn NaN/inf into None so Jinja can safely render them.

    Also coerces string values from CSV into floats so that
    Jinja2 format filters (e.g. '%.4f') work correctly.
    """
    # Coerce strings that look like numbers into actual floats
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except (ValueError, AttributeError):
            return None  # non-numeric string
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _split_stocks(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    text = str(raw)
    return [t.strip() for t in text.split(",") if t.strip()]


def _row_to_dict(row: dict) -> dict:
    tickers = _split_stocks(row.get("stocks"))
    selected_raw = row.get("selected")
    selected_flag = str(selected_raw).strip().upper() == "Y" if selected_raw is not None else False
    return {
        "group_id": int(row.get("group_id", 0) or 0),
        "stocks_raw": row.get("stocks", ""),
        "tickers": tickers,
        "val_accuracy": _clean_number(row.get("val_accuracy")),
        "val_sharpe": _clean_number(row.get("val_sharpe")),
        "overall_accuracy": _clean_number(row.get("overall_accuracy")),
        "acc_55": _clean_number(row.get("acc_55")),
        "trades_55": _clean_number(row.get("trades_55")),
        "acc_60": _clean_number(row.get("acc_60")),
        "trades_60": _clean_number(row.get("trades_60")),
        "acc_65": _clean_number(row.get("acc_65")),
        "trades_65": _clean_number(row.get("trades_65")),
        "ticker_positive_ratio": _clean_number(row.get("ticker_positive_ratio")),
        "selected": selected_flag,
    }


@bp.route("/")
def index():
    df = load_tournament()
    rows: list[dict] = []
    if df is not None and not df.empty:
        rows = [_row_to_dict(r) for r in df.to_dict(orient="records")]

    selected_count = sum(1 for r in rows if r["selected"])
    total_count = len(rows)

    return render_template(
        "tournament/index.html",
        rows=rows,
        selected_count=selected_count,
        total_count=total_count,
    )


@bp.route("/<int:group_id>")
def detail(group_id: int):
    df = load_tournament()
    tournament_row: dict | None = None
    if df is not None and not df.empty:
        match = df[df["group_id"] == group_id]
        if not match.empty:
            tournament_row = _row_to_dict(match.iloc[0].to_dict())

    manifest_entry: dict | None = None
    for entry in load_manifest():
        if isinstance(entry, dict) and int(entry.get("group_id", -1)) == group_id:
            manifest_entry = entry
            break

    tickers = load_group_tickers(group_id)
    if not tickers and tournament_row:
        tickers = tournament_row["tickers"]
    if not tickers and manifest_entry:
        tickers = [str(t) for t in manifest_entry.get("stocks", [])]

    history = load_training_history(group_id)
    history_payload: dict | None = None
    if isinstance(history, dict):
        def _series(key: str) -> list[float]:
            values = history.get(key, [])
            out: list[float] = []
            try:
                for v in values:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        continue
                    out.append(f)
            except Exception:
                return []
            return out

        history_payload = {
            "accuracy": _series("accuracy"),
            "val_accuracy": _series("val_accuracy"),
            "loss": _series("loss"),
            "val_loss": _series("val_loss"),
        }

    not_found = tournament_row is None and manifest_entry is None and not tickers and history_payload is None

    window_sharpes: list[float] = []
    if manifest_entry and isinstance(manifest_entry.get("window_sharpes"), list):
        for v in manifest_entry["window_sharpes"]:
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    continue
                window_sharpes.append(f)
            except Exception:
                continue

    window_trades: list[Any] = []
    if manifest_entry and isinstance(manifest_entry.get("window_trades"), list):
        window_trades = list(manifest_entry["window_trades"])

    return render_template(
        "tournament/detail.html",
        group_id=group_id,
        not_found=not_found,
        tickers=tickers,
        row=tournament_row,
        manifest=manifest_entry,
        history=history_payload,
        window_sharpes=window_sharpes,
        window_trades=window_trades,
    )
