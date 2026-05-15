"""Portfolio page — equity curves, backtest metrics, sweep experiments."""
from __future__ import annotations

from flask import Blueprint, render_template

from ...services import data_service, stats

bp = Blueprint("portfolio", __name__, url_prefix="/portfolio", template_folder="../../templates")


def _safe_stocks(entry: dict) -> str:
    """Safely format stocks from manifest — handle string or list."""
    stocks = entry.get("stocks", "")
    if isinstance(stocks, list):
        return ", ".join(str(s) for s in stocks)
    return str(stocks)


@bp.route("/")
def index():
    equity_curves = data_service.list_equity_curves()
    sweep_curves = data_service.list_sweep_equity_curves()
    summary = stats.portfolio_summary()
    regime_dist = stats.regime_distribution()
    manifest = data_service.load_manifest()

    # Pre-process manifest for safe template rendering
    manifest_rows = []
    for entry in manifest:
        manifest_rows.append({
            "group_id": entry.get("group_id", ""),
            "stocks": _safe_stocks(entry),
            "val_sharpe": entry.get("val_sharpe", 0),
            "val_accuracy": entry.get("val_accuracy", 0),
            "val_trades": entry.get("val_trades", 0),
        })

    return render_template(
        "portfolio/index.html",
        equity_curves=equity_curves,
        sweep_curves=sweep_curves,
        summary=summary,
        regime_dist=regime_dist,
        manifest_rows=manifest_rows,
    )
