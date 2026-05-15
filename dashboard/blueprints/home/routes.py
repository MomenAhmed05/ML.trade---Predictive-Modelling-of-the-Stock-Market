"""Home page — backtest results front-and-center, with regime status."""
from __future__ import annotations

from flask import Blueprint, render_template

from ...services import data_service, stats

bp = Blueprint("home", __name__, template_folder="../../templates")


@bp.route("/")
def index():
    # Primary: backtest metrics (the thing the user cares about most)
    backtest = data_service.load_backtest_metrics()
    sweep_metrics = data_service.load_sweep_metrics()
    equity_curves = data_service.list_equity_curves()
    sweep_curves = data_service.list_sweep_equity_curves()

    # Secondary: regime + tournament overview
    regime_dist = stats.regime_distribution()
    current_regime = stats.current_regime()
    summary = stats.portfolio_summary()

    regime_label = current_regime.upper() if current_regime else "UNKNOWN"
    hero_regime = f"{regime_label} REGIME" if regime_label != "UNKNOWN" else "PIPELINE READY"

    has_data = bool(backtest) or bool(regime_dist) or bool(equity_curves) or (summary and summary.survivors > 0)

    return render_template(
        "home/index.html",
        hero_regime=hero_regime,
        regime_label=regime_label,
        current_regime=current_regime,
        regime_dist=regime_dist,
        summary=summary,
        backtest=backtest,
        sweep_metrics=sweep_metrics,
        equity_curves=equity_curves,
        sweep_curves=sweep_curves,
        has_data=has_data,
    )
