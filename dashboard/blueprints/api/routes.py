"""API blueprint — JSON endpoints for all dashboard data."""
from __future__ import annotations

from flask import Blueprint, jsonify

from ...services import data_service, stats

bp = Blueprint("api", __name__, url_prefix="/api", template_folder="../../templates")


@bp.route("/")
def index():
    return jsonify({
        "endpoints": [
            {"method": "GET", "path": "/api/",                       "desc": "This index"},
            {"method": "GET", "path": "/api/tournament",             "desc": "Tournament leaderboard CSV data"},
            {"method": "GET", "path": "/api/manifest",               "desc": "Master manifest JSON"},
            {"method": "GET", "path": "/api/regime",                 "desc": "Regime series and distribution"},
            {"method": "GET", "path": "/api/portfolio",              "desc": "Portfolio summary statistics"},
            {"method": "GET", "path": "/api/equity-curves",          "desc": "List equity curve images"},
            {"method": "GET", "path": "/api/comparison",             "desc": "LSTM vs MTL comparison data"},
            {"method": "GET", "path": "/api/sweep-curves",           "desc": "Sweep experiment equity curves"},
            {"method": "GET", "path": "/api/health",                 "desc": "Health check"},
        ]
    })


@bp.route("/tournament")
def tournament():
    df = data_service.load_tournament()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@bp.route("/manifest")
def manifest():
    return jsonify(data_service.load_manifest())


@bp.route("/regime")
def regime():
    series = data_service.load_regime()
    dist = stats.regime_distribution()
    current = stats.current_regime()
    return jsonify({
        "series": series,
        "distribution": dist,
        "current": current,
        "n_samples": len(series),
    })


@bp.route("/portfolio")
def portfolio():
    return jsonify(stats.portfolio_summary())


@bp.route("/equity-curves")
def equity_curves():
    return jsonify(data_service.list_equity_curves())


@bp.route("/comparison")
def comparison():
    df = data_service.load_comparison()
    if df is None:
        return jsonify(None)
    return jsonify(df.to_dict(orient="records"))


@bp.route("/sweep-curves")
def sweep_curves():
    return jsonify(data_service.list_sweep_equity_curves())


@bp.route("/health")
def health():
    return jsonify({"status": "ok"})
