"""Compare page — LSTM vs MTL side-by-side comparison."""
from __future__ import annotations

from flask import Blueprint, render_template

from ...services import data_service, stats

bp = Blueprint("compare", __name__, url_prefix="/compare", template_folder="../../templates")


@bp.route("/")
def index():
    comparison = data_service.load_comparison()
    summary = stats.portfolio_summary()

    # Pre-process DataFrame into list of dicts for Jinja2
    comparison_rows: list[dict] = []
    comparison_columns: list[str] = []
    if comparison is not None and not comparison.empty:
        comparison_rows = comparison.to_dict(orient="records")
        comparison_columns = list(comparison.columns)

    return render_template(
        "compare/index.html",
        comparison_rows=comparison_rows,
        comparison_columns=comparison_columns,
        summary=summary,
    )
