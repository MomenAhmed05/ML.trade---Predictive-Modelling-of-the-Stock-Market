"""Flask app factory with blueprint auto-discovery."""
from __future__ import annotations

import importlib
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, send_from_directory, url_for
from werkzeug.routing import BuildError

from .config import ALLOWED_STATIC_ROOTS, REPO_ROOT
from .services import stats


def _safe_url_for(endpoint: str, **values) -> str | None:
    """url_for that returns None for unknown endpoints (unregistered blueprints)."""
    try:
        return url_for(endpoint, **values)
    except BuildError:
        return None


def create_app() -> Flask:
    app = Flask(__name__)

    # Expose safe_url_for to templates so _nav.html can degrade gracefully.
    app.jinja_env.globals["safe_url_for"] = _safe_url_for

    # Custom filter: format a number with comma thousands separator.
    # Usage:  {{ value | comma }}  ->  "10,000"
    def _comma(value):
        """Format as integer with comma thousands separator (e.g. 10,000).
        Truncates fractional part — use commaf for decimals.
        """
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)

    def _commaf(value, decimals=2):
        try:
            return f"{float(value):,.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    app.jinja_env.filters["comma"] = _comma
    app.jinja_env.filters["commaf"] = _commaf

    # Inject badge/nav values every request so _badge.html and _nav.html have them.
    @app.context_processor
    def inject_globals() -> dict:
        try:
            selected, total = stats.survivor_count()
            regime = stats.current_regime()
        except Exception:  # pragma: no cover — defensive, loaders already swallow errors
            selected, total, regime = 0, 0, "UNKNOWN"
        return {
            "survivor_count_selected": selected,
            "survivor_count_total": total,
            "current_regime": regime,
        }

    # Auto-discover blueprints under dashboard/blueprints/<name>/__init__.py
    blueprints_dir = Path(__file__).parent / "blueprints"
    registered: list[str] = []
    if blueprints_dir.exists():
        for sub in sorted(blueprints_dir.iterdir()):
            if not sub.is_dir() or not (sub / "__init__.py").exists():
                continue
            mod_name = f"dashboard.blueprints.{sub.name}"
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                app.logger.warning("Failed to import blueprint %s: %s", sub.name, e)
                continue
            bp = getattr(mod, "bp", None)
            if bp is None:
                continue
            try:
                app.register_blueprint(bp)
                registered.append(sub.name)
            except Exception as e:
                app.logger.warning("Failed to register blueprint %s: %s", sub.name, e)
    app.config["_registered_blueprints"] = registered

    # Custom error handlers — dark-themed pages that preserve the JUO aesthetic.
    @app.errorhandler(404)
    def _not_found(exc):
        return render_template("error.html", code=404, message="Page not found"), 404

    @app.errorhandler(500)
    def _server_error(exc):
        return render_template("error.html", code=500, message="Internal server error"), 500

    @app.route("/_health")
    def _health():
        return jsonify({"status": "ok", "blueprints": registered})

    # The home blueprint registers `/` — no placeholder needed.
    # If home is not registered, redirect to tournament as fallback.
    if "home" not in registered and "tournament" in registered:
        @app.route("/")
        def _root():
            return redirect(url_for("tournament.index"))

    @app.route("/files/<path:rel>")
    def _files(rel: str):
        """Serve files from results/ or models/ only (path-traversal safe)."""
        target = (REPO_ROOT / rel).resolve()
        for root in ALLOWED_STATIC_ROOTS:
            try:
                target.relative_to(root.resolve())
            except ValueError:
                continue
            if not target.is_file():
                abort(404)
            return send_from_directory(target.parent, target.name)
        abort(404)

    return app
