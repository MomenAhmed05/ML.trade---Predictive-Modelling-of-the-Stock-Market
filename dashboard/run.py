"""CLI entry point: `python -m dashboard.run --port 5000`."""
from __future__ import annotations

import argparse

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ML.TRADE dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
