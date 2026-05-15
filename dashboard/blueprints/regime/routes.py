"""Regime page — HMM state distribution, timeline, current regime."""
from __future__ import annotations

from flask import Blueprint, render_template

from ...services import data_service, stats

bp = Blueprint("regime", __name__, url_prefix="/regime", template_folder="../../templates")


def _compress_regime_timeline(series: list[str], max_segments: int = 300) -> list[dict]:
    """Compress consecutive same-regime segments to avoid DOM explosion.

    Returns a list of dicts: [{"label": "BULL", "count": 42}, ...]
    where count = number of consecutive timesteps with the same label.
    """
    if not series:
        return []

    segments: list[dict] = []
    current = series[0]
    count = 1

    for label in series[1:]:
        if label == current and len(segments) + 1 < max_segments:
            count += 1
        else:
            segments.append({"label": current, "count": count})
            current = label
            count = 1

    segments.append({"label": current, "count": count})
    return segments


def _segment_stats(segments: list[dict]) -> dict:
    """Per-regime segment statistics: count, avg length, longest run."""
    by_label: dict[str, dict] = {}
    for seg in segments:
        label = seg["label"]
        if label not in by_label:
            by_label[label] = {"count": 0, "total": 0, "longest": 0}
        by_label[label]["count"] += 1
        by_label[label]["total"] += seg["count"]
        by_label[label]["longest"] = max(by_label[label]["longest"], seg["count"])

    for label, data in by_label.items():
        data["avg"] = data["total"] / max(1, data["count"])
    return by_label


@bp.route("/")
def index():
    regime_series = data_service.load_regime()
    regime_dist = stats.regime_distribution()
    current_regime = stats.current_regime()

    # Compress timeline for rendering
    timeline_segments = _compress_regime_timeline(regime_series)

    # Per-regime segment aggregates
    segment_stats = _segment_stats(timeline_segments)
    n_transitions = max(0, len(timeline_segments) - 1)
    longest_run = max((seg["count"] for seg in timeline_segments), default=0)

    return render_template(
        "regime/index.html",
        regime_series=regime_series,
        regime_dist=regime_dist,
        current_regime=current_regime,
        n_regime_samples=len(regime_series),
        timeline_segments=timeline_segments,
        segment_stats=segment_stats,
        n_transitions=n_transitions,
        longest_run=longest_run,
    )
