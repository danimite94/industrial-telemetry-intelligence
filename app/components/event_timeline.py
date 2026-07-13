"""Event timeline HTML component — UI only, no analytics."""

import math
import re

import pandas as pd

_SEVERITY_DOT = {
    "Critical": "critical",
    "Warning":  "warning",
    "Info":     "info",
}

_SEVERITY_ICON = {
    "Critical": "🔴",
    "Warning":  "🟠",
    "Info":     "🔵",
}

# Strips the variable parts of a message (timestamps, car numbers, lap
# numbers, lap-time values) so that repeated race-control messages that only
# differ by those details (e.g. "WAVED BLUE FLAG FOR CAR 2 (SAR) TIMED AT
# 18:21:59" vs "...FOR CAR 77 (BOT) TIMED AT 19:03:27") collapse into one
# cluster instead of listing every occurrence.
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
_LAPTIME_RE = re.compile(r"\b\d{1,2}:\d{2}\.\d{2,3}\b")
_CAR_RE = re.compile(r"CAR\s+\d+\s*\([A-Z]{2,4}\)")
_LAP_RE = re.compile(r"\bLAP\s+\d+\b", re.IGNORECASE)


def _cluster_key(row: pd.Series) -> tuple:
    desc = str(row.get("Description") or "")
    desc = _CLOCK_RE.sub("", desc)
    desc = _LAPTIME_RE.sub("", desc)
    desc = _CAR_RE.sub("CAR", desc)
    desc = _LAP_RE.sub("LAP", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    return (
        _safe_str(row.get("EventType")),
        _safe_str(row.get("Severity")),
        _safe_str(row.get("Driver")),
        desc,
    )


def _cluster_events(events: pd.DataFrame) -> list[list[pd.Series]]:
    """Group consecutive rows (in the given order) that share a normalized
    event signature. Order is preserved — clustering never reorders events,
    it only collapses adjacent repeats."""
    clusters: list[list[pd.Series]] = []
    last_key = None
    for _, row in events.iterrows():
        key = _cluster_key(row)
        if last_key == key and clusters:
            clusters[-1].append(row)
        else:
            clusters.append([row])
            last_key = key
    return clusters


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
        if math.isnan(s):
            return "—"
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"
    except (TypeError, ValueError):
        return "—"


def _safe_str(val) -> str:
    if val is None:
        return ""
    try:
        if isinstance(val, float) and math.isnan(val):
            return ""
    except TypeError:
        pass
    s = str(val)
    return "" if s.lower() == "none" else s


def _single_event_html(row: pd.Series) -> str:
    severity = _safe_str(row.get("Severity")) or "Info"
    dot_cls  = _SEVERITY_DOT.get(severity, "info")
    icon     = _SEVERITY_ICON.get(severity, "🔵")

    time_str  = _fmt_time(row.get("SessionTimeSeconds"))
    etype     = _safe_str(row.get("EventType"))
    driver    = _safe_str(row.get("Driver"))
    driver_pfx = f"[{driver}] " if driver else ""
    desc      = _safe_str(row.get("Description"))
    action    = _safe_str(row.get("Action"))

    action_html = f'<div class="timeline-action">→ {action}</div>' if action else ""

    return (
        f'<div class="timeline-event {dot_cls}">'
        f'<div class="timeline-time">{time_str} · {etype}</div>'
        f'<div class="timeline-desc">{icon} {driver_pfx}{desc}</div>'
        f'{action_html}'
        f"</div>"
    )


def _cluster_html(rows: list[pd.Series]) -> str:
    if len(rows) == 1:
        return _single_event_html(rows[0])

    head = rows[0]
    severity = _safe_str(head.get("Severity")) or "Info"
    dot_cls  = _SEVERITY_DOT.get(severity, "info")
    icon     = _SEVERITY_ICON.get(severity, "🔵")
    etype    = _safe_str(head.get("EventType"))
    driver   = _safe_str(head.get("Driver"))
    driver_pfx = f"[{driver}] " if driver else ""
    desc     = _safe_str(head.get("Description"))
    time_str = _fmt_time(head.get("SessionTimeSeconds"))

    detail_items = "".join(_single_event_html(r) for r in rows)

    return (
        f'<details class="timeline-cluster">'
        f'<summary class="timeline-event {dot_cls}">'
        f'<div class="timeline-time">{time_str} · {etype} · {len(rows)}× clustered</div>'
        f'<div class="timeline-desc">{icon} {driver_pfx}{len(rows)}× {desc} '
        f'<span class="timeline-expand">expand ▾</span></div>'
        f'</summary>'
        f'<div class="timeline-cluster-items">{detail_items}</div>'
        f'</details>'
    )


def event_timeline_html(events: pd.DataFrame, max_events: int = 15) -> str:
    if events.empty:
        return '<div class="timeline"><p style="color:#94A3B8;font-size:0.78rem;">No events.</p></div>'

    clusters = _cluster_events(events)[:max_events]
    items = [_cluster_html(rows) for rows in clusters]

    return '<div class="timeline">' + "".join(items) + "</div>"
