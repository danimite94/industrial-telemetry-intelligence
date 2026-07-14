"""Asset card HTML component — UI only, no analytics."""

import math

import pandas as pd

from components.health_badge import health_badge_html

_CARD_CLASS = {
    "Healthy":  "healthy",
    "Warning":  "warning",
    "Critical": "critical",
}


def _fmt(val, suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        if math.isnan(float(val)):
            return "—"
    except (TypeError, ValueError):
        pass
    if suffix == " laps":
        return f"{int(float(val))} laps"
    if suffix == "":
        return str(val)
    return f"{val}{suffix}"


def asset_card_html(row: pd.Series, index: int = 0) -> str:
    status   = row.get("HealthStatus", "Healthy")
    card_cls = _CARD_CLASS.get(status, "healthy")
    badge    = health_badge_html(status)
    driver   = row.get("Driver", "—")
    compound = row.get("CurrentCompound") or "—"
    tyre_str = _fmt(row.get("TyreLife"))
    lap_str  = _fmt(row.get("CurrentLap"))
    stint_str = _fmt(row.get("Stint"))
    risk     = row.get("RiskCategory") or "—"
    rec      = row.get("ActiveRecommendation") or "—"

    return f"""<div class="asset-card {card_cls}" style="--i:{index}">
  <div class="asset-card-header">
    <span class="asset-driver-code">{driver}</span>
    {badge}
  </div>
  <div class="asset-meta">
    <span class="asset-meta-label">Tyre</span>
    <span class="asset-meta-value">{compound}</span>
    <span class="asset-meta-label">Age</span>
    <span class="asset-meta-value">{tyre_str}</span>
    <span class="asset-meta-label">Lap</span>
    <span class="asset-meta-value">{lap_str}</span>
    <span class="asset-meta-label">Stint</span>
    <span class="asset-meta-value">{stint_str}</span>
    <span class="asset-meta-label">Risk</span>
    <span class="asset-meta-value">{risk}</span>
  </div>
  <div class="asset-recommendation">{rec}</div>
</div>"""
