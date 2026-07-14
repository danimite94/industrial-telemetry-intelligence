"""Status banner HTML component — UI only, no analytics."""


def status_banner_html(kpis: dict, track_status: str) -> str:
    track_color = {
        "Green":       "kpi-healthy",
        "Yellow":      "kpi-warning",
        "Red":         "kpi-critical",
        "Safety Car":  "kpi-warning",
        "Virtual SC":  "kpi-warning",
    }.get(track_status, "kpi-neutral")

    return f"""<div class="kpi-banner">
  <div class="kpi-card" style="--i:0">
    <div class="kpi-number kpi-neutral">{kpis['DriverCount']}</div>
    <div class="kpi-label">Assets</div>
  </div>
  <div class="kpi-card" style="--i:1">
    <div class="kpi-number kpi-healthy">{kpis['Healthy']}</div>
    <div class="kpi-label">Healthy</div>
  </div>
  <div class="kpi-card" style="--i:2">
    <div class="kpi-number kpi-warning">{kpis['Warning']}</div>
    <div class="kpi-label">Warning</div>
  </div>
  <div class="kpi-card" style="--i:3">
    <div class="kpi-number kpi-critical">{kpis['Critical']}</div>
    <div class="kpi-label">Critical</div>
  </div>
  <div class="kpi-card" style="--i:4">
    <div class="kpi-number kpi-info">{kpis['ActiveAlerts']}</div>
    <div class="kpi-label">Active Events</div>
  </div>
  <div class="kpi-card" style="--i:5">
    <div class="kpi-number {track_color}" style="font-size:1.6rem;padding-top:4px">{track_status}</div>
    <div class="kpi-label">Track Status</div>
  </div>
</div>"""
