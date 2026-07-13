"""Industrial Asset Monitoring and Decision Support System dashboard.

FastF1 telemetry is used as a public surrogate for ESP / SCADA / production
sensor data. Run with: streamlit run app/streamlit_app.py

Navigation hierarchy:
    Level 1 – Fleet Overview   : fleet health KPIs, asset health table, event log
    Level 2 – Driver Detail    : per-driver health, expected vs actual, trends
    Level 3 – Telemetry Replay : arcade replay (python app/arcade_replay.py)

All remaining tabs preserve the full Phase 1-10 analytics pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.graph_objects as go

from src import config
from components.asset_card import asset_card_html
from components.event_timeline import event_timeline_html
from components.health_badge import health_badge_html
from components.sparkline import sparkline_figure
from components.status_banner import status_banner_html
from src.anomaly_detection import get_anomaly_table
from src.baseline_models import (
    average_lap_time_by_driver,
    consistency_score,
    degradation_summary,
    fastest_lap_per_driver,
)
from src.context_engine import (
    OBSERVED_LAPTIME_INCREASE,
    OBSERVED_LOW_SPEED,
    OBSERVED_SPEED_ANOMALY,
    TRACK_STATUS_LABELS,
    align_context_to_session,
    calculate_context_changes,
    generate_context_summary,
    get_context_at_timestamp,
    load_context,
)
from src.data_cleaning import load_and_clean_all, load_and_clean_fleet, load_and_clean_season
from src.decision_support import build_recommendations_table
from src.degradation_analysis import degradation_per_stint
from src.event_log import (
    EVENT_COLUMNS,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    build_event_log,
)
from src.fleet_health import (
    HEALTH_COLOR_LABEL,
    HEALTH_CRITICAL,
    HEALTH_HEALTHY,
    HEALTH_WARNING,
    compute_fleet_health,
    fleet_kpis,
)
from src.fleet_analysis import (
    degradation_by_year,
    field_average_gap_by_year,
    fleet_benchmark_table,
    teammate_gap_by_year,
)
from src.health_assessment import (
    HEALTH_EXPLAINED,
    HEALTH_PARTIALLY_EXPLAINED,
    HEALTH_UNEXPLAINED,
    assess_anomaly_health,
    health_summary,
)
from src.operational_assistant import answer_question
from src.predictive_models import (
    compare_models,
    degradation_risk_scores,
    explain_linear_model,
    explain_tree_model,
    forecast_stint_degradation,
)
from src.seasonal_analysis import (
    driver_position_trend,
    season_driver_kpis,
    season_team_kpis,
    speed_trap_trend,
)
from src.visualisation import (
    plot_degradation_by_year,
    plot_degradation_forecast,
    plot_driver_comparison,
    plot_feature_importance,
    plot_lap_times,
    plot_model_comparison,
    plot_pace_gap_trend,
    plot_pit_recommendations,
    plot_position_trend,
    plot_predicted_vs_actual,
    plot_speed_trace,
    plot_speed_trap_trend,
    plot_throttle_brake_trace,
    plot_tyre_life_vs_laptime,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Industrial Telemetry Intelligence", layout="wide")

_CSS_FILE = Path(__file__).parent / "styles" / "dark_theme.css"
if _CSS_FILE.exists():
    st.markdown(f"<style>{_CSS_FILE.read_text()}</style>", unsafe_allow_html=True)

st.title("Industrial Asset Monitoring and Decision Support System")
st.caption("Demonstrated on Formula 1 telemetry as a public surrogate for industrial sensor data")

# ---------------------------------------------------------------------------
# Helper functions (UI only – no analytics logic)
# ---------------------------------------------------------------------------

_REQUIRED_FILES = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]


def _data_missing() -> list[Path]:
    return [f for f in _REQUIRED_FILES if not f.exists()]


def _style_health(val: str) -> str:
    if val == HEALTH_CRITICAL:
        return "background-color: #8b1a1a; color: white; font-weight: bold"
    if val == HEALTH_WARNING:
        return "background-color: #7a5200; color: white; font-weight: bold"
    if val == HEALTH_HEALTHY:
        return "background-color: #1a5c1a; color: white; font-weight: bold"
    return ""


def _style_severity(val: str) -> str:
    if val == SEVERITY_CRITICAL:
        return "background-color: #8b1a1a; color: white"
    if val == SEVERITY_WARNING:
        return "background-color: #7a5200; color: white"
    return ""


def _last_track_status(laps_df: pd.DataFrame) -> str:
    if "TrackStatus" not in laps_df.columns or laps_df.empty:
        return "Unknown"
    last = laps_df.dropna(subset=["LapStartTimeSeconds"]).sort_values("LapStartTimeSeconds").iloc[-1]
    raw = str(last.get("TrackStatus", "")).strip()
    return TRACK_STATUS_LABELS.get(raw[-1] if raw else "", "Unknown")


def _fleet_risk_distribution(risk_df: pd.DataFrame) -> go.Figure:
    """Forward-looking degradation risk (Phase 5 tertiles) across every
    current stint — distinct from the Healthy/Warning/Critical KPI banner,
    which reflects *current* anomaly/recommendation state. This answers
    "where is the fleet heading", not "what does it look like right now"."""
    order = ["Low", "Medium", "High"]
    colors = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}
    if risk_df.empty:
        counts = pd.Series([0, 0, 0], index=order)
    else:
        counts = risk_df["RiskCategory"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.values, y=order, orientation="h",
        marker=dict(color=[colors[c] for c in order]),
        text=counts.values, textposition="outside",
        textfont=dict(color="#F8FAFC"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=30, t=10, b=20), height=220,
        xaxis=dict(showgrid=False, color="#94A3B8", title="Stints"),
        yaxis=dict(showgrid=False, color="#F8FAFC", autorange="reversed"),
        showlegend=False,
    )
    return fig


def _augment_with_risk(health_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """Attach each driver's *current*-stint projected degradation increase
    (Phase 5) onto the fleet health table, so ties in health severity break
    on forward-looking risk instead of arbitrary row order."""
    if risk_df.empty or health_df.empty:
        return health_df.assign(ProjectedIncreaseSeconds=float("nan"))
    lookup = risk_df.set_index(["Driver", "Stint"])["ProjectedIncreaseSeconds"]
    values = [
        lookup.get((row["Driver"], row["Stint"]), float("nan"))
        for _, row in health_df.iterrows()
    ]
    return health_df.assign(ProjectedIncreaseSeconds=values)


def _sort_by_health(health_df: pd.DataFrame) -> pd.DataFrame:
    """Primary sort: health severity (Critical > Warning > Healthy).
    Secondary sort (tiebreak): projected degradation risk, worst first —
    only applied when `_augment_with_risk` has been called first."""
    order = {HEALTH_CRITICAL: 0, HEALTH_WARNING: 1, HEALTH_HEALTHY: 2}
    df = health_df.assign(_order=health_df["HealthStatus"].map(order).fillna(3))
    if "ProjectedIncreaseSeconds" in df.columns:
        df = df.assign(_risk_sort=-df["ProjectedIncreaseSeconds"].fillna(float("-inf")))
        return (
            df.sort_values(["_order", "_risk_sort"])
            .drop(columns=["_order", "_risk_sort"])
            .reset_index(drop=True)
        )
    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _fmt_meta(val, suffix=""):
    if val is None:
        return "—"
    try:
        import math
        if math.isnan(float(val)):
            return "—"
    except (TypeError, ValueError):
        pass
    return f"{int(float(val))}{suffix}" if suffix == " laps" else str(val)


def _time_to_intervention(current_stint_rec: pd.DataFrame) -> str:
    """Laps remaining until the Phase 7 degradation forecast crosses the pit
    threshold for the driver's *current* stint — the real, existing proxy
    for "remaining useful life" (no new RUL model)."""
    if current_stint_rec.empty:
        return "—"
    row = current_stint_rec.iloc[0]
    crossing = row.get("ProjectedCrossingStintLap")
    current_lap = row.get("CurrentStintLap")
    if crossing is None or pd.isna(crossing) or current_lap is None or pd.isna(current_lap):
        action = str(row.get("RecommendedAction") or "")
        return "Stable" if action.startswith("No action") else "—"
    remaining = int(crossing) - int(current_lap)
    return "Now" if remaining <= 0 else f"{remaining} laps"


def _driver_header_html(dhr: pd.Series, current_stint_rec: pd.DataFrame | None = None) -> str:
    status = dhr.get("HealthStatus", HEALTH_HEALTHY)
    status_cls = {HEALTH_HEALTHY: "healthy", HEALTH_WARNING: "warning", HEALTH_CRITICAL: "critical"}.get(status, "healthy")
    badge = health_badge_html(status)
    driver = dhr.get("Driver", "—")
    time_to_intervention = _time_to_intervention(
        current_stint_rec if current_stint_rec is not None else pd.DataFrame()
    )

    return f"""<div class="driver-header {status_cls}">
  <div>
    <p class="driver-name">{driver}</p>
    {badge}
  </div>
  <div class="driver-meta-grid">
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Tyre</span>
      <span class="driver-meta-item-value">{dhr.get("CurrentCompound") or "—"}</span>
    </div>
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Age</span>
      <span class="driver-meta-item-value">{_fmt_meta(dhr.get("TyreLife"), " laps")}</span>
    </div>
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Lap</span>
      <span class="driver-meta-item-value">{_fmt_meta(dhr.get("CurrentLap"))}</span>
    </div>
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Stint</span>
      <span class="driver-meta-item-value">{_fmt_meta(dhr.get("Stint"))}</span>
    </div>
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Risk</span>
      <span class="driver-meta-item-value">{dhr.get("RiskCategory") or "—"}</span>
    </div>
    <div class="driver-meta-item">
      <span class="driver-meta-item-label">Time to Intervention</span>
      <span class="driver-meta-item-value">{time_to_intervention}</span>
    </div>
  </div>
</div>"""


_RISK_COLOR = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}
_REC_ICON = {"Pit now": "🔴", "Pit within": "🟠", "Monitor": "🟠", "No action": "🟢"}
_CONFIDENCE_CLASS = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}


def _confidence_badge_html(confidence: str | None) -> str:
    if not confidence:
        return ""
    cls = _CONFIDENCE_CLASS.get(confidence, "conf-medium")
    return f'<span class="conf-badge {cls}">Confidence: {confidence}</span>'


def _last_anomaly_for(anomalies_df: pd.DataFrame, driver: str) -> pd.Series | None:
    driver_anomalies = anomalies_df[anomalies_df["Driver"] == driver] if not anomalies_df.empty else anomalies_df
    if driver_anomalies.empty:
        return None
    return driver_anomalies.sort_values("LapNumber").iloc[-1]


def _evidence_row_html(label: str, value: str) -> str:
    return (
        f'<div class="rec-evidence-row">'
        f'<span class="rec-evidence-label">{label}</span>'
        f'<span class="rec-evidence-value">{value}</span>'
        f'</div>'
    )


def _decision_card_html(row: pd.Series, fits_df: pd.DataFrame, last_anomaly: pd.Series | None) -> str:
    """One stint's recommendation as a full decision card: action, evidence,
    remaining useful life, confidence, root cause, and source phase — never
    just a bare action label. Every field is read from data already computed
    by Phase 5/7/10 (`degradation_per_stint`, `build_recommendations_table`,
    `assess_anomaly_health`); nothing here is a new calculation."""
    risk = row.get("RiskCategory") or "—"
    action = row.get("RecommendedAction") or "—"
    compound = row.get("Compound") or "—"
    stint = row.get("Stint")
    stint_str = str(int(stint)) if stint is not None and not pd.isna(stint) else "—"
    slope = row.get("DegradationSecondsPerLap")
    slope_str = f"{slope:+.4f} s/lap" if slope is not None and not pd.isna(slope) else "—"

    # RiskCategory (Phase 5, relative to this race's other stints) and
    # RecommendedAction (Phase 7, an absolute threshold) can legitimately
    # disagree — a stint can rank High risk while still being "No action
    # needed" against the fixed pit threshold. Displaying the bare Phase 7
    # text alone would look identical to a genuinely healthy stint, so a
    # High/Medium risk stint always states that disagreement explicitly
    # instead of just showing "No action needed" (mirrors the same fix in
    # `fleet_health.compute_fleet_health`).
    if action.startswith("No action") and risk in ("High", "Medium"):
        action = f"Monitor — {risk} degradation risk (below Phase 7 pit threshold)"

    fit_row = fits_df[fits_df["Stint"] == stint] if stint is not None and not fits_df.empty else pd.DataFrame()
    start_pace = float(fit_row["StartingLapTimeEstimate"].iloc[0]) if not fit_row.empty else None
    start_pace_str = f"{start_pace:.3f}s" if start_pace is not None else "unknown"
    evidence_str = f"Degradation {slope_str} from a {start_pace_str} stint-start baseline"

    rul_str = _time_to_intervention(pd.DataFrame([row]))
    if rul_str == "—":
        rul_str = f"Beyond the {config.DECISION_MAX_HORIZON_LAPS}-lap forecast horizon"
    elif rul_str == "Now":
        rul_str = "Threshold already crossed"
    elif rul_str != "Stable":
        rul_str = f"{rul_str} until pit threshold (stint lap {int(row.get('ProjectedCrossingStintLap'))})"

    risk_color = _RISK_COLOR.get(risk, "#94A3B8")
    icon = next((v for k, v in _REC_ICON.items() if k in str(action)), "ℹ️")

    if last_anomaly is not None:
        confidence = last_anomaly.get("Confidence")
        anomaly_status = last_anomaly.get("HealthStatus", "—")
        anomaly_lap = last_anomaly.get("LapNumber")
        anomaly_lap_str = str(int(anomaly_lap)) if anomaly_lap is not None and not pd.isna(anomaly_lap) else "—"
        explanation = last_anomaly.get("Explanation") or "No contextual explanation available."
        evidence_rows = (
            _evidence_row_html("Last anomaly", f"Lap {anomaly_lap_str} — {anomaly_status}")
            + _evidence_row_html("Root cause", explanation)
        )
        confidence_html = _confidence_badge_html(confidence)
        source = "Phase 5 (forecast) → Phase 7 (threshold) · Phase 10 (context)"
    else:
        evidence_rows = ""
        confidence_html = ""
        source = "Phase 5 (forecast) → Phase 7 (threshold)"

    return f"""<div class="rec-card">
  <div class="rec-header">
    <span class="rec-icon">{icon}</span>
    <span class="rec-action">{action}</span>
    <span class="rec-risk" style="color:{risk_color}">{risk} Risk</span>
    {confidence_html}
  </div>
  <div class="rec-meta">Stint {stint_str} · {compound}</div>
  <div class="rec-evidence">
    {_evidence_row_html("Evidence", evidence_str)}
    {_evidence_row_html("Remaining useful life", rul_str)}
    {evidence_rows}
  </div>
  <div class="rec-source">Source: {source}</div>
</div>"""


def _anomaly_only_card_html(dhr: pd.Series, last_anomaly: pd.Series) -> str:
    """Fallback decision card for a Warning/Critical driver whose escalation
    came from an anomaly with no fitted degradation stint (so there is no
    Phase 7 recommendation row at all) — still explains itself instead of
    falling through to a bare 'No recommendations available' message."""
    status = last_anomaly.get("HealthStatus", "—")
    confidence = last_anomaly.get("Confidence")
    anomaly_lap = last_anomaly.get("LapNumber")
    anomaly_lap_str = str(int(anomaly_lap)) if anomaly_lap is not None and not pd.isna(anomaly_lap) else "—"
    explanation = last_anomaly.get("Explanation") or "No contextual explanation available."
    action = dhr.get("ActiveRecommendation") or "Investigate — unexplained anomaly"
    icon = "🔴" if "Investigate" in action else "🟠"

    return f"""<div class="rec-card">
  <div class="rec-header">
    <span class="rec-icon">{icon}</span>
    <span class="rec-action">{action}</span>
    {_confidence_badge_html(confidence)}
  </div>
  <div class="rec-evidence">
    {_evidence_row_html("Last anomaly", f"Lap {anomaly_lap_str} — {status}")}
    {_evidence_row_html("Root cause", explanation)}
  </div>
  <div class="rec-source">Source: Phase 10 (context-aware health assessment)</div>
</div>"""


def _recommendations_html(recs_df: pd.DataFrame, fits_df: pd.DataFrame, anomalies_df: pd.DataFrame, driver: str) -> str:
    last_anomaly = _last_anomaly_for(anomalies_df, driver)
    cards = [_decision_card_html(row, fits_df, last_anomaly) for _, row in recs_df.iterrows()]
    return '<div class="rec-list">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Tab definitions – Fleet Overview and Driver Detail come first
# ---------------------------------------------------------------------------

(
    overview_tab, driver_tab,
    race_tab, season_tab, fleet_tab, predictive_tab,
    assistant_tab, decision_tab, context_tab, health_tab,
) = st.tabs([
    "Fleet Overview",
    "Driver Detail",
    "Race Detail (Phase 1-2)",
    "Season Monitoring (Phase 3)",
    "Fleet Monitoring (Phase 4)",
    "Predictive Analytics (Phase 5)",
    "Operational Assistant (Phase 6)",
    "Decision Support (Phase 7)",
    "Operational Context (Phase 9)",
    "Health Assessment (Phase 10)",
])

# ===========================================================================
# Level 1 – Fleet Overview
# ===========================================================================

with overview_tab:
    missing = _data_missing()
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.write("Missing files:", [str(f) for f in missing])
    else:
        ov_laps, _, _ = load_and_clean_all()
        ov_context = align_context_to_session(load_context())
        ov_anomalies = assess_anomaly_health(ov_laps, ov_context)
        ov_recs = build_recommendations_table(ov_laps)
        ov_risk = degradation_risk_scores(ov_laps)
        ov_health = _augment_with_risk(compute_fleet_health(ov_laps, ov_anomalies, ov_recs), ov_risk)
        ov_kpis = fleet_kpis(ov_health, ov_laps)
        ov_health_summary = health_summary(ov_anomalies)
        ov_event_log = build_event_log(ov_laps, ov_context, ov_anomalies, ov_recs)
        ov_track_status = _last_track_status(ov_laps)

        # --- Row 1: Status banner -------------------------------------------
        st.markdown(status_banner_html(ov_kpis, ov_track_status), unsafe_allow_html=True)

        # --- Row 1b: Executive KPIs ------------------------------------------
        ov_health_ranked = _sort_by_health(ov_health)
        highest_risk_driver = ov_health_ranked.iloc[0]["Driver"] if not ov_health_ranked.empty else "—"
        interventions_due = int(ov_recs["RecommendedAction"].str.startswith("Pit").sum()) if not ov_recs.empty else 0
        avg_degradation = ov_risk["DegradationSecondsPerLap"].mean() if not ov_risk.empty else None
        confidence_pct = (
            100.0 if ov_health_summary["TotalFlagged"] == 0 else ov_health_summary["NoiseReductionPct"]
        )
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Highest Risk Asset", highest_risk_driver,
                   help="Top of the health-severity ranking, tie-broken by projected degradation risk (Phase 5).")
        x2.metric("Interventions Due", interventions_due,
                   help="Stints with an active 'Pit now' / 'Pit within N laps' recommendation (Phase 7).")
        x3.metric("Avg Degradation Rate", f"{avg_degradation:+.4f} s/lap" if avg_degradation is not None else "—",
                   help="Mean projected lap-time degradation slope across all current stints (Phase 5).")
        x4.metric("Operational Confidence", f"{confidence_pct:.0f}%",
                   help="Share of flagged anomalies fully explained by operational context (Phase 10).")

        # --- Row 2: Fleet risk distribution + top-risk assets ----------------
        col_risk, col_top = st.columns([1, 2])
        with col_risk:
            st.markdown('<p class="section-heading">Fleet Risk Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(_fleet_risk_distribution(ov_risk), use_container_width=True)
        with col_top:
            st.markdown('<p class="section-heading">Highest Risk Assets</p>', unsafe_allow_html=True)
            top5 = ov_health_ranked.head(5)
            st.markdown(
                '<div class="asset-grid">'
                + "".join(asset_card_html(row) for _, row in top5.iterrows())
                + "</div>",
                unsafe_allow_html=True,
            )

        # --- Row 3: Recent event timeline -----------------------------------
        st.markdown('<p class="section-heading">Recent Events</p>', unsafe_allow_html=True)
        recent_events = ov_event_log.sort_values("SessionTimeSeconds", ascending=False)
        st.markdown(event_timeline_html(recent_events), unsafe_allow_html=True)

        # --- Row 4: All assets grid -------------------------------------------
        st.markdown('<p class="section-heading">All Assets</p>', unsafe_allow_html=True)
        st.caption(
            "Health status derived from anomaly assessment (Phase 10) and pit recommendations "
            "(Phase 7). No new models — all inputs come from existing analytics."
        )
        f1, f2, f3 = st.columns(3)
        with f1:
            sort_by = st.selectbox(
                "Sort by", ["Health status", "Projected risk", "Tyre age", "Driver"], key="overview_sort",
            )
        with f2:
            status_filter = st.multiselect(
                "Filter: health status", [HEALTH_CRITICAL, HEALTH_WARNING, HEALTH_HEALTHY],
                default=[HEALTH_CRITICAL, HEALTH_WARNING, HEALTH_HEALTHY], key="overview_status_filter",
            )
        with f3:
            compound_options = sorted(c for c in ov_health["CurrentCompound"].dropna().unique() if c)
            compound_filter = st.multiselect(
                "Filter: tyre compound", compound_options, default=compound_options, key="overview_compound_filter",
            )

        filtered_health = ov_health[
            ov_health["HealthStatus"].isin(status_filter) & ov_health["CurrentCompound"].isin(compound_filter)
        ]
        if sort_by == "Health status":
            sorted_health = _sort_by_health(filtered_health)
        elif sort_by == "Projected risk":
            sorted_health = filtered_health.sort_values("ProjectedIncreaseSeconds", ascending=False, na_position="last")
        elif sort_by == "Tyre age":
            sorted_health = filtered_health.sort_values("TyreLife", ascending=False, na_position="last")
        else:
            sorted_health = filtered_health.sort_values("Driver")

        if sorted_health.empty:
            st.info("No assets match the selected filters.")
        else:
            st.markdown(
                '<div class="asset-grid">'
                + "".join(asset_card_html(row) for _, row in sorted_health.iterrows())
                + "</div>",
                unsafe_allow_html=True,
            )

        # --- Driver selector → Driver Detail --------------------------------
        st.divider()
        available_drivers = sorted(ov_laps["Driver"].unique())
        selected_for_detail = st.selectbox(
            "Select driver to view in Driver Detail tab:",
            available_drivers,
            index=available_drivers.index(
                st.session_state.get("detail_driver", config.COMPARISON_DRIVERS[0])
                if st.session_state.get("detail_driver") in available_drivers
                else available_drivers[0]
            ),
            key="overview_driver_selector",
        )
        if selected_for_detail:
            st.session_state["detail_driver"] = selected_for_detail
        st.info("Switch to the **Driver Detail** tab above to see the full driver view.")

        st.divider()
        st.subheader("Information Architecture")
        st.markdown(
            """
            This overview implements **Level 1** of the industrial monitoring hierarchy:

            | Level | View | Question answered |
            |-------|------|-------------------|
            | 1 | Fleet Overview (this tab) | What requires my attention? |
            | 2 | Driver Detail | Why? |
            | 3 | Telemetry Replay (`arcade_replay.py`) | What happened exactly? |

            Health status rules are deterministic and reuse existing modules only:
            **Critical** = unexplained anomaly (Phase 10) or "Pit now" (Phase 7);
            **Warning** = partially-explained anomaly or active pit recommendation;
            **Healthy** = no anomalies, no actionable recommendation.
            """
        )

# ===========================================================================
# Level 2 – Driver Detail
# ===========================================================================

with driver_tab:
    missing = _data_missing()
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.write("Missing files:", [str(f) for f in missing])
    else:
        dd_laps, _, dd_telemetry = load_and_clean_all()
        dd_context = align_context_to_session(load_context())
        dd_anomalies = assess_anomaly_health(dd_laps, dd_context)
        dd_recs = build_recommendations_table(dd_laps)
        dd_health = compute_fleet_health(dd_laps, dd_anomalies, dd_recs)
        dd_drivers = sorted(dd_laps["Driver"].unique())

        # Driver selector – pre-populated from Fleet Overview session state
        default_driver = st.session_state.get("detail_driver", config.COMPARISON_DRIVERS[0])
        if default_driver not in dd_drivers:
            default_driver = dd_drivers[0]
        selected_driver = st.selectbox(
            "Driver", dd_drivers, index=dd_drivers.index(default_driver),
            key="driver_detail_selector",
        )
        st.session_state["detail_driver"] = selected_driver

        # Resolve driver health row
        driver_health_rows = dd_health[dd_health["Driver"] == selected_driver]
        if driver_health_rows.empty:
            st.warning(f"No health data for {selected_driver}.")
        else:
            dhr = driver_health_rows.iloc[0]
            driver_laps = dd_laps[dd_laps["Driver"] == selected_driver].copy()
            fits = degradation_per_stint(driver_laps)
            current_stint = dhr["Stint"]
            current_fit = fits[fits["Stint"] == current_stint] if not fits.empty and current_stint is not None else pd.DataFrame()
            current_stint_rec = (
                dd_recs[(dd_recs["Driver"] == selected_driver) & (dd_recs["Stint"] == current_stint)]
                if not dd_recs.empty else pd.DataFrame()
            )

            # --- Driver header card -----------------------------------------
            st.markdown(_driver_header_html(dhr, current_stint_rec), unsafe_allow_html=True)

            # --- Expected vs Actual ----------------------------------------
            st.markdown('<p class="section-heading">Expected vs Actual</p>', unsafe_allow_html=True)
            avg_df = average_lap_time_by_driver(dd_laps)
            avg_row = avg_df[avg_df["Driver"] == selected_driver]
            expected_avg = float(avg_row["AvgLapTimeSeconds"].iloc[0]) if not avg_row.empty else None

            last_lap_rows = driver_laps.sort_values("LapNumber")
            actual_last = float(last_lap_rows.iloc[-1]["LapTimeSeconds"]) if not last_lap_rows.empty else None

            expected_slope = float(current_fit["DegradationSecondsPerLap"].iloc[0]) if not current_fit.empty else None

            e1, e2, e3 = st.columns(3)
            if expected_avg is not None:
                delta_pace = (actual_last - expected_avg) if actual_last is not None else None
                e1.metric("Expected Avg Pace", f"{expected_avg:.3f}s",
                          help="Driver's average lap time across this race (baseline).")
                e2.metric("Last Lap Time",
                          f"{actual_last:.3f}s" if actual_last is not None else "—",
                          delta=f"{delta_pace:+.3f}s" if delta_pace is not None else None)
            if expected_slope is not None:
                e3.metric("Degradation Slope", f"{expected_slope:+.4f} s/lap",
                          delta="Degrading" if expected_slope > 0 else "Improving",
                          help="Linear fit of lap time vs stint lap number (Phase 5).")

            # --- Lap time trend + sparkline ---------------------------------
            st.markdown('<p class="section-heading">Lap Time Trend</p>', unsafe_allow_html=True)
            spark_col, chart_col = st.columns([1, 4])
            with spark_col:
                lap_times_series = (
                    driver_laps.sort_values("LapNumber")["LapTimeSeconds"]
                    .dropna().reset_index(drop=True)
                )
                if not lap_times_series.empty:
                    st.plotly_chart(sparkline_figure(lap_times_series, "#3B82F6"),
                                   use_container_width=True)
            with chart_col:
                st.plotly_chart(plot_lap_times(driver_laps), use_container_width=True)

            # --- Tyre degradation + sparkline --------------------------------
            st.markdown('<p class="section-heading">Tyre Degradation</p>', unsafe_allow_html=True)
            spark_col2, chart_col2 = st.columns([1, 4])
            with spark_col2:
                tyre_life_series = (
                    driver_laps.sort_values("LapNumber")["TyreLife"]
                    .dropna().reset_index(drop=True)
                )
                if not tyre_life_series.empty:
                    st.plotly_chart(sparkline_figure(tyre_life_series, "#F59E0B"),
                                   use_container_width=True)
            with chart_col2:
                st.plotly_chart(plot_tyre_life_vs_laptime(driver_laps), use_container_width=True)

            # --- Decision cards -----------------------------------------------
            st.markdown('<p class="section-heading">Decision Support</p>', unsafe_allow_html=True)
            driver_recs = dd_recs[dd_recs["Driver"] == selected_driver] if not dd_recs.empty else pd.DataFrame()
            driver_last_anomaly = _last_anomaly_for(dd_anomalies, selected_driver)

            if not driver_recs.empty:
                st.markdown(
                    _recommendations_html(driver_recs, fits, dd_anomalies, selected_driver),
                    unsafe_allow_html=True,
                )
            elif dhr["HealthStatus"] != HEALTH_HEALTHY and driver_last_anomaly is not None:
                # No fitted degradation stint exists, but the driver is still
                # Warning/Critical from an anomaly — explain that instead of
                # silently falling through to "no recommendations".
                st.markdown(_anomaly_only_card_html(dhr, driver_last_anomaly), unsafe_allow_html=True)
            else:
                st.success(f"No active recommendations for {selected_driver} — stable pace, no anomalies.")

            # --- Driver event timeline --------------------------------------
            st.markdown('<p class="section-heading">Driver Events</p>', unsafe_allow_html=True)
            dd_event_log = build_event_log(dd_laps, dd_context, dd_anomalies, dd_recs)
            driver_events = dd_event_log[
                dd_event_log["Driver"] == selected_driver
            ].sort_values("SessionTimeSeconds", ascending=False)
            if driver_events.empty:
                st.success(f"No events for {selected_driver}.")
            else:
                st.markdown(event_timeline_html(driver_events), unsafe_allow_html=True)

            st.divider()

            # --- Replay launch ---------------------------------------------
            st.subheader("Telemetry Replay (Level 3)")
            st.caption(
                "The replay shows the fastest lap with live speed, throttle, brake, and gear gauges, "
                "plus the operational context panel (Phase 9)."
            )
            if st.button(f"▶ Launch Replay — {selected_driver}", key="launch_replay"):
                import subprocess
                replay_script = Path(__file__).parent / "arcade_replay.py"
                subprocess.Popen(
                    [sys.executable, str(replay_script), "--drivers", selected_driver],
                    cwd=Path(__file__).parent.parent,
                )
                st.success(f"Replay launched for {selected_driver} — check your desktop.")

# ===========================================================================
# Phase 1-2 – Race Detail (preserved exactly)
# ===========================================================================

with race_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]

    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.write("Missing files:", [str(f) for f in missing])
        st.stop()

    laps_df, weather_df, telemetry_df = load_and_clean_all()
    st.success(
        f"Loaded {config.SEASON_YEAR} {config.EVENT_NAME} - Session {config.SESSION_NAME}: "
        f"{laps_df['Driver'].nunique()} drivers, {len(laps_df)} laps, {len(telemetry_df)} telemetry samples."
    )

    st.divider()
    st.subheader("Driver Selection")

    available_drivers = sorted(laps_df["Driver"].unique())
    default_drivers = [d for d in config.COMPARISON_DRIVERS if d in available_drivers] or available_drivers[:2]

    col1, col2 = st.columns(2)
    with col1:
        driver_a = st.selectbox("Driver A", available_drivers, index=available_drivers.index(default_drivers[0]))
    with col2:
        remaining = [d for d in available_drivers if d != driver_a]
        default_b_index = remaining.index(default_drivers[1]) if len(default_drivers) > 1 and default_drivers[1] in remaining else 0
        driver_b = st.selectbox("Driver B", remaining, index=default_b_index)

    st.divider()
    st.subheader("Lap Time Trend")
    st.plotly_chart(plot_lap_times(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), width="stretch")

    st.subheader("Tyre Degradation")
    st.plotly_chart(plot_tyre_life_vs_laptime(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), width="stretch")
    st.dataframe(degradation_summary(laps_df[laps_df["Driver"].isin([driver_a, driver_b])]), width="stretch")

    st.subheader("Telemetry Trace (Fastest Lap)")
    telemetry_drivers = set(telemetry_df["Driver"].unique()) if not telemetry_df.empty else set()
    if driver_a in telemetry_drivers:
        st.plotly_chart(plot_speed_trace(telemetry_df, driver_a), width="stretch")
        st.plotly_chart(plot_throttle_brake_trace(telemetry_df, driver_a), width="stretch")
    if {driver_a, driver_b}.issubset(telemetry_drivers):
        st.plotly_chart(plot_driver_comparison(telemetry_df, driver_a, driver_b), width="stretch")
    else:
        st.info(
            f"No cached fastest-lap telemetry for {sorted({driver_a, driver_b} - telemetry_drivers)} "
            "(likely no completed timed lap for this driver in the session). "
            "Re-run `python -m src.data_ingestion` if data/processed is stale."
        )

    st.divider()
    st.subheader("Baseline Metrics")
    b1, b2 = st.columns(2)
    with b1:
        st.write("Average lap time by driver")
        st.dataframe(average_lap_time_by_driver(laps_df), width="stretch")
        st.write("Fastest lap per driver")
        st.dataframe(fastest_lap_per_driver(laps_df), width="stretch")
    with b2:
        st.write("Consistency score (lap time std dev, lower = more consistent)")
        st.dataframe(consistency_score(laps_df), width="stretch")

    st.subheader("Anomaly Table (|z-score| > threshold)")
    st.dataframe(get_anomaly_table(laps_df), width="stretch")

    st.divider()
    st.subheader("From Lap Telemetry to Asset Monitoring")
    st.markdown(
        """
        - **Lap = duty cycle.** Each lap is one operating cycle of the asset (e.g. one pump cycle, one ESP run interval).
        - **Tyre stint = run-to-failure interval.** Lap time climbing across a stint is the same shape as vibration,
          temperature, or vibration trending upward as a pump or ESP wears toward failure.
        - **Lap time z-score = SCADA alarm rule.** Flagging a lap because it deviates from that driver's own normal
          range is the simplest possible anomaly detector, identical in form to a sensor threshold alarm.
        - **Speed/throttle/brake trace = multi-sensor stream.** These are the closest public analogue to high-frequency
          pressure, flow, and current readings from a wellhead or ESP controller.

        Phases 1-2 deliberately use only descriptive baselines (averages, std dev, linear degradation slope,
        z-score) so that later phases (seasonal/fleet monitoring, forecasting, decision support, and an LLM
        explanation layer) have a clear, honest improvement to measure against.
        """
    )

# ===========================================================================
# Phase 3 – Season Monitoring (preserved exactly)
# ===========================================================================

with season_tab:
    st.subheader("Season Data Loading Status")

    if not config.SEASON_LAPS_FILE.exists():
        st.error(
            "Season data not found. Run `python -m src.data_ingestion season` first to download "
            f"and cache all {config.SEASON_YEAR} race rounds (this loops over every round in "
            "src/config.py:SEASON_ROUNDS and can take a while)."
        )
        st.stop()

    season_laps_df = load_and_clean_season()
    rounds_loaded = sorted(season_laps_df["RoundNumber"].unique())
    st.success(
        f"Loaded {config.SEASON_YEAR} season: {len(rounds_loaded)} rounds, "
        f"{season_laps_df['Driver'].nunique()} drivers, {len(season_laps_df)} laps."
    )
    if len(rounds_loaded) < len(config.SEASON_ROUNDS):
        st.warning(
            f"Only {len(rounds_loaded)} of {len(config.SEASON_ROUNDS)} configured rounds are cached so far "
            "- season trends below reflect rounds loaded to date."
        )

    st.divider()
    st.subheader("Driver Selection")
    season_drivers = sorted(season_laps_df["Driver"].unique())
    default_season_drivers = [d for d in config.COMPARISON_DRIVERS if d in season_drivers] or season_drivers[:2]
    selected_drivers = st.multiselect(
        "Drivers to plot", season_drivers, default=default_season_drivers
    )

    position_df = driver_position_trend(season_laps_df)
    speed_df = speed_trap_trend(season_laps_df)

    st.subheader("Season Finishing Position Trend")
    st.plotly_chart(
        plot_position_trend(position_df[position_df["Driver"].isin(selected_drivers)], "Driver"),
        width="stretch",
    )

    st.subheader("Season Speed Trap Trend")
    st.plotly_chart(
        plot_speed_trap_trend(speed_df[speed_df["Driver"].isin(selected_drivers)]),
        width="stretch",
    )

    st.divider()
    st.subheader("Season KPIs and Asset Health Indicators")
    st.write(
        "PositionTrendSlope: positive = finishing position getting worse across the season "
        "(declining asset health); negative = improving."
    )
    k1, k2 = st.columns(2)
    with k1:
        st.write("Driver KPIs")
        st.dataframe(season_driver_kpis(season_laps_df), width="stretch")
    with k2:
        st.write("Team KPIs")
        st.dataframe(season_team_kpis(season_laps_df), width="stretch")

    st.divider()
    st.subheader("From Race Telemetry to Fleet Monitoring")
    st.markdown(
        """
        - **Race = operating cycle.** Each race is one cycle of the asset (driver/car), the same way each
          well test or compressor run is one operating cycle in production monitoring.
        - **Season = fleet observation window.** Tracking every driver/team across a season is the same
          shape as monitoring a fleet of wells, ESPs, or turbines over the same time horizon.
        - **PositionTrendSlope = long-horizon degradation slope.** Within one race, lap time degrades across a
          tyre stint (Phases 1-2); across a season, finishing position drifting upward is the same kind of
          slow performance decline, just measured cycle-to-cycle instead of lap-to-lap.
        - **Speed trap trend = aggregated sensor KPI.** Rather than every raw telemetry sample, this is a
          single high-frequency reading aggregated per cycle - the same compression engineers apply when
          tracking peak pressure or peak current per well test instead of storing the full waveform.
        """
    )

# ===========================================================================
# Phase 4 – Fleet Monitoring (preserved exactly)
# ===========================================================================

with fleet_tab:
    st.subheader("Fleet Data Loading Status")

    if not config.FLEET_LAPS_FILE.exists():
        st.error(
            "Multi-year data not found. Run `python -m src.data_ingestion fleet` first to download "
            f"and cache {config.FLEET_EVENT_NAME} for every year in "
            "src/config.py:FLEET_YEARS."
        )
        st.stop()

    fleet_laps_df = load_and_clean_fleet()
    years_loaded = sorted(fleet_laps_df["Year"].unique())
    st.success(
        f"Loaded {config.FLEET_EVENT_NAME}: {len(years_loaded)} years ({min(years_loaded)}-{max(years_loaded)}), "
        f"{fleet_laps_df['Driver'].nunique()} distinct drivers, {len(fleet_laps_df)} laps."
    )
    if len(years_loaded) < len(config.FLEET_YEARS):
        st.warning(
            f"Only {len(years_loaded)} of {len(config.FLEET_YEARS)} configured years are cached so far "
            "- fleet trends below reflect years loaded to date."
        )

    st.divider()
    st.subheader("Driver Selection")
    fleet_drivers = sorted(fleet_laps_df["Driver"].unique())
    default_fleet_drivers = [d for d in config.COMPARISON_DRIVERS if d in fleet_drivers] or fleet_drivers[:2]
    selected_fleet_drivers = st.multiselect(
        "Drivers to plot", fleet_drivers, default=default_fleet_drivers
    )

    teammate_df = teammate_gap_by_year(fleet_laps_df)
    field_df = field_average_gap_by_year(fleet_laps_df)
    degradation_df = degradation_by_year(fleet_laps_df)

    st.subheader("Multi-Year Teammate Gap Trend (primary benchmark)")
    st.write(
        "Pace vs. your own teammate that year - same chassis and engine, so this cancels out "
        "almost all of the regulation/car-development effect, leaving mostly driver/strategy "
        "difference. NaN for a driver whose team had no other driver that year."
    )
    st.plotly_chart(
        plot_pace_gap_trend(
            teammate_df[teammate_df["Driver"].isin(selected_fleet_drivers)],
            "Driver", "TeammateGapPct",
            "Multi-Year Teammate Gap Trend by Driver", "Pace vs Teammate (%)",
        ),
        width="stretch",
    )

    with st.expander("Secondary pace benchmarks (Field Average Gap, Fastest Lap Gap)"):
        st.write(
            "Field Average Gap compares against the whole field's average pace that year. "
            "Fastest Lap Gap compares against a single lap, which is one noisy data point "
            "(e.g. an early red flag can leave an artificially slow 'fastest' lap) - kept here "
            "for context, not as the primary benchmark."
        )
        st.plotly_chart(
            plot_pace_gap_trend(
                field_df[field_df["Driver"].isin(selected_fleet_drivers)],
                "Driver", "FieldAverageGapPct",
                "Multi-Year Field Average Gap Trend by Driver", "Pace vs Field Average (%)",
            ),
            width="stretch",
        )

    st.subheader("Multi-Year Tyre Degradation Trend")
    st.plotly_chart(
        plot_degradation_by_year(degradation_df[degradation_df["Driver"].isin(selected_fleet_drivers)]),
        width="stretch",
    )

    st.divider()
    st.subheader("Reliability and Consistency")
    st.write(
        "RaceCompletionRatePct is a reliability *proxy* (laps completed vs. the most anyone "
        "completed that year) - FastF1's official DNF/retirement-reason data isn't reliably "
        "available for every season, so this stands in for it."
    )

    st.divider()
    st.subheader("Benchmarking Table and Year-over-Year Shift")
    st.write(
        "Shift: Declined / Improved / Stable, comparing each driver's TeammateGapPct this year "
        "against the previous year they appeared in this race (falls back to 'N/A' with no "
        "teammate data)."
    )
    st.dataframe(
        fleet_benchmark_table(fleet_laps_df).query("Driver in @selected_fleet_drivers"),
        width="stretch",
    )

    st.divider()
    st.subheader("From Multi-Year Comparison to Fleet Benchmarking")
    st.markdown(
        """
        - **Year = long observation window.** Comparing the same race across years is the same shape as
          comparing a fleet of wells, ESPs, or turbines across multiple years of operation.
        - **Teammate gap = same-asset-class benchmarking.** Comparing a driver to their own teammate (same
          chassis, same engine) is the same idea as comparing two nominally identical pumps on the same
          site - it cancels out generation/regulation effects, leaving the difference that actually matters.
        - **RaceCompletionRatePct = reliability proxy.** Standing in for true DNF/mechanical-retirement
          classification (not reliably available from FastF1 for every season), the same way a production
          system might track "operating hours vs. expected hours" when failure-cause logging is incomplete.
        - **PitStopCount / AvgPitStopRecoveryLaps = operational efficiency.** How often, and how costly,
          an intervention was - directly analogous to maintenance-stop frequency and recovery time for an
          industrial asset.
        - **Year-over-year Shift = structural change detector.** A single bad season is normal variation;
          a sustained shift in TeammateGapPct across consecutive years is the long-horizon equivalent of
          flagging that an asset's baseline performance has genuinely changed, not just had a noisy day.
        """
    )

# ===========================================================================
# Phase 5 – Predictive Analytics (preserved exactly)
# ===========================================================================

with predictive_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.stop()

    predictive_laps_df, _, _ = load_and_clean_all()
    st.success(
        f"Using {config.SEASON_YEAR} {config.EVENT_NAME} (single race): "
        f"{predictive_laps_df['Driver'].nunique()} drivers, {len(predictive_laps_df)} laps."
    )
    st.write(
        "Scope (v1): pooled across all drivers, without driver identity as a feature, so the "
        "model has to learn the general tyre-degradation pattern rather than memorize each "
        "driver's baseline pace. The first lap of every stint has no lap history yet, so it's "
        "excluded from training/evaluation."
    )

    st.divider()
    st.subheader("Lap Time Forecast: Model Comparison")
    st.write(
        "All models are evaluated on the same chronological split (test laps happen later in "
        "the race than anything trained on) - this is a forecast, not interpolation."
    )

    results_df, artifacts = compare_models(predictive_laps_df)
    st.dataframe(results_df, width="stretch")
    st.plotly_chart(plot_model_comparison(results_df), width="stretch")
    st.caption(
        "If a baseline (Naive lag-1 / Mean) beats the trained models here, that's a genuine "
        "finding, not a bug - lap-to-lap correlation is high, so a dumb baseline can be hard "
        "to beat at a 1-lap-ahead horizon. See README for the actual result on this race."
    )

    selected_model = st.selectbox("Model to inspect", list(artifacts["predictions"].keys()))
    st.plotly_chart(
        plot_predicted_vs_actual(
            artifacts["test_df"], artifacts["y_test"], artifacts["predictions"][selected_model], selected_model
        ),
        width="stretch",
    )

    st.divider()
    st.subheader("Explain Model Outputs")
    e1, e2 = st.columns(2)
    with e1:
        st.write("Linear Regression coefficients")
        st.plotly_chart(
            plot_feature_importance(
                explain_linear_model(artifacts["models"]["Linear Regression"], list(artifacts["X_train"].columns)),
                "Coefficient", "Linear Regression Coefficients",
            ),
            width="stretch",
        )
    with e2:
        st.write("Random Forest feature importances")
        st.plotly_chart(
            plot_feature_importance(
                explain_tree_model(artifacts["models"]["Random Forest"], list(artifacts["X_train"].columns)),
                "Importance", "Random Forest Feature Importances",
            ),
            width="stretch",
        )

    st.divider()
    st.subheader("Degradation Forecast")
    forecast_drivers = sorted(predictive_laps_df["Driver"].unique())
    f1, f2, f3 = st.columns(3)
    with f1:
        forecast_driver = st.selectbox("Driver", forecast_drivers, index=0)
    driver_stints = sorted(predictive_laps_df.loc[predictive_laps_df["Driver"] == forecast_driver, "Stint"].dropna().unique())
    with f2:
        forecast_stint = st.selectbox("Stint", driver_stints, index=0)
    with f3:
        laps_ahead = st.slider("Laps ahead", min_value=1, max_value=10, value=5)

    observed = predictive_laps_df[
        (predictive_laps_df["Driver"] == forecast_driver) & (predictive_laps_df["Stint"] == forecast_stint)
    ].copy()
    observed["StintLap"] = range(1, len(observed) + 1)
    forecast = forecast_stint_degradation(predictive_laps_df, forecast_driver, int(forecast_stint), laps_ahead)
    if forecast.empty:
        st.info("Not enough laps in this stint to fit a degradation slope.")
    else:
        st.plotly_chart(plot_degradation_forecast(observed, forecast), width="stretch")

    st.divider()
    st.subheader("Degradation Risk Scores")
    st.write(
        "ProjectedIncreaseSeconds: forecasted lap-time increase over the next few laps if the "
        "current degradation trend continues. RiskCategory (Low/Medium/High) is assigned from "
        "this race's own distribution of projected increases (tertiles), not a fixed constant."
    )
    st.dataframe(degradation_risk_scores(predictive_laps_df), width="stretch")

    st.divider()
    st.subheader("From Forecasting to Decision Support")
    st.markdown(
        """
        - **Lap-time forecast = next-reading prediction.** Predicting a future sensor reading from
          recent operating history is the same shape whether the sensor is lap time or pump vibration.
        - **Honest baselines first.** A naive "next reading = last reading" forecast and a flat
          "always predict the average" forecast are evaluated alongside the trained models - if a
          model can't beat them, that's worth knowing before trusting it operationally.
        - **Degradation forecast = early-warning extrapolation.** Projecting the current wear trend
          forward a few cycles is the same logic as forecasting when a component crosses a wear
          threshold, using only the data already collected.
        - **Risk score, not a recommendation.** Phase 5 stops at scoring and explaining risk;
          turning that into an actual maintenance/pit-stop recommendation is Phase 6+ territory.
        """
    )

# ===========================================================================
# Phase 6 – Operational Assistant (preserved exactly)
# ===========================================================================

with assistant_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.stop()

    assistant_laps_df, _, _ = load_and_clean_all()
    st.success(
        f"Using {config.SEASON_YEAR} {config.EVENT_NAME} (single race): "
        f"{assistant_laps_df['Driver'].nunique()} drivers, {len(assistant_laps_df)} laps."
    )
    st.write(
        f"Runs against a **local** Ollama server (`{config.OLLAMA_MODEL}` at {config.OLLAMA_BASE_URL}), "
        "not a hosted API - the same approach would let this run against confidential real "
        "ESP/SCADA data without sending anything to a third-party provider. Start Ollama with "
        "`ollama serve` if you see a connection error below."
    )

    st.divider()
    st.subheader("Ask a Question")
    st.write(
        "The assistant only answers questions it can parse into a **driver code + lap number** "
        "(e.g. \"Why did VER lose performance after lap 32?\"). If it can't identify both, or "
        "finds no matching lap, it says so and never calls the LLM - every answer it does give "
        "is grounded in retrieved telemetry, not invented."
    )

    example_questions = [
        "Why did BOT lose performance at lap 13?",
        "Why did HUL lose performance at lap 2?",
        "Why did VER lose performance after lap 18?",
        "How was the race overall?",  # deliberately unanswerable - no driver/lap to parse
    ]
    question = st.selectbox("Example question (or type your own below)", example_questions)
    custom_question = st.text_input("Or type your own question", value="")
    final_question = custom_question.strip() or question

    if st.button("Ask"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                result = answer_question(assistant_laps_df, final_question)
            except Exception as exc:
                st.error(
                    f"Could not reach the local Ollama server: {exc}\n\n"
                    "Make sure Ollama is running (`ollama serve`) and the model in "
                    f"src/config.py:OLLAMA_MODEL (`{config.OLLAMA_MODEL}`) is pulled "
                    f"(`ollama pull {config.OLLAMA_MODEL}`)."
                )
                result = None

        if result is not None:
            st.write(f"**Parsed:** driver = `{result['parsed']['driver']}`, lap = `{result['parsed']['lap_number']}`")

            if not result["grounded"]:
                st.warning(result["answer"])
            else:
                with st.expander("Evidence retrieved (what the LLM was actually given)"):
                    st.code(result["evidence_summary"])
                st.markdown("**Answer**")
                st.write(result["answer"])

                citation_check = result["citation_check"]
                if not citation_check["has_citations"]:
                    st.error("This answer cites no lap numbers at all - it isn't grounded in the evidence.")
                elif not citation_check["all_citations_valid"]:
                    st.error(
                        f"This answer cites lap(s) {citation_check['invalid_laps']} that aren't in the "
                        "retrieved evidence - treat it as unverified."
                    )
                else:
                    st.success(f"All cited laps ({citation_check['cited_laps']}) match the retrieved evidence.")

                if result["speculative_phrases"]:
                    st.warning(
                        "Flagged phrasing that may go beyond the evidence (heuristic check, not a hard "
                        f"block - review the answer above): {', '.join(result['speculative_phrases'])}"
                    )

    st.divider()
    st.subheader("From Evidence Retrieval to Trustworthy Explanations")
    st.markdown(
        """
        - **Retrieve before generate = audit trail.** The evidence block shown above is exactly
          what the LLM was given - nothing more. For an operational decision, being able to show
          *why* an assistant said what it said is as important as the answer itself.
        - **No driver/lap parsed = no LLM call.** Refusing to guess when the question can't be
          grounded is what "no hallucinated conclusions" means in practice, not just a prompt
          instruction hoping the model behaves.
        - **Local LLM = confidentiality.** Running against Ollama instead of a hosted API is the
          same architecture you'd need for real, proprietary ESP/SCADA data that can't leave the
          plant network.
        """
    )

# ===========================================================================
# Phase 7 – Decision Support (preserved exactly)
# ===========================================================================

with decision_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.stop()

    decision_laps_df, _, _ = load_and_clean_all()
    st.success(
        f"Using {config.SEASON_YEAR} {config.EVENT_NAME} (single race): "
        f"{decision_laps_df['Driver'].nunique()} drivers, {len(decision_laps_df)} laps."
    )
    st.write(
        f"A pit/maintenance window is recommended once the existing degradation forecast "
        f"(Phase 5) projects lap time crossing **{config.DECISION_PIT_THRESHOLD_PCT}% above "
        f"this stint's own starting pace**, within the next {config.DECISION_MAX_HORIZON_LAPS} "
        "laps. This reuses the same linear fit as the Predictive Analytics tab - it doesn't "
        "add a new model, it turns the existing forecast into a decision."
    )

    recommendations_table = build_recommendations_table(decision_laps_df)

    st.divider()
    st.subheader("Recommended Pit Windows")
    st.plotly_chart(plot_pit_recommendations(recommendations_table), width="stretch")

    st.divider()
    st.subheader("All Recommendations")
    st.write(
        "RiskCategory (Phase 5, relative to this race's other stints) and RecommendedAction "
        "(Phase 7, an absolute threshold) answer different questions and can legitimately "
        "disagree - e.g. a stint can rank High risk relative to its peers while still being "
        "many laps from this fixed threshold."
    )
    decision_drivers = sorted(recommendations_table["Driver"].unique()) if not recommendations_table.empty else []
    selected_decision_drivers = st.multiselect("Filter by driver", decision_drivers, default=decision_drivers)
    if selected_decision_drivers:
        st.dataframe(
            recommendations_table[recommendations_table["Driver"].isin(selected_decision_drivers)],
            width="stretch",
        )
    else:
        st.dataframe(recommendations_table, width="stretch")

    st.divider()
    st.subheader("From Forecast to Recommendation")
    st.markdown(
        """
        - **A forecast becomes a decision once it crosses a threshold.** Phase 5 already forecasts
          how lap time will evolve; Phase 7 adds the one extra step of saying *when that forecast
          implies an action is needed* - the same step that turns a vibration trend into a
          maintenance work order.
        - **The threshold is a configurable constant, not derived from the data** (see
          `src/config.py:DECISION_PIT_THRESHOLD_PCT`) - simple and transparent, but it means the
          recommendation timing depends on a number chosen in advance, not statistically fitted.
        - **Validated against real strategy**: for several drivers (e.g. STR, ZHO, RUS in the
          2024 Bahrain race), the model's projected crossing lap lands a few laps after their
          actual pit lap - consistent with teams pitting proactively before a hard performance
          cliff, not reactively after one.
        """
    )

# ===========================================================================
# Phase 9 – Operational Context (preserved exactly)
# ===========================================================================

with context_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.stop()

    st.write(
        "Telemetry alone doesn't explain *why* it changed. This tab pairs the existing "
        "telemetry with the operational context behind it - tyre state, track status, "
        "weather, and race control events - all read through `src/context_engine.py`. "
        "No part of this logic lives in this dashboard file; it only calls that module."
    )

    context_data = align_context_to_session(load_context())
    context_laps_df = context_data["laps"]

    st.divider()
    st.subheader("Select a Moment")

    context_drivers = sorted(context_laps_df["Driver"].unique())
    default_context_driver = (
        config.COMPARISON_DRIVERS[0] if config.COMPARISON_DRIVERS[0] in context_drivers else context_drivers[0]
    )
    col1, col2 = st.columns(2)
    with col1:
        context_driver = st.selectbox(
            "Driver", context_drivers, index=context_drivers.index(default_context_driver), key="context_driver"
        )
    driver_lap_rows = context_laps_df[context_laps_df["Driver"] == context_driver].sort_values("LapNumber")
    available_laps = driver_lap_rows["LapNumber"].dropna().astype(int).tolist()
    with col2:
        context_lap = st.select_slider("Lap", options=available_laps, value=available_laps[len(available_laps) // 2])

    session_time_seconds = float(
        driver_lap_rows.loc[driver_lap_rows["LapNumber"] == context_lap, "LapStartTimeSeconds"].iloc[0]
    )

    context_now = get_context_at_timestamp(context_data, session_time_seconds, context_driver)
    context_changes = calculate_context_changes(context_data, session_time_seconds, context_driver)

    st.divider()
    st.subheader("Operational Context")

    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        st.metric("Tyre", f"{context_now.get('Compound', 'Unknown')}")
        st.metric("Tyre Age", f"{context_now.get('TyreLife', 'Unknown')} laps")
    with card_col2:
        st.metric("Track", context_now.get("TrackStatus", "Unknown"))
        st.metric("Air Temp", f"{context_now.get('AirTemp', float('nan')):.1f}°C" if "AirTemp" in context_now else "—")
    with card_col3:
        st.metric("Track Temp", f"{context_now.get('TrackTemp', float('nan')):.1f}°C" if "TrackTemp" in context_now else "—")
        st.metric("Recent Event", context_now.get("RecentEvent", "No significant events"))

    st.divider()
    st.subheader("Context Trends")

    trend_cols = st.columns(3)
    trend_targets = [
        ("TrackTemp", "Track Temperature", "°C"),
        ("WindSpeed", "Wind Speed", " km/h"),
        ("TyreLife", "Tyre Life", " laps"),
    ]
    for column, (variable, label, unit) in zip(trend_cols, trend_targets):
        change = context_changes.get(variable)
        with column:
            if change is None:
                st.write(f"**{label}**: not enough history yet")
            else:
                st.write(
                    f"**{label}**  \n"
                    f"Current: {change['current']:.1f}{unit}  \n"
                    f"Previous: {change['previous']:.1f}{unit}  \n"
                    f"Trend: {change['trend']}"
                )

    st.divider()
    st.subheader("Context-Aware Explanation")
    st.write(
        "Pick what telemetry appeared to show at this moment; the explanation below is "
        "generated by `generate_context_summary()` from the real context above, not inferred "
        "by an LLM or model."
    )
    observed_effect_label = st.selectbox(
        "What did telemetry show?",
        ["No specific signal", "Lower speed than expected", "Lap time increased", "Speed anomaly detected"],
        key="observed_effect",
    )
    observed_effect_map = {
        "No specific signal": None,
        "Lower speed than expected": OBSERVED_LOW_SPEED,
        "Lap time increased": OBSERVED_LAPTIME_INCREASE,
        "Speed anomaly detected": OBSERVED_SPEED_ANOMALY,
    }
    context_summary = generate_context_summary(
        context_now, context_changes, observed_effect_map[observed_effect_label]
    )

    status_color = {"green": ":green", "amber": ":orange", "red": ":red"}[context_summary["color"]]
    st.markdown(f"**Context Status:** {status_color}[{context_summary['status']}]")
    if context_summary["confidence"] is not None:
        st.markdown(f"**Context Confidence:** {context_summary['confidence']}")
    for sentence in context_summary["interpretation"]:
        st.write(f"- {sentence}")

    st.divider()
    st.subheader("Why Operational Context Matters")
    st.markdown(
        """
        - **Telemetry alone is insufficient for decision support.** The same drop in speed can
          mean tyre degradation, a yellow flag, or an unexplained anomaly - the difference is
          entirely in the operational context, not the speed trace itself.
        - **This panel adds no new model.** Every value is read directly from FastF1 data
          (weather, lap/tyre/track-status, race control messages) already cached by Phase 1's
          ingestion, or linearly interpolated/diffed between two real samples - see
          `src/context_engine.py`.
        - **The Context Engine is domain-independent by construction.** A future ESP/SCADA
          implementation would point the same five functions (`load_context`,
          `align_context_to_session`, `get_context_at_timestamp`, `calculate_context_changes`,
          `generate_context_summary`) at different data sources; no other module in this
          dashboard or pipeline would need to change.
        """
    )

# ===========================================================================
# Phase 10 – Health Assessment (preserved exactly)
# ===========================================================================

with health_tab:
    st.subheader("Data Loading Status")

    required_files = [config.LAPS_FILE, config.WEATHER_FILE, config.TELEMETRY_FILE]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            "Processed data not found. Run `python -m src.data_ingestion` first to download "
            f"and cache the {config.SEASON_YEAR} {config.EVENT_NAME} ({config.SESSION_NAME}) session."
        )
        st.stop()

    st.write(
        "Phase 2's z-score anomaly flag is context-blind: it raises an alarm on every lap that "
        "deviates from a driver's own baseline, regardless of *why*. This tab re-scores each of "
        "those alarms through Phase 9's context engine via `src/health_assessment.py` - no new "
        "model, just `anomaly_detection.py` + `context_engine.py` combined."
    )

    health_laps_df, _, _ = load_and_clean_all()
    health_context = align_context_to_session(load_context())
    assessed_anomalies = assess_anomaly_health(health_laps_df, health_context)
    summary = health_summary(assessed_anomalies)

    st.divider()
    st.subheader("Noise Reduction")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flagged by z-score", summary["TotalFlagged"])
    col2.metric("Explained", summary["Explained"])
    col3.metric("Partially Explained", summary["PartiallyExplained"])
    col4.metric("Unexplained", summary["Unexplained"], help="The ones actually worth investigating")

    st.metric("Noise Reduction", f"{summary['NoiseReductionPct']}%")
    st.write(
        "**Validated on this real race**: every one of the 44 anomalies the z-score detector "
        "flags in the 2024 Bahrain race turns out to be explained - either a pit out-lap on a "
        "fresh tyre, or the opening laps under a Yellow flag. Context doesn't add new findings "
        "here so much as it removes false alarms: a 44-row anomaly table becomes a 0-row "
        "action list, which is the actual point of pairing telemetry with context."
    )

    st.divider()
    st.subheader("Assessed Anomalies")

    health_filter = st.multiselect(
        "Filter by health status",
        [HEALTH_EXPLAINED, HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED],
        default=[HEALTH_EXPLAINED, HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED],
    )
    display_columns = ["Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "LapTimeZScore", "HealthStatus", "Confidence", "Explanation"]
    if not assessed_anomalies.empty:
        st.dataframe(
            assessed_anomalies[assessed_anomalies["HealthStatus"].isin(health_filter)][display_columns],
            width="stretch",
        )
    else:
        st.write("No anomalies flagged for this dataset.")

    st.divider()
    st.subheader("From Detection to Health Assessment")
    st.markdown(
        """
        - **This is the missing middle step from Phase 9's own architecture diagram**:
          `Telemetry + Operational Context -> Health Assessment -> Recommendation`. Phase 9 built
          the context engine; Phase 10 is the first thing that actually consumes it to re-score
          an existing signal, rather than just displaying context side by side with telemetry.
        - **Only slow-lap anomalies are assessed.** A lap unusually *fast* isn't a health
          concern - there's nothing to explain or escalate, so the negative-z-score half of
          Phase 2's anomaly table is intentionally excluded here.
        - **A real gap was found and fixed while building this**: the context engine's
          explanation rules only matched a "low speed" signal, not a "lap time increased" one -
          even though a z-score anomaly *is* a lap-time signal. Re-scoring would have silently
          downgraded every real pit out-lap to "Partially Explained" instead of "Explained" had
          this not been caught (see `src/context_engine.py:PERFORMANCE_DROP_EFFECTS`).
        """
    )
