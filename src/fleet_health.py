"""Per-driver health status for the industrial fleet overview.

Deterministic rules only, built entirely from existing analytics outputs.
No new statistical models or synthetic data.

Health states:
    Healthy  (Green)  – no actionable anomalies, no pit recommendation,
                        degradation risk Low or absent.
    Warning  (Amber)  – at least one partially-explained anomaly, OR an
                        active pit recommendation, OR High/Medium risk.
    Critical (Red)    – at least one unexplained anomaly OR a "Pit now"
                        recommendation.

The severity ordering is:
    Healthy < Warning < Critical

Each driver's status is the *worst* of contributions from anomalies,
recommendations, and risk scores.
"""

import pandas as pd

from src.health_assessment import HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED

HEALTH_HEALTHY = "Healthy"
HEALTH_WARNING = "Warning"
HEALTH_CRITICAL = "Critical"

# Fallback text when a driver is Warning/Critical but no pit recommendation
# exists for the current stint — avoids the internally-inconsistent state of
# a Warning/Critical badge paired with "No action needed".
_ANOMALY_FALLBACK_ACTION = {
    HEALTH_UNEXPLAINED: "Investigate — unexplained anomaly detected",
    HEALTH_PARTIALLY_EXPLAINED: "Monitor — partially explained anomaly",
}

# Deterministic severity ordering – higher index wins
_SEVERITY_ORDER = {HEALTH_HEALTHY: 0, HEALTH_WARNING: 1, HEALTH_CRITICAL: 2}

# Displayed with Streamlit's :green / :orange / :red colour helpers
HEALTH_COLOR_LABEL = {
    HEALTH_HEALTHY: "green",
    HEALTH_WARNING: "orange",
    HEALTH_CRITICAL: "red",
}


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _SEVERITY_ORDER.get(s, 0))


def compute_fleet_health(
    laps_df: pd.DataFrame,
    assessed_anomalies: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """One row per driver: health status, worst anomaly, active recommendation.

    Parameters
    ----------
    laps_df:
        Cleaned lap data from ``load_and_clean_all()``.
    assessed_anomalies:
        Output of ``assess_anomaly_health()``.  Pass an empty DataFrame if
        health assessment is unavailable.
    recommendations:
        Output of ``build_recommendations_table()``.  Pass an empty DataFrame
        if decision support is unavailable.

    Returns
    -------
    pd.DataFrame with one row per driver and columns:
        Driver, HealthStatus, WorstAnomaly, ActiveRecommendation,
        RiskCategory, CurrentCompound, TyreLife, CurrentLap, Stint.
    """
    drivers = sorted(laps_df["Driver"].unique())
    records = []

    for driver in drivers:
        status = HEALTH_HEALTHY
        worst_anomaly: str | None = None
        active_recommendation: str | None = None
        risk_category = "Low"

        # --- Anomaly contribution -------------------------------------------
        if not assessed_anomalies.empty and "HealthStatus" in assessed_anomalies.columns:
            driver_anomalies = assessed_anomalies[assessed_anomalies["Driver"] == driver]
            for hs in driver_anomalies["HealthStatus"]:
                if hs == HEALTH_UNEXPLAINED:
                    status = _worst(status, HEALTH_CRITICAL)
                    worst_anomaly = HEALTH_UNEXPLAINED
                elif hs == HEALTH_PARTIALLY_EXPLAINED:
                    status = _worst(status, HEALTH_WARNING)
                    if worst_anomaly != HEALTH_UNEXPLAINED:
                        worst_anomaly = HEALTH_PARTIALLY_EXPLAINED

        # --- Recommendation / risk contribution ------------------------------
        if not recommendations.empty and "Driver" in recommendations.columns:
            driver_recs = recommendations[recommendations["Driver"] == driver]
            if not driver_recs.empty:
                risk_values = driver_recs.get("RiskCategory", pd.Series(dtype=str)).tolist()
                if "High" in risk_values:
                    risk_category = "High"
                    status = _worst(status, HEALTH_WARNING)
                elif "Medium" in risk_values:
                    risk_category = "Medium"
                    status = _worst(status, HEALTH_WARNING)

                for action in driver_recs.get("RecommendedAction", pd.Series(dtype=str)).tolist():
                    if action == "Pit now":
                        status = _worst(status, HEALTH_CRITICAL)
                        active_recommendation = action
                        break
                    elif str(action).startswith("Pit within"):
                        status = _worst(status, HEALTH_WARNING)
                        if active_recommendation is None:
                            active_recommendation = action

        # --- Latest tyre / lap state ----------------------------------------
        driver_laps = laps_df[laps_df["Driver"] == driver]
        if not driver_laps.empty:
            latest = driver_laps.sort_values("LapNumber").iloc[-1]
            compound = latest.get("Compound", "Unknown")
            tyre_life = latest.get("TyreLife")
            current_lap = latest.get("LapNumber")
            stint = latest.get("Stint")
        else:
            compound, tyre_life, current_lap, stint = "Unknown", None, None, None

        if active_recommendation is None and status != HEALTH_HEALTHY:
            if worst_anomaly in _ANOMALY_FALLBACK_ACTION:
                active_recommendation = _ANOMALY_FALLBACK_ACTION[worst_anomaly]
            elif risk_category in ("High", "Medium"):
                active_recommendation = f"Monitor — {risk_category} degradation risk"

        records.append({
            "Driver": driver,
            "HealthStatus": status,
            "WorstAnomaly": worst_anomaly or "None",
            "ActiveRecommendation": active_recommendation or "No action needed",
            "RiskCategory": risk_category,
            "CurrentCompound": compound,
            "TyreLife": int(tyre_life) if pd.notna(tyre_life) else None,
            "CurrentLap": int(current_lap) if pd.notna(current_lap) else None,
            "Stint": int(stint) if pd.notna(stint) else None,
        })

    return pd.DataFrame(records)


def fleet_kpis(health_df: pd.DataFrame, laps_df: pd.DataFrame) -> dict:
    """Aggregate fleet-level KPIs for the overview dashboard header.

    Returns a dict with keys:
        DriverCount, TotalLaps, Healthy, Warning, Critical, ActiveAlerts.
    """
    if health_df.empty:
        return {
            "DriverCount": 0,
            "TotalLaps": len(laps_df),
            "Healthy": 0,
            "Warning": 0,
            "Critical": 0,
            "ActiveAlerts": 0,
        }
    healthy = int((health_df["HealthStatus"] == HEALTH_HEALTHY).sum())
    warning = int((health_df["HealthStatus"] == HEALTH_WARNING).sum())
    critical = int((health_df["HealthStatus"] == HEALTH_CRITICAL).sum())
    return {
        "DriverCount": len(health_df),
        "TotalLaps": len(laps_df),
        "Healthy": healthy,
        "Warning": warning,
        "Critical": critical,
        "ActiveAlerts": warning + critical,
    }
