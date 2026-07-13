import pandas as pd
import pytest

from src.fleet_health import (
    HEALTH_CRITICAL,
    HEALTH_HEALTHY,
    HEALTH_WARNING,
    compute_fleet_health,
    fleet_kpis,
)
from src.health_assessment import HEALTH_EXPLAINED, HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED


@pytest.fixture
def two_driver_laps():
    """VER and LEC, 5 laps each."""
    rows = []
    for driver in ["VER", "LEC"]:
        for lap in range(1, 6):
            rows.append({
                "Driver": driver,
                "LapNumber": lap,
                "LapTimeSeconds": 90.0 + (lap - 1) * 0.3,
                "Compound": "SOFT",
                "TyreLife": float(lap),
                "Stint": 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def empty_anomalies():
    return pd.DataFrame(columns=["Driver", "LapNumber", "HealthStatus"])


@pytest.fixture
def empty_recommendations():
    return pd.DataFrame(columns=["Driver", "Stint", "Compound", "RiskCategory", "RecommendedAction"])


def test_compute_fleet_health_returns_dataframe(two_driver_laps, empty_anomalies, empty_recommendations):
    result = compute_fleet_health(two_driver_laps, empty_anomalies, empty_recommendations)
    assert isinstance(result, pd.DataFrame)


def test_compute_fleet_health_one_row_per_driver(two_driver_laps, empty_anomalies, empty_recommendations):
    result = compute_fleet_health(two_driver_laps, empty_anomalies, empty_recommendations)
    assert set(result["Driver"]) == {"VER", "LEC"}
    assert len(result) == 2


def test_required_columns_present(two_driver_laps, empty_anomalies, empty_recommendations):
    result = compute_fleet_health(two_driver_laps, empty_anomalies, empty_recommendations)
    for col in ["Driver", "HealthStatus", "WorstAnomaly", "ActiveRecommendation", "RiskCategory",
                "CurrentCompound", "TyreLife", "CurrentLap", "Stint"]:
        assert col in result.columns


def test_no_issues_gives_healthy(two_driver_laps, empty_anomalies, empty_recommendations):
    result = compute_fleet_health(two_driver_laps, empty_anomalies, empty_recommendations)
    assert (result["HealthStatus"] == HEALTH_HEALTHY).all()


def test_unexplained_anomaly_gives_critical(two_driver_laps, empty_recommendations):
    anomalies = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [3],
        "HealthStatus": [HEALTH_UNEXPLAINED],
    })
    result = compute_fleet_health(two_driver_laps, anomalies, empty_recommendations)
    ver_row = result[result["Driver"] == "VER"]
    assert ver_row.iloc[0]["HealthStatus"] == HEALTH_CRITICAL
    # LEC is unaffected
    lec_row = result[result["Driver"] == "LEC"]
    assert lec_row.iloc[0]["HealthStatus"] == HEALTH_HEALTHY


def test_unexplained_anomaly_with_no_recommendation_is_not_no_action_needed(two_driver_laps, empty_recommendations):
    """A Critical driver escalated purely by an anomaly must not claim 'No action needed'."""
    anomalies = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [3],
        "HealthStatus": [HEALTH_UNEXPLAINED],
    })
    result = compute_fleet_health(two_driver_laps, anomalies, empty_recommendations)
    ver_row = result[result["Driver"] == "VER"]
    assert ver_row.iloc[0]["ActiveRecommendation"] != "No action needed"
    assert "Investigate" in ver_row.iloc[0]["ActiveRecommendation"]


def test_partially_explained_anomaly_gives_warning(two_driver_laps, empty_recommendations):
    anomalies = pd.DataFrame({
        "Driver": ["LEC"],
        "LapNumber": [2],
        "HealthStatus": [HEALTH_PARTIALLY_EXPLAINED],
    })
    result = compute_fleet_health(two_driver_laps, anomalies, empty_recommendations)
    lec_row = result[result["Driver"] == "LEC"]
    assert lec_row.iloc[0]["HealthStatus"] == HEALTH_WARNING


def test_partially_explained_anomaly_with_no_recommendation_is_not_no_action_needed(two_driver_laps, empty_recommendations):
    anomalies = pd.DataFrame({
        "Driver": ["LEC"],
        "LapNumber": [2],
        "HealthStatus": [HEALTH_PARTIALLY_EXPLAINED],
    })
    result = compute_fleet_health(two_driver_laps, anomalies, empty_recommendations)
    lec_row = result[result["Driver"] == "LEC"]
    assert lec_row.iloc[0]["ActiveRecommendation"] != "No action needed"
    assert "Monitor" in lec_row.iloc[0]["ActiveRecommendation"]


def test_explained_anomaly_does_not_degrade_health(two_driver_laps, empty_recommendations):
    anomalies = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [1],
        "HealthStatus": [HEALTH_EXPLAINED],
    })
    result = compute_fleet_health(two_driver_laps, anomalies, empty_recommendations)
    ver_row = result[result["Driver"] == "VER"]
    # Explained anomaly should not trigger Warning or Critical
    assert ver_row.iloc[0]["HealthStatus"] == HEALTH_HEALTHY


def test_pit_now_recommendation_gives_critical(two_driver_laps, empty_anomalies):
    recs = pd.DataFrame({
        "Driver": ["VER"],
        "Stint": [1],
        "Compound": ["SOFT"],
        "RiskCategory": ["High"],
        "RecommendedAction": ["Pit now"],
    })
    result = compute_fleet_health(two_driver_laps, empty_anomalies, recs)
    ver_row = result[result["Driver"] == "VER"]
    assert ver_row.iloc[0]["HealthStatus"] == HEALTH_CRITICAL


def test_pit_within_recommendation_gives_warning(two_driver_laps, empty_anomalies):
    recs = pd.DataFrame({
        "Driver": ["LEC"],
        "Stint": [1],
        "Compound": ["SOFT"],
        "RiskCategory": ["Medium"],
        "RecommendedAction": ["Pit within 3 laps (by stint lap 8)"],
    })
    result = compute_fleet_health(two_driver_laps, empty_anomalies, recs)
    lec_row = result[result["Driver"] == "LEC"]
    assert lec_row.iloc[0]["HealthStatus"] == HEALTH_WARNING


def test_high_risk_gives_warning_not_critical(two_driver_laps, empty_anomalies):
    recs = pd.DataFrame({
        "Driver": ["VER"],
        "Stint": [1],
        "Compound": ["SOFT"],
        "RiskCategory": ["High"],
        "RecommendedAction": ["No action needed within the next 10 laps"],
    })
    result = compute_fleet_health(two_driver_laps, empty_anomalies, recs)
    ver_row = result[result["Driver"] == "VER"]
    assert ver_row.iloc[0]["HealthStatus"] == HEALTH_WARNING
    # The stint's own recommendation text is "No action needed within the next
    # 10 laps", but the driver is still Warning (High risk) — the fallback
    # must not collapse to the bare "No action needed" the driver-level badge
    # would contradict.
    assert ver_row.iloc[0]["ActiveRecommendation"] != "No action needed"
    assert "High degradation risk" in ver_row.iloc[0]["ActiveRecommendation"]


def test_unexplained_beats_warning(two_driver_laps):
    """Worst severity wins: an unexplained anomaly escalates past Warning."""
    anomalies = pd.DataFrame({
        "Driver": ["VER", "VER"],
        "LapNumber": [1, 3],
        "HealthStatus": [HEALTH_PARTIALLY_EXPLAINED, HEALTH_UNEXPLAINED],
    })
    recs = pd.DataFrame(columns=["Driver", "Stint", "Compound", "RiskCategory", "RecommendedAction"])
    result = compute_fleet_health(two_driver_laps, anomalies, recs)
    ver_row = result[result["Driver"] == "VER"]
    assert ver_row.iloc[0]["HealthStatus"] == HEALTH_CRITICAL


def test_fleet_kpis_counts(two_driver_laps):
    health_df = pd.DataFrame({
        "Driver": ["VER", "LEC", "NOR"],
        "HealthStatus": [HEALTH_HEALTHY, HEALTH_WARNING, HEALTH_CRITICAL],
    })
    kpis = fleet_kpis(health_df, two_driver_laps)
    assert kpis["DriverCount"] == 3
    assert kpis["TotalLaps"] == len(two_driver_laps)
    assert kpis["Healthy"] == 1
    assert kpis["Warning"] == 1
    assert kpis["Critical"] == 1
    assert kpis["ActiveAlerts"] == 2


def test_fleet_kpis_empty(two_driver_laps):
    health_df = pd.DataFrame(columns=["Driver", "HealthStatus"])
    kpis = fleet_kpis(health_df, two_driver_laps)
    assert kpis["DriverCount"] == 0
    assert kpis["Healthy"] == 0
    assert kpis["ActiveAlerts"] == 0
