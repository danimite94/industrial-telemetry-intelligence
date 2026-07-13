"""Tests for Phase 9A UI components — pure HTML/Plotly generators, no analytics."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from components.asset_card import asset_card_html
from components.event_timeline import event_timeline_html
from components.health_badge import health_badge_html
from components.sparkline import sparkline_figure
from components.status_banner import status_banner_html


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def healthy_row():
    return pd.Series({
        "Driver": "VER", "HealthStatus": "Healthy",
        "CurrentCompound": "SOFT", "TyreLife": 8.0,
        "CurrentLap": 23, "Stint": 2,
        "RiskCategory": "Low", "ActiveRecommendation": "No action needed",
    })


@pytest.fixture
def warning_row():
    return pd.Series({
        "Driver": "LEC", "HealthStatus": "Warning",
        "CurrentCompound": "HARD", "TyreLife": 20.0,
        "CurrentLap": 40, "Stint": 3,
        "RiskCategory": "High", "ActiveRecommendation": "Pit within 3 laps",
    })


@pytest.fixture
def critical_row():
    return pd.Series({
        "Driver": "SAI", "HealthStatus": "Critical",
        "CurrentCompound": "MEDIUM", "TyreLife": None,
        "CurrentLap": None, "Stint": None,
        "RiskCategory": None, "ActiveRecommendation": "Pit now",
    })


@pytest.fixture
def sample_events():
    return pd.DataFrame([
        {"SessionTimeSeconds": 500.0, "Severity": "Warning",  "EventType": "Pit",          "Driver": "VER", "Description": "Pit stop", "Action": "Monitor"},
        {"SessionTimeSeconds": 600.0, "Severity": "Critical", "EventType": "Anomaly",      "Driver": "LEC", "Description": "Unexplained slowdown", "Action": "Investigate"},
        {"SessionTimeSeconds": 700.0, "Severity": "Info",     "EventType": "Race Control", "Driver": None,  "Description": "DRS Enabled", "Action": ""},
    ])


@pytest.fixture
def kpis():
    return {"DriverCount": 20, "Healthy": 1, "Warning": 18, "Critical": 1, "ActiveAlerts": 19, "TotalLaps": 1127}


# ---------------------------------------------------------------------------
# health_badge
# ---------------------------------------------------------------------------

def test_health_badge_healthy():
    html = health_badge_html("Healthy")
    assert "badge-healthy" in html
    assert "Healthy" in html


def test_health_badge_warning():
    html = health_badge_html("Warning")
    assert "badge-warning" in html


def test_health_badge_critical():
    html = health_badge_html("Critical")
    assert "badge-critical" in html


def test_health_badge_unknown():
    html = health_badge_html("Unknown")
    assert "health-badge" in html


# ---------------------------------------------------------------------------
# asset_card
# ---------------------------------------------------------------------------

def test_asset_card_healthy_contains_driver(healthy_row):
    html = asset_card_html(healthy_row)
    assert "VER" in html
    assert "healthy" in html
    assert "SOFT" in html


def test_asset_card_warning_border(warning_row):
    html = asset_card_html(warning_row)
    assert "warning" in html
    assert "LEC" in html
    assert "20" in html


def test_asset_card_critical_none_values(critical_row):
    html = asset_card_html(critical_row)
    assert "critical" in html
    assert "SAI" in html
    assert "—" in html


def test_asset_card_returns_string(healthy_row):
    assert isinstance(asset_card_html(healthy_row), str)


def test_asset_card_contains_badge(warning_row):
    html = asset_card_html(warning_row)
    assert "health-badge" in html


# ---------------------------------------------------------------------------
# event_timeline
# ---------------------------------------------------------------------------

def test_event_timeline_empty():
    html = event_timeline_html(pd.DataFrame())
    assert "timeline" in html
    assert "No events" in html


def test_event_timeline_renders_events(sample_events):
    html = event_timeline_html(sample_events)
    assert "Pit stop" in html
    assert "Unexplained slowdown" in html
    assert "DRS Enabled" in html


def test_event_timeline_severity_classes(sample_events):
    html = event_timeline_html(sample_events)
    assert "critical" in html
    assert "warning" in html
    assert "info" in html


def test_event_timeline_formats_time(sample_events):
    html = event_timeline_html(sample_events)
    assert "8:20" in html  # 500s = 8:20


def test_event_timeline_max_events(sample_events):
    html = event_timeline_html(sample_events, max_events=1)
    assert "Pit stop" in html
    assert "Unexplained slowdown" not in html


def test_event_timeline_action_rendered(sample_events):
    html = event_timeline_html(sample_events)
    assert "Monitor" in html


def test_event_timeline_none_driver_no_brackets(sample_events):
    html = event_timeline_html(sample_events)
    assert "[None]" not in html


@pytest.fixture
def repeated_blue_flag_events():
    """Race-control messages that only differ by car number and timestamp —
    the exact shape FastF1 produces for repeated blue flags."""
    return pd.DataFrame([
        {"SessionTimeSeconds": t, "Severity": "Warning", "EventType": "Race Control",
         "Driver": None, "Description": f"WAVED BLUE FLAG FOR CAR {car} TIMED AT {clock}", "Action": ""}
        for t, car, clock in [
            (100.0, "2 (SAR)", "18:21:59"),
            (200.0, "2 (SAR)", "18:26:22"),
            (300.0, "77 (BOT)", "18:59:32"),
        ]
    ])


def test_event_timeline_clusters_repeated_events(repeated_blue_flag_events):
    html = event_timeline_html(repeated_blue_flag_events)
    assert "3× WAVED BLUE FLAG" in html
    assert "timeline-cluster" in html
    assert "expand" in html


def test_event_timeline_cluster_preserves_individual_rows(repeated_blue_flag_events):
    html = event_timeline_html(repeated_blue_flag_events)
    # All three original timestamps are still present inside the expandable detail
    assert "18:21:59" in html
    assert "18:26:22" in html
    assert "18:59:32" in html


def test_event_timeline_does_not_cluster_distinct_events(sample_events):
    """Three genuinely different events must render as three separate items, not a cluster."""
    html = event_timeline_html(sample_events)
    assert "timeline-cluster" not in html


def test_event_timeline_cluster_respects_max_events(repeated_blue_flag_events):
    """A 3-row cluster still counts as a single item against max_events."""
    other = pd.DataFrame([{
        "SessionTimeSeconds": 400.0, "Severity": "Info", "EventType": "Pit",
        "Driver": "VER", "Description": "Pit stop", "Action": "",
    }])
    events = pd.concat([repeated_blue_flag_events, other], ignore_index=True)
    html = event_timeline_html(events, max_events=2)
    assert "3× WAVED BLUE FLAG" in html
    assert "Pit stop" in html


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------

def test_sparkline_returns_figure():
    import plotly.graph_objects as go
    values = pd.Series([90.0, 91.0, 92.5, 91.8, 93.0])
    fig = sparkline_figure(values)
    assert isinstance(fig, go.Figure)


def test_sparkline_custom_color():
    values = pd.Series([1.0, 2.0, 3.0])
    fig = sparkline_figure(values, color="#F59E0B")
    trace = fig.data[0]
    assert "#F59E0B" in trace.line.color


def test_sparkline_height():
    fig = sparkline_figure(pd.Series([1, 2, 3]))
    assert fig.layout.height == 70


# ---------------------------------------------------------------------------
# status_banner
# ---------------------------------------------------------------------------

def test_status_banner_contains_kpis(kpis):
    html = status_banner_html(kpis, "Green")
    assert "20" in html
    assert "18" in html
    assert "Green" in html


def test_status_banner_green_track(kpis):
    html = status_banner_html(kpis, "Green")
    assert "kpi-healthy" in html


def test_status_banner_yellow_track(kpis):
    html = status_banner_html(kpis, "Yellow")
    assert "kpi-warning" in html


def test_status_banner_critical_track(kpis):
    html = status_banner_html(kpis, "Red")
    assert "kpi-critical" in html


def test_status_banner_returns_string(kpis):
    assert isinstance(status_banner_html(kpis, "Green"), str)
