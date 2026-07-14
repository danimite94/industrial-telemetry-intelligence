"""Pure data/geometry helpers for the Arcade track replay.

Kept separate from the Arcade window itself so the interpolation and
coordinate-scaling logic can be unit tested without an actual display -
the same "test before complexity" split used elsewhere in this project.
"""

import math

import numpy as np
import pandas as pd


def load_driver_lap_telemetry(telemetry_df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Telemetry for one driver's cached fastest lap, sorted by time."""
    lap = telemetry_df[telemetry_df["Driver"] == driver].copy()
    lap = lap.dropna(subset=["X", "Y"]).sort_values("Time").reset_index(drop=True)
    lap["ElapsedSeconds"] = lap["Time"].dt.total_seconds()
    return lap


def track_bounds(lap_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """(min_x, max_x, min_y, max_y) of the track outline for this lap."""
    return (
        lap_df["X"].min(),
        lap_df["X"].max(),
        lap_df["Y"].min(),
        lap_df["Y"].max(),
    )


def scale_to_screen(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    screen_width: int,
    screen_height: int,
    margin: int = 60,
) -> tuple[float, float]:
    """Map a telemetry (X, Y) point into screen-pixel coordinates, preserving
    aspect ratio so the track shape isn't distorted."""
    min_x, max_x, min_y, max_y = bounds
    track_width = max(max_x - min_x, 1.0)
    track_height = max(max_y - min_y, 1.0)

    available_width = screen_width - 2 * margin
    available_height = screen_height - 2 * margin
    scale = min(available_width / track_width, available_height / track_height)

    screen_x = margin + (x - min_x) * scale
    screen_y = margin + (y - min_y) * scale
    return screen_x, screen_y


def compute_track_edges(
    points: list[tuple[float, float]],
    half_width: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Two parallel boundary lines offset `half_width` either side of the centerline.

    FastF1 doesn't give real track width, so this is a fixed-width visual
    stand-in: at each point, offset perpendicular to the local direction of
    travel (estimated from the neighbouring points).
    """
    n = len(points)
    if n == 0:
        return [], []

    left_edge: list[tuple[float, float]] = []
    right_edge: list[tuple[float, float]] = []
    for i in range(n):
        prev_point = points[i - 1] if i > 0 else points[i]
        next_point = points[i + 1] if i < n - 1 else points[i]
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        length = (dx**2 + dy**2) ** 0.5 or 1.0
        normal_x, normal_y = -dy / length, dx / length

        x, y = points[i]
        left_edge.append((x + normal_x * half_width, y + normal_y * half_width))
        right_edge.append((x - normal_x * half_width, y - normal_y * half_width))

    return left_edge, right_edge


def checker_line_segments(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    num_segments: int = 6,
) -> list[tuple[tuple[float, float], tuple[float, float], bool]]:
    """Split the line from `point_a` to `point_b` into equal sub-segments for a
    checkered start/finish line, each tagged True (black) / False (white)."""
    segments = []
    for i in range(num_segments):
        t0 = i / num_segments
        t1 = (i + 1) / num_segments
        start = (point_a[0] + (point_b[0] - point_a[0]) * t0, point_a[1] + (point_b[1] - point_a[1]) * t0)
        end = (point_a[0] + (point_b[0] - point_a[0]) * t1, point_a[1] + (point_b[1] - point_a[1]) * t1)
        segments.append((start, end, i % 2 == 0))
    return segments


def gauge_needle_point(
    center: tuple[float, float],
    radius: float,
    value: float,
    value_min: float,
    value_max: float,
    start_angle_deg: float = -120.0,
    end_angle_deg: float = 120.0,
) -> tuple[float, float]:
    """Tip of a speedometer-style needle for `value` on a dial swept from
    `start_angle_deg` (lower-left, at value_min) through 0 (straight up) to
    `end_angle_deg` (lower-right, at value_max)."""
    clamped = min(max(value, value_min), value_max)
    span = value_max - value_min
    fraction = (clamped - value_min) / span if span else 0.0
    angle_deg = start_angle_deg + fraction * (end_angle_deg - start_angle_deg)
    angle_rad = math.radians(angle_deg)

    cx, cy = center
    return cx + radius * math.sin(angle_rad), cy + radius * math.cos(angle_rad)


def get_frame_at_time(lap_df: pd.DataFrame, elapsed_seconds: float) -> dict:
    """Nearest telemetry sample at or before `elapsed_seconds` (clamped to the
    lap's start/end), as a plain dict for the Arcade window to draw."""
    clamped = min(max(elapsed_seconds, lap_df["ElapsedSeconds"].iloc[0]), lap_df["ElapsedSeconds"].iloc[-1])
    idx = np.searchsorted(lap_df["ElapsedSeconds"].to_numpy(), clamped, side="right") - 1
    idx = min(max(idx, 0), len(lap_df) - 1)
    row = lap_df.iloc[idx]
    return {
        "X": float(row["X"]),
        "Y": float(row["Y"]),
        "Speed": float(row["Speed"]),
        "Throttle": float(row["Throttle"]),
        "Brake": bool(row["Brake"]),
        "Gear": int(row["nGear"]),
        "ElapsedSeconds": float(row["ElapsedSeconds"]),
    }


def lap_duration_seconds(lap_df: pd.DataFrame) -> float:
    return float(lap_df["ElapsedSeconds"].iloc[-1] - lap_df["ElapsedSeconds"].iloc[0])


def load_multi_driver_lap_telemetry(
    telemetry_df: pd.DataFrame, drivers: list[str]
) -> dict[str, pd.DataFrame]:
    """Each driver's cached fastest lap, keyed by driver code.

    Each lap keeps its own independent clock starting at ElapsedSeconds == 0
    (set by `load_driver_lap_telemetry`), so playing them on a shared replay
    clock compares pace lap-for-lap rather than wall-clock session time.
    Drivers with no cached telemetry are silently skipped rather than raising,
    so one missing driver doesn't abort a multi-driver replay.
    """
    laps = {}
    for driver in drivers:
        lap = load_driver_lap_telemetry(telemetry_df, driver)
        if not lap.empty:
            laps[driver] = lap
    return laps


def multi_track_bounds(lap_dfs: dict[str, pd.DataFrame]) -> tuple[float, float, float, float]:
    """Union of `track_bounds` across every driver's lap, so the track outline
    fits every car's racing line even where they diverge slightly."""
    all_bounds = [track_bounds(lap_df) for lap_df in lap_dfs.values()]
    min_x = min(b[0] for b in all_bounds)
    max_x = max(b[1] for b in all_bounds)
    min_y = min(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    return (min_x, max_x, min_y, max_y)


def multi_lap_duration_seconds(lap_dfs: dict[str, pd.DataFrame]) -> float:
    """Longest of the selected drivers' lap durations.

    Faster drivers simply hold at their final telemetry sample (the same
    clamping `get_frame_at_time` already does for a single driver) until the
    shared replay clock loops, rather than the replay resetting early.
    """
    return max(lap_duration_seconds(lap_df) for lap_df in lap_dfs.values())


def trace_progress_points(
    lap_df: pd.DataFrame,
    value_col: str,
    elapsed_seconds: float,
    duration_seconds: float,
    value_max: float,
    panel_width: float,
    panel_height: float,
    x_margin: float = 10.0,
    y_margin: float = 10.0,
) -> list[tuple[float, float]]:
    """Screen-space points for the portion of `value_col` sampled from lap
    start up to `elapsed_seconds` - an "oscilloscope sweep" that grows one
    point at a time as a replay plays, filling the panel by the time
    `elapsed_seconds` reaches `duration_seconds` and holding there after.
    """
    if duration_seconds <= 0:
        return []
    clamped = min(max(elapsed_seconds, 0.0), duration_seconds)
    visible = lap_df[lap_df["ElapsedSeconds"] <= clamped]
    if visible.empty:
        return []
    usable_width = max(panel_width - 2 * x_margin, 1.0)
    usable_height = max(panel_height - 2 * y_margin, 1.0)
    points = []
    for t, value in zip(visible["ElapsedSeconds"], visible[value_col]):
        x = x_margin + (t / duration_seconds) * usable_width
        fraction = min(max(value / value_max, 0.0), 1.0) if value_max else 0.0
        y = y_margin + fraction * usable_height
        points.append((x, y))
    return points


def trace_marker_xs(
    lap_df: pd.DataFrame,
    marker_col: str,
    elapsed_seconds: float,
    duration_seconds: float,
    panel_width: float,
    x_margin: float = 10.0,
) -> list[float]:
    """Screen x-positions where a boolean column (e.g. Brake) is true, within
    the same visible-so-far window as `trace_progress_points`."""
    if duration_seconds <= 0:
        return []
    clamped = min(max(elapsed_seconds, 0.0), duration_seconds)
    visible = lap_df[(lap_df["ElapsedSeconds"] <= clamped) & lap_df[marker_col].astype(bool)]
    if visible.empty:
        return []
    usable_width = max(panel_width - 2 * x_margin, 1.0)
    return [x_margin + (t / duration_seconds) * usable_width for t in visible["ElapsedSeconds"]]


def next_playback_speed(current: float, speeds: tuple[float, ...] = (1.0, 2.0, 4.0)) -> float:
    """Cycle to the next fast-forward multiplier, wrapping back to the first
    after the last. Falls back to the first step if `current` isn't one of
    the configured steps (defensive - shouldn't happen in practice)."""
    try:
        idx = speeds.index(current)
    except ValueError:
        return speeds[0]
    return speeds[(idx + 1) % len(speeds)]
