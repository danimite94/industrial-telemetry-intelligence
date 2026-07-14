import pandas as pd
import pytest

from src.replay_data import (
    checker_line_segments,
    compute_track_edges,
    gauge_needle_point,
    get_frame_at_time,
    lap_duration_seconds,
    load_driver_lap_telemetry,
    load_multi_driver_lap_telemetry,
    multi_lap_duration_seconds,
    multi_track_bounds,
    next_playback_speed,
    scale_to_screen,
    trace_marker_xs,
    trace_progress_points,
    track_bounds,
)


@pytest.fixture
def sample_telemetry_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "LEC"],
            "Time": pd.to_timedelta([0.0, 1.0, 2.0, 0.0], unit="s"),
            "X": [0.0, 100.0, 200.0, 50.0],
            "Y": [0.0, 50.0, 100.0, 25.0],
            "Speed": [200.0, 250.0, 300.0, 210.0],
            "Throttle": [50.0, 100.0, 100.0, 60.0],
            "Brake": [False, False, True, False],
            "nGear": [4, 6, 7, 4],
        }
    )


def test_load_driver_lap_telemetry_filters_and_sorts(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    assert len(lap) == 3
    assert lap["ElapsedSeconds"].tolist() == [0.0, 1.0, 2.0]


def test_track_bounds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    bounds = track_bounds(lap)
    assert bounds == (0.0, 200.0, 0.0, 100.0)


def test_scale_to_screen_maps_into_margin_box():
    bounds = (0.0, 200.0, 0.0, 100.0)
    sx, sy = scale_to_screen(0.0, 0.0, bounds, screen_width=800, screen_height=600, margin=60)
    assert sx == pytest.approx(60.0)
    assert sy == pytest.approx(60.0)

    sx_max, sy_max = scale_to_screen(200.0, 100.0, bounds, screen_width=800, screen_height=600, margin=60)
    assert sx_max <= 800 - 60 + 1e-6
    assert sy_max <= 600 - 60 + 1e-6


def test_get_frame_at_time_returns_nearest_sample(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    frame = get_frame_at_time(lap, 1.4)
    assert frame["ElapsedSeconds"] == 1.0
    assert frame["Speed"] == 250.0
    assert frame["Gear"] == 6


def test_get_frame_at_time_clamps_to_lap_bounds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    before_start = get_frame_at_time(lap, -5.0)
    after_end = get_frame_at_time(lap, 999.0)
    assert before_start["ElapsedSeconds"] == 0.0
    assert after_end["ElapsedSeconds"] == 2.0
    assert after_end["Brake"] is True


def test_lap_duration_seconds(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    assert lap_duration_seconds(lap) == pytest.approx(2.0)


def test_compute_track_edges_offsets_perpendicular_to_straight_line():
    # Horizontal line moving in +x: perpendicular offset should be purely vertical.
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    left, right = compute_track_edges(points, half_width=5.0)

    assert len(left) == len(right) == 3
    for (lx, ly), (rx, ry), (px, py) in zip(left, right, points):
        assert lx == pytest.approx(px)
        assert rx == pytest.approx(px)
        assert ly == pytest.approx(py + 5.0)
        assert ry == pytest.approx(py - 5.0)


def test_compute_track_edges_empty_input():
    left, right = compute_track_edges([], half_width=5.0)
    assert left == []
    assert right == []


def test_checker_line_segments_covers_full_span_and_alternates():
    segments = checker_line_segments((0.0, 0.0), (12.0, 0.0), num_segments=6)

    assert len(segments) == 6
    assert segments[0][0] == pytest.approx((0.0, 0.0))
    assert segments[-1][1] == pytest.approx((12.0, 0.0))
    # Consecutive segments connect end-to-end.
    for (_, end_prev, _), (start_next, _, _) in zip(segments, segments[1:]):
        assert end_prev == pytest.approx(start_next)
    # Alternates black/white starting with black.
    assert [is_black for _, _, is_black in segments] == [True, False, True, False, True, False]


def test_gauge_needle_point_min_value_points_lower_left():
    x, y = gauge_needle_point((0.0, 0.0), radius=10.0, value=0.0, value_min=0.0, value_max=100.0)
    assert x < 0
    assert y < 0


def test_gauge_needle_point_max_value_points_lower_right():
    x, y = gauge_needle_point((0.0, 0.0), radius=10.0, value=100.0, value_min=0.0, value_max=100.0)
    assert x > 0
    assert y < 0


def test_gauge_needle_point_midpoint_points_straight_up():
    x, y = gauge_needle_point((0.0, 0.0), radius=10.0, value=50.0, value_min=0.0, value_max=100.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(10.0)


def test_gauge_needle_point_clamps_out_of_range_values():
    low = gauge_needle_point((0.0, 0.0), radius=10.0, value=-50.0, value_min=0.0, value_max=100.0)
    at_min = gauge_needle_point((0.0, 0.0), radius=10.0, value=0.0, value_min=0.0, value_max=100.0)
    assert low == pytest.approx(at_min)


def test_load_multi_driver_lap_telemetry_loads_each_requested_driver(sample_telemetry_df):
    laps = load_multi_driver_lap_telemetry(sample_telemetry_df, ["VER", "LEC"])
    assert set(laps.keys()) == {"VER", "LEC"}
    assert len(laps["VER"]) == 3
    assert len(laps["LEC"]) == 1


def test_load_multi_driver_lap_telemetry_skips_unknown_drivers(sample_telemetry_df):
    laps = load_multi_driver_lap_telemetry(sample_telemetry_df, ["VER", "ZZZ"])
    assert set(laps.keys()) == {"VER"}


def test_multi_track_bounds_is_union_of_each_driver(sample_telemetry_df):
    laps = load_multi_driver_lap_telemetry(sample_telemetry_df, ["VER", "LEC"])
    bounds = multi_track_bounds(laps)
    # VER spans X 0-200, Y 0-100; LEC is a single point at (50, 25), inside VER's span.
    assert bounds == (0.0, 200.0, 0.0, 100.0)


def test_multi_lap_duration_seconds_is_the_longest_driver(sample_telemetry_df):
    laps = load_multi_driver_lap_telemetry(sample_telemetry_df, ["VER", "LEC"])
    # VER's lap spans 2.0s; LEC's single-sample lap has zero duration.
    assert multi_lap_duration_seconds(laps) == pytest.approx(2.0)


def test_trace_progress_points_grows_as_elapsed_advances(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")  # 3 samples, t=0,1,2
    # Negative elapsed clamps to 0 - the replay clock never actually goes
    # negative, and t=0's own sample is immediately visible at the sweep start.
    at_start = trace_progress_points(lap, "Speed", -1.0, 2.0, 300.0, 200.0, 100.0)
    one_point = trace_progress_points(lap, "Speed", 0.0, 2.0, 300.0, 200.0, 100.0)
    two_points = trace_progress_points(lap, "Speed", 1.0, 2.0, 300.0, 200.0, 100.0)
    full = trace_progress_points(lap, "Speed", 2.0, 2.0, 300.0, 200.0, 100.0)

    assert len(at_start) == 1
    assert len(one_point) == 1
    assert len(two_points) == 2
    assert len(full) == 3


def test_trace_progress_points_x_is_monotonic_and_within_margins(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    points = trace_progress_points(lap, "Speed", 2.0, 2.0, 300.0, 200.0, 100.0, x_margin=10.0, y_margin=10.0)

    xs = [x for x, _ in points]
    assert xs == sorted(xs)
    assert all(10.0 - 1e-9 <= x <= 190.0 + 1e-9 for x in xs)
    assert all(10.0 - 1e-9 <= y <= 90.0 + 1e-9 for _, y in points)


def test_trace_progress_points_clamps_past_duration(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")
    at_duration = trace_progress_points(lap, "Speed", 2.0, 2.0, 300.0, 200.0, 100.0)
    past_duration = trace_progress_points(lap, "Speed", 999.0, 2.0, 300.0, 200.0, 100.0)
    assert at_duration == past_duration


def test_trace_progress_points_zero_duration_is_empty(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "LEC")  # single sample, zero duration
    assert trace_progress_points(lap, "Speed", 0.0, 0.0, 300.0, 200.0, 100.0) == []


def test_trace_marker_xs_only_true_rows_within_visible_window(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "VER")  # Brake True only at t=2.0
    before_brake = trace_marker_xs(lap, "Brake", 1.0, 2.0, 200.0)
    at_brake = trace_marker_xs(lap, "Brake", 2.0, 2.0, 200.0)

    assert before_brake == []
    assert len(at_brake) == 1
    assert 10.0 <= at_brake[0] <= 190.0


def test_trace_marker_xs_zero_duration_is_empty(sample_telemetry_df):
    lap = load_driver_lap_telemetry(sample_telemetry_df, "LEC")
    assert trace_marker_xs(lap, "Brake", 0.0, 0.0, 200.0) == []


def test_next_playback_speed_cycles_through_steps():
    assert next_playback_speed(1.0, speeds=(1.0, 2.0, 4.0)) == 2.0
    assert next_playback_speed(2.0, speeds=(1.0, 2.0, 4.0)) == 4.0


def test_next_playback_speed_wraps_to_first():
    assert next_playback_speed(4.0, speeds=(1.0, 2.0, 4.0)) == 1.0


def test_next_playback_speed_falls_back_for_unknown_current():
    assert next_playback_speed(3.0, speeds=(1.0, 2.0, 4.0)) == 1.0
