"""Arcade replay: one or more drivers' fastest laps, position dot(s) + live
gauges for a single focus driver.

This is the "digital control room" view: car(s) moving around the track on
the bottom of the window, with a gauge cluster (speed dial, throttle dial,
brake lamp, gear box) across the top for the focus driver - the same
instrument-panel layout as a SCADA operator screen monitoring one asset
while still seeing every asset's position on the shared track view.
Tyre-degradation colour, anomaly alerts, pit-stop prompts, and an
asset-health score are deliberately left out of this version - see the
README roadmap for where those land.

Each driver's lap plays on its own independent clock starting at t=0 (their
own fastest lap), not session wall-clock time, so the replay compares pace
lap-for-lap rather than where each car happened to be during the race.

Run with:
    python app/arcade_replay.py --drivers VER,LEC,NOR
    python app/arcade_replay.py --drivers VER          # single driver, as before
"""

import argparse
import sys
from pathlib import Path

import arcade

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPLAY_DRIVERS
from src.context_engine import align_context_to_session, get_context_at_timestamp, load_context
from src.data_cleaning import load_and_clean_all
from src.replay_data import (
    checker_line_segments,
    compute_track_edges,
    gauge_needle_point,
    get_frame_at_time,
    load_multi_driver_lap_telemetry,
    multi_lap_duration_seconds,
    multi_track_bounds,
    scale_to_screen,
)

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 850
GAUGE_PANEL_HEIGHT = 260  # top strip reserved for the instrument cluster
TRACK_AREA_HEIGHT = SCREEN_HEIGHT - GAUGE_PANEL_HEIGHT  # track is drawn below the gauges

MARGIN = 60
TRACK_SHIFT_X = 120  # pixels; pushes the track right, away from the left edge
TRACK_HALF_WIDTH = 10  # pixels either side of the centerline (FastF1 has no real track width)

SPEED_MAX_KPH = 350.0
THROTTLE_MAX_PCT = 100.0

GAUGE_CENTER_Y = TRACK_AREA_HEIGHT + GAUGE_PANEL_HEIGHT * 0.42  # vertical center within the panel
SPEED_GAUGE_CENTER = (180, GAUGE_CENTER_Y)
THROTTLE_GAUGE_CENTER = (420, GAUGE_CENTER_Y)
BRAKE_LAMP_CENTER = (650, GAUGE_CENTER_Y)
GEAR_BOX_CENTER = (850, GAUGE_CENTER_Y)
GAUGE_RADIUS = 75
BRAKE_LAMP_RADIUS = 40
GEAR_BOX_HALF_SIZE = 40

# ── Industrial control-room theme (matches app/styles/dark_theme.css) ───────
INK_950 = (10, 12, 15)
SURFACE_1 = (18, 21, 27)
SURFACE_3 = (29, 34, 43)
BORDER_STRONG = (58, 65, 78)
TEXT_PRIMARY = (241, 244, 248)
TEXT_SECONDARY = (152, 164, 181)
ACCENT = (45, 212, 200)
CRITICAL = (240, 71, 60)

# Distinct colours cycled across drivers; the focus driver (gauge panel) is
# always the first one passed on the command line. 20 entries covers the
# full grid without repeating a colour. Generated in OKLCH (fixed lightness
# band + gamut-safe chroma per hue, dataviz-skill method) so every entry
# stays readable on the ink-black track view; each car is also always
# labelled with its driver code, so identity never depends on colour alone.
CAR_COLORS = [
    (204, 33, 111),
    (240, 40, 79),
    (198, 64, 25),
    (198, 106, 31),
    (162, 101, 0),
    (165, 129, 0),
    (125, 121, 0),
    (112, 149, 32),
    (30, 140, 40),
    (25, 158, 110),
    (0, 140, 119),
    (0, 157, 162),
    (0, 134, 165),
    (0, 145, 200),
    (27, 118, 203),
    (92, 121, 246),
    (118, 74, 244),
    (171, 77, 246),
    (180, 34, 194),
    (223, 39, 174),
]

LEGEND_TOP_Y = TRACK_AREA_HEIGHT - 20  # just below the divider, inside the track view
LEGEND_ROW_HEIGHT = 22
LEGEND_COLUMNS = 5
LEGEND_COLUMN_WIDTH = SCREEN_WIDTH // LEGEND_COLUMNS


class ReplayWindow(arcade.Window):
    def __init__(self, lap_dfs: dict, drivers: list[str], laps_df=None):
        title = "Industrial Telemetry Replay - " + ", ".join(drivers)
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title)
        arcade.set_background_color(INK_950)
        self.lap_dfs = lap_dfs
        self.drivers = drivers
        self.focus_driver = drivers[0]
        self.colors = {driver: CAR_COLORS[i % len(CAR_COLORS)] for i, driver in enumerate(drivers)}
        self.bounds = multi_track_bounds(lap_dfs)
        self.duration = multi_lap_duration_seconds(lap_dfs)
        self.elapsed = 0.0

        # Phase 9: live Operational Context readout for the focus driver.
        # The replay's own clock is lap-relative (t=0 at the fastest lap's
        # start); the context engine needs absolute session-elapsed time, so
        # `self.operational_context_time_offset` is that lap's LapStartTimeSeconds. If
        # context can't be loaded/matched, the readout is simply skipped.
        self.operational_context = None
        self.operational_context_time_offset = 0.0
        if laps_df is not None:
            try:
                self.operational_context = align_context_to_session(load_context())
                focus_lap_number = lap_dfs[self.focus_driver]["LapNumber"].iloc[0]
                offset_rows = laps_df[
                    (laps_df["Driver"] == self.focus_driver) & (laps_df["LapNumber"] == focus_lap_number)
                ]
                if not offset_rows.empty:
                    self.operational_context_time_offset = float(offset_rows["LapStartTimeSeconds"].iloc[0])
            except (FileNotFoundError, KeyError):
                self.operational_context = None

        reference_lap = lap_dfs[self.focus_driver]
        centerline = [
            self._shift(scale_to_screen(x, y, self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN))
            for x, y in zip(reference_lap["X"], reference_lap["Y"])
        ]
        self.left_edge, self.right_edge = compute_track_edges(centerline, TRACK_HALF_WIDTH)
        self.start_finish_segments = checker_line_segments(
            self.left_edge[0], self.right_edge[0], num_segments=6
        )

    @staticmethod
    def _shift(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x + TRACK_SHIFT_X, y

    def on_update(self, delta_time: float):
        self.elapsed += delta_time
        if self.elapsed > self.duration:
            self.elapsed = 0.0  # loop the replay

    def on_draw(self):
        self.clear()
        frames = {driver: get_frame_at_time(lap_df, self.elapsed) for driver, lap_df in self.lap_dfs.items()}
        self._draw_track(frames)
        self._draw_legend()
        self._draw_gauge_panel(frames[self.focus_driver])

    def _draw_track(self, frames: dict):
        arcade.draw_line_strip(self.left_edge, BORDER_STRONG, 3)
        arcade.draw_line_strip(self.right_edge, BORDER_STRONG, 3)

        for (sx, sy), (ex, ey), is_black in self.start_finish_segments:
            color = INK_950 if is_black else TEXT_PRIMARY
            arcade.draw_line(sx, sy, ex, ey, color, line_width=6)

        for driver, frame in frames.items():
            car_x, car_y = self._shift(
                scale_to_screen(frame["X"], frame["Y"], self.bounds, SCREEN_WIDTH, TRACK_AREA_HEIGHT, MARGIN)
            )
            arcade.draw_circle_filled(car_x, car_y, 8, self.colors[driver])
            arcade.draw_text(
                driver, car_x, car_y + 12, self.colors[driver], 12, anchor_x="center"
            )

    def _draw_legend(self):
        for i, driver in enumerate(self.drivers):
            column, row = i % LEGEND_COLUMNS, i // LEGEND_COLUMNS
            x = MARGIN + column * LEGEND_COLUMN_WIDTH
            y = LEGEND_TOP_Y - row * LEGEND_ROW_HEIGHT
            arcade.draw_circle_filled(x, y, 6, self.colors[driver])
            suffix = " (gauges)" if driver == self.focus_driver else ""
            arcade.draw_text(f"{driver}{suffix}", x + 14, y - 7, TEXT_PRIMARY, 12)

    def _draw_gauge_panel(self, frame: dict):
        arcade.draw_lrbt_rectangle_filled(
            0, SCREEN_WIDTH, TRACK_AREA_HEIGHT, SCREEN_HEIGHT, SURFACE_1
        )
        arcade.draw_line(0, TRACK_AREA_HEIGHT, SCREEN_WIDTH, TRACK_AREA_HEIGHT, ACCENT, 2)

        title = f"Focus: {self.focus_driver}   Lap time: {self.elapsed:5.1f}s / {self.duration:5.1f}s"
        arcade.draw_text(
            title, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 30, TEXT_PRIMARY, 16,
            anchor_x="center",
        )

        self._draw_dial(SPEED_GAUGE_CENTER, frame["Speed"], 0, SPEED_MAX_KPH, "SPEED", " km/h")
        self._draw_dial(THROTTLE_GAUGE_CENTER, frame["Throttle"], 0, THROTTLE_MAX_PCT, "THROTTLE", "%")
        self._draw_brake_lamp(BRAKE_LAMP_CENTER, frame["Brake"])
        self._draw_gear_box(GEAR_BOX_CENTER, frame["Gear"])
        self._draw_context_readout()

    def _draw_context_readout(self):
        if self.operational_context is None:
            return
        session_time = self.elapsed + self.operational_context_time_offset
        context_now = get_context_at_timestamp(self.operational_context, session_time, self.focus_driver)

        tyre = f"{context_now.get('Compound', 'Unknown')} ({context_now.get('TyreLife', '?')} laps)"
        track = context_now.get("TrackStatus", "Unknown")
        air_temp = f"{context_now['AirTemp']:.1f}C" if "AirTemp" in context_now else "—"
        track_temp = f"{context_now['TrackTemp']:.1f}C" if "TrackTemp" in context_now else "—"

        readout = f"Tyre: {tyre}   Track: {track}   Air: {air_temp}   Track Temp: {track_temp}"
        arcade.draw_text(
            readout, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50, TEXT_SECONDARY, 12,
            anchor_x="center",
        )

    @staticmethod
    def _draw_dial(center, value, value_min, value_max, label, unit):
        cx, cy = center
        arcade.draw_circle_outline(cx, cy, GAUGE_RADIUS, BORDER_STRONG, border_width=3)
        needle_x, needle_y = gauge_needle_point(center, GAUGE_RADIUS - 10, value, value_min, value_max)
        arcade.draw_line(cx, cy, needle_x, needle_y, ACCENT, line_width=3)
        arcade.draw_circle_filled(cx, cy, 5, TEXT_PRIMARY)
        arcade.draw_text(
            f"{value:.0f}{unit}", cx, cy - GAUGE_RADIUS - 22, TEXT_PRIMARY, 14, anchor_x="center"
        )
        arcade.draw_text(
            label, cx, cy + GAUGE_RADIUS + 6, TEXT_SECONDARY, 12, anchor_x="center"
        )

    @staticmethod
    def _draw_brake_lamp(center, is_braking: bool):
        cx, cy = center
        lamp_color = CRITICAL if is_braking else SURFACE_3
        arcade.draw_circle_filled(cx, cy, BRAKE_LAMP_RADIUS, lamp_color)
        arcade.draw_circle_outline(cx, cy, BRAKE_LAMP_RADIUS, BORDER_STRONG, border_width=2)
        arcade.draw_text(
            "BRAKE", cx, cy - BRAKE_LAMP_RADIUS - 22, TEXT_PRIMARY, 12, anchor_x="center"
        )

    @staticmethod
    def _draw_gear_box(center, gear: int):
        cx, cy = center
        arcade.draw_lrbt_rectangle_outline(
            cx - GEAR_BOX_HALF_SIZE, cx + GEAR_BOX_HALF_SIZE,
            cy - GEAR_BOX_HALF_SIZE, cy + GEAR_BOX_HALF_SIZE,
            BORDER_STRONG, border_width=3,
        )
        arcade.draw_text(
            str(gear), cx, cy, TEXT_PRIMARY, 30, anchor_x="center", anchor_y="center"
        )
        arcade.draw_text(
            "GEAR", cx, cy - GEAR_BOX_HALF_SIZE - 22, TEXT_SECONDARY, 12, anchor_x="center"
        )


def main():
    parser = argparse.ArgumentParser(description="Arcade telemetry replay (multi-driver)")
    parser.add_argument(
        "--drivers",
        default=",".join(REPLAY_DRIVERS),
        help="Comma-separated 3-letter FastF1 driver codes, e.g. VER,LEC,NOR. "
        "The first one shown gets the gauge panel. Ignored if --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay every driver with cached telemetry (the full grid) instead of --drivers.",
    )
    args = parser.parse_args()

    laps_df, _, telemetry_df = load_and_clean_all()

    if args.all:
        all_drivers = sorted(telemetry_df["Driver"].unique())
        focus_default = REPLAY_DRIVERS[0]
        requested_drivers = (
            [focus_default] + [d for d in all_drivers if d != focus_default]
            if focus_default in all_drivers
            else all_drivers
        )
    else:
        requested_drivers = [d.strip().upper() for d in args.drivers.split(",") if d.strip()]

    lap_dfs = load_multi_driver_lap_telemetry(telemetry_df, requested_drivers)
    missing = [d for d in requested_drivers if d not in lap_dfs]
    if missing:
        print(f"Warning: no cached telemetry for {missing}, skipping.")
    if not lap_dfs:
        raise SystemExit(
            f"No cached telemetry for any of {requested_drivers}. "
            "Run `python -m src.data_ingestion` first, or pick drivers from the cached set."
        )

    drivers = [d for d in requested_drivers if d in lap_dfs]
    ReplayWindow(lap_dfs, drivers, laps_df=laps_df)
    arcade.run()


if __name__ == "__main__":
    main()
