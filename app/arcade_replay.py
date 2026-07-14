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
    next_playback_speed,
    scale_to_screen,
    trace_marker_xs,
    trace_progress_points,
)

SCREEN_WIDTH = 1400  # wide enough that neither the gauges nor the trace panel feels cramped
SCREEN_HEIGHT = 850  # kept at the original window size so it still fits on-screen
GAUGE_PANEL_HEIGHT = 260  # top strip: instrument cluster + telemetry trace, side by side
TRACK_AREA_HEIGHT = SCREEN_HEIGHT - GAUGE_PANEL_HEIGHT  # lower strip: track view (full width)
GAUGE_PANEL_Y0 = TRACK_AREA_HEIGHT  # bottom edge of the top strip

GAUGE_AREA_WIDTH = SCREEN_WIDTH // 2  # left half of the top strip: instrument cluster
TRACE_PANEL_WIDTH = SCREEN_WIDTH - GAUGE_AREA_WIDTH  # right half of the top strip: telemetry trace
TRACE_PANEL_X0 = GAUGE_AREA_WIDTH  # left edge of the trace panel within the top strip

MARGIN = 60
TRACK_SHIFT_X = 120  # pixels; pushes the track right, away from the left edge
TRACK_HALF_WIDTH = 10  # pixels either side of the centerline (FastF1 has no real track width)

SPEED_MAX_KPH = 350.0
THROTTLE_MAX_PCT = 100.0

GAUGE_CENTER_Y = GAUGE_PANEL_Y0 + GAUGE_PANEL_HEIGHT * 0.42  # vertical center within the panel
# Same relative spacing as the original single-column layout (0.18/0.42/0.65/0.85 of the width),
# now applied to the narrower gauge-cluster half of the top strip.
SPEED_GAUGE_CENTER = (0.18 * GAUGE_AREA_WIDTH, GAUGE_CENTER_Y)
THROTTLE_GAUGE_CENTER = (0.42 * GAUGE_AREA_WIDTH, GAUGE_CENTER_Y)
BRAKE_LAMP_CENTER = (0.65 * GAUGE_AREA_WIDTH, GAUGE_CENTER_Y)
GEAR_BOX_CENTER = (0.85 * GAUGE_AREA_WIDTH, GAUGE_CENTER_Y)
GAUGE_RADIUS = 60
BRAKE_LAMP_RADIUS = 32
GEAR_BOX_HALF_SIZE = 32

TRACE_X_MARGIN = 28  # inside the trace panel, not the full-width MARGIN
TRACE_Y_MARGIN = 34  # leaves room for the title row (top) and baseline (bottom)

# ── Industrial control-room theme (matches app/styles/dark_theme.css) ───────
INK_950 = (10, 12, 15)
SURFACE_1 = (18, 21, 27)
SURFACE_2 = (23, 27, 34)
SURFACE_3 = (29, 34, 43)
BORDER_STRONG = (58, 65, 78)
TEXT_PRIMARY = (241, 244, 248)
TEXT_SECONDARY = (152, 164, 181)
ACCENT = (45, 212, 200)
CRITICAL = (240, 71, 60)
CRITICAL_FAINT = (240, 71, 60, 70)  # brake-zone ticks in the trace panel

# Distinct colours cycled across drivers; the focus driver (gauge panel) is
# always the first one passed on the command line. 20 entries covers the
# full grid without repeating a colour. Generated in OKLCH (fixed lightness
# band + gamut-safe chroma per hue, dataviz-skill method), with hues ordered
# by a van der Corput sequence rather than a plain sweep - so any *prefix* of
# this list (the first 2, 3, 5 entries: whatever a typical --drivers run
# actually uses) is already maximally hue-separated, not just the full set
# of 20 taken together. Each car is also always labelled with its driver
# code, so identity never depends on colour alone.
CAR_COLORS = [
    (214, 35, 117),
    (0, 165, 143),
    (152, 117, 0),
    (99, 129, 246),
    (193, 86, 27),
    (0, 155, 199),
    (78, 141, 28),
    (197, 64, 246),
    (221, 36, 57),
    (0, 162, 174),
    (124, 129, 0),
    (147, 110, 246),
    (172, 105, 0),
    (36, 145, 229),
    (24, 145, 91),
    (229, 42, 194),
    (218, 36, 90),
    (0, 164, 159),
    (139, 123, 0),
    (124, 121, 246),
]

LEGEND_TOP_Y = GAUGE_PANEL_Y0 - 20  # just below the gauge-panel divider, inside the track view
LEGEND_ROW_HEIGHT = 22
LEGEND_COLUMNS = 5
LEGEND_COLUMN_WIDTH = SCREEN_WIDTH // LEGEND_COLUMNS

# ── Transport controls (pause / fast-forward), top-right of the track view ──
SPEED_STEPS = (1.0, 2.0, 4.0)
BUTTON_HEIGHT = 34
BUTTON_Y1 = GAUGE_PANEL_Y0 - 14
BUTTON_Y0 = BUTTON_Y1 - BUTTON_HEIGHT
FF_BUTTON_WIDTH = 90
PAUSE_BUTTON_WIDTH = 110
BUTTON_GAP = 12
FF_BUTTON_X1 = SCREEN_WIDTH - MARGIN
FF_BUTTON_X0 = FF_BUTTON_X1 - FF_BUTTON_WIDTH
PAUSE_BUTTON_X1 = FF_BUTTON_X0 - BUTTON_GAP
PAUSE_BUTTON_X0 = PAUSE_BUTTON_X1 - PAUSE_BUTTON_WIDTH
PAUSE_BUTTON_RECT = (PAUSE_BUTTON_X0, BUTTON_Y0, PAUSE_BUTTON_X1, BUTTON_Y1)
FF_BUTTON_RECT = (FF_BUTTON_X0, BUTTON_Y0, FF_BUTTON_X1, BUTTON_Y1)


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
        self.paused = False
        self.playback_speed = SPEED_STEPS[0]

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
        if self.paused:
            return
        self.elapsed += delta_time * self.playback_speed
        if self.elapsed > self.duration:
            self.elapsed = 0.0  # loop the replay

    @staticmethod
    def _hit(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        if self._hit(PAUSE_BUTTON_RECT, x, y):
            self.paused = not self.paused
        elif self._hit(FF_BUTTON_RECT, x, y):
            self.playback_speed = next_playback_speed(self.playback_speed, SPEED_STEPS)

    def on_key_press(self, key: int, modifiers: int):
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.F:
            self.playback_speed = next_playback_speed(self.playback_speed, SPEED_STEPS)

    def on_draw(self):
        self.clear()
        frames = {driver: get_frame_at_time(lap_df, self.elapsed) for driver, lap_df in self.lap_dfs.items()}
        self._draw_track(frames)
        self._draw_legend()
        self._draw_transport_controls()
        self._draw_gauge_panel(frames[self.focus_driver])
        self._draw_trace_panel(frames[self.focus_driver])

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

    def _draw_transport_controls(self):
        """Pause and fast-forward buttons, clickable (`on_mouse_press`) or
        keyboard-driven (Space / F) - both toggle the same `self.paused` /
        `self.playback_speed` state that `on_update` already reads, so the
        track dots, gauges, and telemetry trace all pause or speed up
        together automatically."""
        self._draw_button(PAUSE_BUTTON_RECT, "⏸  PAUSE" if not self.paused else "▶  PLAY", active=self.paused)
        self._draw_button(FF_BUTTON_RECT, f"⏩  {self.playback_speed:.0f}X", active=self.playback_speed > 1.0)

    @staticmethod
    def _draw_button(rect: tuple[float, float, float, float], label: str, active: bool):
        x0, y0, x1, y1 = rect
        arcade.draw_lrbt_rectangle_filled(x0, x1, y0, y1, SURFACE_2)
        arcade.draw_lrbt_rectangle_outline(x0, x1, y0, y1, ACCENT if active else BORDER_STRONG, border_width=2)
        arcade.draw_text(
            label, (x0 + x1) / 2, (y0 + y1) / 2, ACCENT if active else TEXT_PRIMARY, 13,
            anchor_x="center", anchor_y="center",
        )

    def _draw_gauge_panel(self, frame: dict):
        arcade.draw_lrbt_rectangle_filled(
            0, SCREEN_WIDTH, GAUGE_PANEL_Y0, SCREEN_HEIGHT, SURFACE_1
        )
        arcade.draw_line(0, GAUGE_PANEL_Y0, SCREEN_WIDTH, GAUGE_PANEL_Y0, ACCENT, 2)

        title = f"Focus: {self.focus_driver}   Lap time: {self.elapsed:5.1f}s / {self.duration:5.1f}s"
        arcade.draw_text(
            title, GAUGE_AREA_WIDTH / 2, SCREEN_HEIGHT - 30, TEXT_PRIMARY, 16,
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
            readout, GAUGE_AREA_WIDTH / 2, SCREEN_HEIGHT - 50, TEXT_SECONDARY, 12,
            anchor_x="center",
        )

    def _draw_trace_panel(self, frame: dict):
        """Live speed trace for every driver in the replay, sharing the top
        strip with the instrument cluster instead of sitting beside or below
        the track view. All lines share one time axis (`self.duration` - the
        same clock that drives car position and holds a finished driver in
        place) so pace can be compared directly: each line grows as that
        driver's lap plays and holds once they cross the line. Drawn in each
        driver's own car colour (matching the track dots and legend) so
        identity carries across every panel; the focus driver is drawn last
        (on top, thicker) with its own head readout and brake-zone ticks.
        Every column plotted (`Speed`, `Brake`) is telemetry already loaded
        for the gauges, just plotted over time instead of read off a dial."""
        panel_x0 = TRACE_PANEL_X0
        arcade.draw_line(panel_x0, GAUGE_PANEL_Y0, panel_x0, SCREEN_HEIGHT, ACCENT, 2)

        arcade.draw_text(
            f"TELEMETRY TRACE — SPEED · focus {self.focus_driver}",
            panel_x0 + TRACE_X_MARGIN, SCREEN_HEIGHT - 20, TEXT_SECONDARY, 11,
        )

        brake_xs = trace_marker_xs(
            self.lap_dfs[self.focus_driver], "Brake", self.elapsed, self.duration,
            TRACE_PANEL_WIDTH, x_margin=TRACE_X_MARGIN,
        )
        for x in brake_xs:
            arcade.draw_line(
                panel_x0 + x, GAUGE_PANEL_Y0 + 8, panel_x0 + x, SCREEN_HEIGHT - TRACE_Y_MARGIN,
                CRITICAL_FAINT, line_width=1,
            )

        focus_head = None
        ordered_drivers = [d for d in self.drivers if d != self.focus_driver] + [self.focus_driver]
        for driver in ordered_drivers:
            is_focus = driver == self.focus_driver
            points = trace_progress_points(
                self.lap_dfs[driver], "Speed", self.elapsed, self.duration, SPEED_MAX_KPH,
                TRACE_PANEL_WIDTH, GAUGE_PANEL_HEIGHT, x_margin=TRACE_X_MARGIN, y_margin=TRACE_Y_MARGIN,
            )
            points = [(panel_x0 + x, GAUGE_PANEL_Y0 + y) for x, y in points]
            if len(points) >= 2:
                arcade.draw_line_strip(points, self.colors[driver], 2 if is_focus else 1)
            if points:
                hx, hy = points[-1]
                arcade.draw_circle_filled(hx, hy, 4 if is_focus else 2.5, self.colors[driver])
                if is_focus:
                    focus_head = (hx, hy)

        if focus_head is not None:
            hx, hy = focus_head
            arcade.draw_text(
                f"{frame['Speed']:.0f} km/h", hx, hy + 10, TEXT_PRIMARY, 11, anchor_x="center",
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
