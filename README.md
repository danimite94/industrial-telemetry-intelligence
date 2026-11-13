# Industrial Telemetry Intelligence

[![CI](https://github.com/djimrastephane/industrial-telemetry-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/djimrastephane/industrial-telemetry-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **Industrial Asset Monitoring and Decision Support System**, prototyped end-to-end on
public Formula 1 telemetry data.

**This is not a sports analytics project.** Real ESP (electric submersible pump), SCADA, and
production well sensor data is almost never public. F1 telemetry is one of the few public
datasets shaped the same way: high-frequency, multi-sensor, multi-asset, and
degradation/failure-relevant - so it's used here to build and validate an industrial analytics
pipeline that could later point at real plant data with minimal rework.

## Glossary (plain-language)

A few terms that come up below, explained once instead of repeatedly:

- **z-score** - how many standard deviations a value is from the average. A simple way to flag
  "this lap was unusually slow for this driver," without needing a complex model.
- **i.i.d. (independent and identically distributed)** - a statistical assumption that each data
  point is unrelated to the ones before or after it. The anomaly detector below makes this
  assumption, which is why it can't tell a real problem apart from "the first lap on new tyres."
- **MAE (mean absolute error)** - the average size of a forecast's mistake, in the same units as
  what's being predicted (here, seconds per lap).
- **Tertile** - splitting data into three equal-sized groups (bottom/middle/top third), used here
  to label risk as Low/Medium/High relative to this race's own laps, not a fixed standard.
- **Baseline** - the simplest possible comparison point (e.g. "assume next lap = last lap").
  A real model only proves its worth if it beats this.

## Confidentiality and grounding (Phase 6 assistant)

The operational assistant is built so it could later run against **real, confidential
ESP/SCADA data** without exposing it:

- **Local LLM only.** Runs against a local [Ollama](https://ollama.com) server - no telemetry,
  question, or answer ever leaves your machine.
- **Every answer is grounded with citations.** The LLM only explains evidence Python retrieved
  first; it never invents lap times or events. `validate_citations()` checks in code that every
  cited lap actually exists in the evidence, rather than trusting the model to comply.
- **No evidence, no LLM call.** If a question can't be matched to a driver and lap, the
  assistant says so directly and the LLM is never invoked.

See [Status & Roadmap](#status--roadmap) and [Current limitations](#current-limitations) for
the honest gaps in these guardrails.

## Operational Context Layer (Phase 9)

Telemetry alone doesn't explain *why* it changed - the same drop in speed can mean tyre wear, a
yellow flag, or a real problem. Phase 9 adds that missing layer:

```
Telemetry  +  Operational Context  ->  Health Assessment  ->  Recommendation
```

Today's "operational context" is Formula 1 weather, tyre state, track status (green/yellow/
safety car), and race control messages. The five context-engine functions are written to be
domain-independent, so a future ESP/SCADA version could swap in real plant data without
changing any other module:

| Formula 1 (current) | Future ESP/SCADA |
|---|---|
| Air Temperature | Reservoir Pressure |
| Track Temperature | Wellhead Temperature |
| Wind (speed/direction) | Choke Setting |
| Humidity | Operating Envelope |
| Rainfall | Operational Constraints |
| Tyre compound / age | Equipment mode / run-to-failure interval |
| Track status (flags, safety car) | Equipment alarm state |
| Race control messages | SCADA alarm/event log |

The point isn't that air temperature *equals* reservoir pressure - it's that telemetry needs
contextual interpretation, on an architecture that survives swapping the context source.

## Industrial Monitoring Interface

The dashboard adopts the information architecture of industrial asset monitoring systems.
Formula 1 telemetry is the demonstration dataset; the interface design is intended to transfer
directly to industrial telemetry systems such as ESP surveillance, wind farm monitoring,
compressor monitoring, manufacturing assets, and hydraulic fracturing fleet monitoring.

No claim is made that Formula 1 telemetry is equivalent to industrial SCADA data. The
transferability lies in the **software architecture and monitoring workflow**, not the physics.

### Navigation hierarchy

```
Fleet Overview  →  Driver Detail  →  Telemetry Replay
(What needs         (Why?)            (What happened?)
 attention?)
```

### Design principles applied

| Principle | Implementation |
|---|---|
| **KPI-first layout** | Fleet Overview opens with 8 metric cards: assets monitored, laps loaded, healthy / warning / critical counts, active alerts, track status, session |
| **Asset health** | Every driver has a Green / Amber / Red status from deterministic rules (no new models) |
| **Expected vs Actual** | Driver Detail shows baseline average pace vs last lap time, and expected degradation slope |
| **Event timeline** | Chronological log of pit stops, track status changes, anomalies, recommendations, and race control messages |
| **Trend-based monitoring** | Lap time trend and tyre degradation charts preferred over point-in-time gauges |
| **Progressive disclosure** | Overview answers "what needs attention?"; Detail answers "why?"; Replay answers "what happened?" |
| **Decision support** | Every alert in the event log carries an Action field: what to do next |

### Health status rules (deterministic, no new models)

| Status | Trigger |
|---|---|
| Critical (Red) | Any unexplained anomaly (Phase 10) OR "Pit now" recommendation (Phase 7) |
| Warning (Amber) | Any partially-explained anomaly OR active pit recommendation OR High/Medium degradation risk |
| Healthy (Green) | None of the above |

### Transferability to industrial systems

The same Fleet Overview → Driver Detail → Replay hierarchy applies directly to:

- **ESP surveillance**: each ESP is an asset; the health table is the fleet; the event log
  carries pump starts, vibration alarms, and motor-current anomalies.
- **Wind farm monitoring**: each turbine is an asset; SCADA alarms replace race control messages;
  power-curve deviations replace lap-time anomalies.
- **Compressor monitoring**: each compressor is an asset; vibration spectrum events replace pit
  stops; valve-cycle counts replace tyre stints.
- **Manufacturing**: each machine or line is an asset; shift logs replace race laps; OEE metrics
  replace lap times.

Swapping the data source means replacing `src/data_ingestion.py` and the column names in
`src/context_engine.py`; no other module needs to change.

## Screenshots

| Tab | What it shows |
|---|---|
| **Fleet Overview** | KPI status banner, fleet health donut, top-risk asset cards, event timeline, full asset grid |
| **Driver Detail** | Driver header card, expected vs actual, sparkline + trend charts, recommendations panel, event timeline |
| Race Detail (1-2) | Tyre degradation, speed/throttle/brake traces, anomaly table |
| Season Monitoring (3) | Position/speed-trap trends across a season, asset-health slope |
| Fleet Monitoring (4) | Cross-year pace/degradation benchmarking |
| Predictive Analytics (5) | Lap-time forecast comparison, degradation forecast, risk scores |
| Operational Assistant (6) | A real question, the retrieved evidence, the grounded LLM answer |
| Decision Support (7) | Recommended pit windows, sorted by urgency |
| Arcade Replay (8) | Multiple cars on the real track shape, live instrument cluster |
| Operational Context (9) | Tyre state, track status, temperatures, and recent race control events for a selected lap |
| Health Assessment (10) | Noise reduction KPIs — z-score alarms re-scored through context, unexplained vs explained breakdown |

![Fleet Overview — dark industrial theme](assets/screenshots/fleet_overview_dark.png)


## Why Formula 1 data?

| F1 telemetry (FastF1) | Industrial equivalent |
|---|---|
| Lap | Duty cycle / operating interval |
| Tyre stint | Run-to-failure interval between maintenance |
| Lap time trending up across a stint | Vibration/temperature/pressure trending up as equipment wears |
| Speed, throttle, brake traces | High-frequency sensor streams (pressure, flow, current, RPM) |
| Tyre compound | Equipment configuration / operating mode |
| Lap time z-score outlier | SCADA threshold alarm |
| Driver-to-driver comparison | Asset-to-asset (well-to-well, pump-to-pump) comparison |
| ESP motor temp/vibration/current drift | Same slow-drift-before-failure pattern as tyre wear |
| SCADA threshold alarms | The `anomaly_detection.py` z-score flag, deliberately simple |
| Per-well production baselines | `baseline_models.py`'s per-driver baselines |

That final platform isn't built up front - per the [Architecture rule](#architecture-rule),
each phase is built, run on real data, and validated before the next one starts.

## Related work

[IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) also pairs FastF1 with
the [Arcade](https://api.arcade.academy/en/latest/) library to render a 2D race replay - the
same general combination `app/arcade_replay.py` uses. It's a much larger, F1-fan-facing replay
tool in its own right (leaderboard, simulated Safety Car, playback controls, GUI/CLI menus,
qualifying support, a telemetry streaming server). This project was built independently, shares
no code or text with it, and differs in scope (the replay here is a small piece of a larger
analytics pipeline, not the whole project), frame architecture (frames computed on demand here
vs. precomputed at a fixed FPS there), and context design (this project's context engine is
written to be domain-independent; theirs is F1-specific UI code). Their README claims an MIT
license, but as of this writing the repo has no actual `LICENSE` file.

## Skills demonstrated

- Real-world API ingestion and local caching (FastF1 → parquet)
- Data cleaning and validation on messy, real time-series data
- Feature engineering on time-series structure (stint-relative counters, rolling means)
- Statistical baselining before modelling (means, std dev, degradation slope, z-score)
- Interactive visualization (Plotly) and a working Streamlit dashboard
- Test-driven development of data pipelines with pytest
- Clear documentation of assumptions, limitations, and an honest improvement roadmap

## Project structure

```
industrial-telemetry-intelligence/
  src/
    config.py               # paths, session selection, thresholds
    data_ingestion.py        # FastF1 session -> laps/weather/telemetry/race-control -> parquet
    data_cleaning.py         # load + basic cleaning of processed parquet
    feature_engineering.py   # stint-lap counter, rolling lap time
    baseline_models.py       # avg lap time, fastest lap, consistency, degradation
    anomaly_detection.py     # per-driver z-score anomaly flag (Phase 2)
    degradation_analysis.py  # linear lap-time slope per tyre stint
    seasonal_analysis.py     # season-wide position/speed trends and KPIs
    fleet_analysis.py        # multi-year relative pace, degradation, year-over-year shift
    predictive_models.py     # lap-time forecast (baselines + Linear/RF/XGBoost), risk scores
    operational_assistant.py # Phase 6: parse question -> retrieve evidence -> local LLM
    decision_support.py      # Phase 7: degradation forecast -> pit/maintenance recommendation
    replay_data.py           # pure geometry/interpolation helpers for the Arcade replay
    context_engine.py        # Phase 9: domain-independent operational context engine
    health_assessment.py     # Phase 10: re-scores Phase 2 anomalies through Phase 9 context
    event_log.py             # Phase 11: unified chronological event log from all sources
    fleet_health.py          # Phase 11: per-driver Green/Amber/Red health status (deterministic)
    visualisation.py         # shared Plotly figures
  app/
    streamlit_app.py         # dashboard: Fleet Overview + Driver Detail + all phase tabs
    arcade_replay.py          # standalone Arcade window: multi-driver cars + live HUD + context
  notebooks/                 # same analysis, exploratory format
  tests/                     # one test file per src module, synthetic fixtures only
  data/                       # raw/processed/cache - gitignored, regenerated locally
  outputs/                    # figures/reports - gitignored, regenerated locally
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, `xgboost` (Phase 5) needs the OpenMP runtime, which isn't bundled:

```bash
brew install libomp
```

Phase 6 needs a local [Ollama](https://ollama.com) server - not a hosted API - so the same
approach works against confidential real ESP/SCADA data later:

```bash
ollama serve                        # start the local server
ollama pull qwen2.5:7b-instruct     # default model (src/config.py:OLLAMA_MODEL)
```

## Running the pipeline

1. **Download and cache one race session** (2024 Bahrain GP, Race). Hits FastF1/Ergast once;
   subsequent runs are offline.

   ```bash
   python -m src.data_ingestion
   ```

   Writes `laps.parquet`, `weather.parquet`, `telemetry.parquet`, `race_control.parquet`.

2. **Sanity-check cleaning/features/baselines** (optional, uses step 1's data):

   ```bash
   python -m src.data_cleaning
   python -m src.feature_engineering
   python -m src.baseline_models
   python -m src.degradation_analysis
   python -m src.anomaly_detection
   ```

3. **Download the full season** (24 rounds; skips any round that fails to load):

   ```bash
   python -m src.data_ingestion season
   ```

4. **Download the multi-year fleet dataset** (same race, 2020-2025):

   ```bash
   python -m src.data_ingestion fleet
   ```

5. **Sanity-check predictive models / the assistant / decision support / context engine**
   (optional, all reuse step 1's data, no new download; the assistant needs Ollama running):

   ```bash
   python -m src.predictive_models
   python -m src.operational_assistant
   python -m src.decision_support
   python -m src.context_engine
   python -m src.health_assessment
   ```

6. **Run the dashboard** (opens on **Fleet Overview** — the industrial monitoring entry point.
   Additional tabs cover every phase: Driver Detail, Race Detail, Season Monitoring, Fleet
   Monitoring, Predictive Analytics, Operational Assistant, Decision Support, Operational
   Context, Health Assessment):

   ```bash
   streamlit run app/streamlit_app.py
   ```

7. **Run the Arcade replay** (separate native window; the gauge panel shows a live Operational
   Context readout for the focus driver):

   ```bash
   python app/arcade_replay.py --drivers VER,LEC,NOR
   python app/arcade_replay.py --drivers VER          # single driver
   python app/arcade_replay.py --all                  # every cached driver (full grid)
   ```

   Each driver's lap plays on its own independent clock (their own fastest lap, not session
   wall-clock time) - a pace comparison, not a recreation of real race positions.

8. **Run the notebooks** (`notebooks/`) for the same analysis in an exploratory format.

9. **Run tests**:

   ```bash
   pytest tests/ -v
   ```

   Tests use synthetic fixtures and a temp directory - they never touch real downloaded data.

## Status & Roadmap

Each phase is built, run on real data, and validated before the next one starts. Dataset:
2024 Bahrain GP Race (all 20 drivers), the full 2024 season (~26,600 laps), and Bahrain GP
2020-2025 (~6,500 laps).

**Phase 1 - Single Driver Exploration ✅** End-to-end ingestion (laps, weather, telemetry) for
one race, cleaned, with stint-relative lap counters and rolling means, Plotly charts, and a
working dashboard.

**Phase 2 - Race Intelligence ✅** Driver-vs-driver comparison for any two drivers; baselines
(average lap time, consistency, fastest lap, degradation slope) across the full grid; z-score
anomaly flagging shown in a dashboard table.

**Phase 3 - Seasonal Monitoring ✅** Season-wide ingestion (24 rounds), kept lean by extracting
only lap time, running position, and a speed-trap reading per lap. Driver/team position trends
and an asset-health slope (`PositionTrendSlope`) per driver/team.

**Phase 4 - Multi-Year Fleet Monitoring ✅** Same race ingested every year 2020-2025.
Cross-year benchmarking led by pace *relative to your own teammate that year* (cancels out most
of the car/regulation effect), plus reliability and pit-stop-efficiency proxies and a
year-over-year `Shift` label (Declined/Improved/Stable).

**Phase 5 - Predictive Analytics ✅** Lap-time forecasting: two honest baselines (assume next
lap ≈ last lap, or the mean) versus three trained models (Linear Regression, Random Forest,
XGBoost), evaluated on a chronological split. **Honest result: the naive "next lap ≈ last lap"
baseline (MAE 0.43s) beat every trained model** - reported as-is rather than tuned away. Also:
degradation forecasting (straight-line extrapolation) and data-driven Low/Medium/High risk
scores (tertiles of this race's own distribution).

**Phase 6 - Operational Intelligence Assistant ✅** Answers questions like *"Why did VER lose
performance after lap 32?"* by retrieving real telemetry evidence first, then asking a local
LLM to explain it - never the reverse. Two code-level checks (not just prompt instructions)
verify compliance: `validate_citations()` confirms cited laps are real, and
`detect_speculative_language()` flags overconfident phrasing. **Validated on a real example**:
asking why BOT lost performance at lap 13 correctly cites the pit stop onto a fresh tyre, and
explicitly declines to diagnose further.

**Phase 7 - Decision Support Recommendations ✅** Turns Phase 5's degradation forecast into an
explicit recommendation ("Pit now" / "Pit within N laps" / "No action needed") by projecting
the existing linear fit to a configurable threshold - no new model. **Validated against real
strategy**: for several drivers, the model's projected threshold-crossing lap lands a few laps
*after* their actual pit lap, consistent with proactive (not reactive) pit strategy.

**Phase 8 - Multi-Driver Replay ✅** Generalizes the Arcade replay to any number of drivers
(or the full 20-car grid via `--all`) moving on the same track at once, each on its own
independent lap clock - a fair pace comparison, not a recreation of real race positions. The
gauge cluster stays focused on one driver while every car is still drawn on track.

**Phase 9 - Operational Context Layer ✅** Adds the context engine described above: weather,
tyre state, track status, and race events, aligned to telemetry's own clock, with deterministic
rule-based explanations (e.g. "track temperature increasing → expect more tyre wear"; "Virtual
Safety Car active → expect a performance reduction"). **Two real gaps were found by validating
against real data, and fixed**: (1) a pit out-lap on a fresh tyre wasn't recognized as an
explanation for a slow lap, even though the engine already had that fact - confirmed against
BOT's real lap 13; (2) the fix only worked for a "low speed" signal, not a "lap time increased"
signal, even though they're the same physical symptom. Both are now covered by regression tests.

**Phase 10 - Context-Aware Health Assessment ✅** The missing middle step from Phase 9's own
diagram: re-scores Phase 2's context-blind anomaly flags through Phase 9's context engine, so
each flagged lap becomes **Explained**, **Partially Explained**, or **Unexplained -
Investigate** instead of one undifferentiated alarm. No new model - just Phase 2 + Phase 9
combined. **Validated on the real race: all 44 flagged anomalies were explained** (pit out-laps
or opening-lap yellow flags) - 100% noise reduction, with zero false positives left to chase.
The "Unexplained" escalation path is proven by a synthetic test, since this race didn't happen
to contain a real unexplained anomaly.

**Phase 11 - Industrial Monitoring Interface ✅** Transforms the tab-first analytics dashboard
into a three-level industrial monitoring interface: Fleet Overview (KPI cards, asset health
table, event log), Driver Detail (expected vs actual, lap trends, per-driver alerts and
recommendations), and Telemetry Replay (existing arcade app). Two new `src/` modules:
`event_log.py` (unified chronological event log from pit, anomaly, recommendation, track-status,
and race-control events) and `fleet_health.py` (deterministic Green/Amber/Red health status per
driver). No new models — all inputs come from Phases 2, 7, and 10.

142 tests total across all phases (`pytest tests/ -v`).

## Architecture rule

At every phase:

1. Build the simplest version first.
2. Visualize before modelling.
3. Use baseline methods before advanced ML.
4. Validate results against domain knowledge.
5. Do not introduce complexity unless the current phase is working.
6. Keep all code production-quality and portfolio-ready.

## Current limitations

Grouped by area - every caveat below is a deliberate, documented trade-off, not an oversight.

**Anomaly detection (Phase 2).** The z-score flag is i.i.d. (see Glossary) - it judges each lap
in isolation, so a tyre-compound change or pit stop can look like a "false alarm" rather than
expected behaviour. Not yet extended to season-wide data (Phase 3 only adds trend KPIs).

**Fleet benchmarking (Phase 4).** `FastestLapGapPct` is one noisy data point (why
`TeammateGapPct` is the primary metric instead); `TeammateGapPct` is `NaN` for a driver whose
team had no other driver that year; `RaceCompletionRatePct` is a lap-count proxy, not official
DNF data; the `Shift` label uses a fixed 0.3-point threshold, not one derived from the data.

**Predictive models (Phase 5).** The naive baseline beating every trained model means none are
currently "production-worthy" for 1-lap-ahead forecasting (a longer horizon or richer features
would be the next thing to try). The degradation forecast is a short-horizon straight-line
extrapolation that doesn't know about upcoming pit stops or non-linear tyre "cliffs." Risk
thresholds are relative to this race's own distribution, not a fixed cross-race standard.

**Operational assistant (Phase 6).** Question parsing is regex/keyword matching (one driver
code + "lap N"), not an embeddings- or LLM-based parser. `detect_speculative_language()` is a
keyword heuristic, so it both under- and over-flags. `validate_citations()` only checks that a
cited lap *exists* - it can't verify the model quoted that lap's value correctly.

**Decision support (Phase 7).** The pit-recommendation threshold (5%) is a fixed constant, not
fitted to the data. It inherits Phase 5's straight-line forecast caveat. `RiskCategory` (Phase
5, relative) and `RecommendedAction` (Phase 7, absolute) can legitimately disagree, and the
dashboard shows both rather than reconciling them. Phase 7 recommends a pit *window*, not a
strategy-aware pit *lap* (it doesn't know about tyre allocation rules, rivals, or safety-car
probability).

**Arcade replay (Phases 8-9).** Track width and gauge ranges are fixed visual constants, not
derived per track. Because each driver's lap clock starts independently, the multi-driver
replay shows *who is faster*, not *who was ahead on track* - conflating the two would
misrepresent real race positions. The track outline is drawn once, from the focus driver's
line only. The live context readout is single-driver only.

**Context engine (Phase 9).** Thresholds (e.g. what counts as a "moderate" temperature change,
or the 15-lap tyre-age cutoff) are fixed constants, not fitted to this race's or track's own
variance - a change that's moderate at a hot track might be unremarkable at a cooler one. The
explanation rules are checked in a fixed priority order, not ranked, so if two explanations
apply at once (e.g. a temperature spike *and* high tyre age) only the first is reported.
Race-control message matching uses one fixed lookback window and returns only the single most
recent message. Track status reflects only the *last* flag raised during a lap - a brief
mid-lap yellow that clears before the lap ends shows as Green.

**Health assessment (Phase 10).** Inherits every limitation above from Phase 2 and Phase 9,
since it only re-scores their outputs. It's currently validated on one real race where every
anomaly happened to be explained - the "Unexplained" escalation path has only been proven with
synthetic data, not yet with a real unexplained anomaly.

I did some changes