# Data extraction work

This directory is for work on data extraction methods and code for working
with our fused data sets. This includes realized power data extraction for
OPF runs and target/feature matrix extraction for training statistical
forecasters.

---

## Contents

| File | Layer | What it does |
| --- | --- | --- |
| `regrow_query.py` | logic | Reads the fused node files. Collapses the duplicated actual values, and selects forecasts causally for a given window. |
| `regrow_extractor.py` | notebook | Interactive front end for `regrow_query.py`: pick nodes, columns and a window, inspect the result, write it out. |
| `regrow_features.py` | logic | Builds the feature matrix `X` and the target matrix `Y` used to fit a forecaster. |
| `regrow_matrices.py` | notebook | Interactive front end for `regrow_features.py`. |

The two notebooks contain no logic of their own; they only collect arguments
and display results. Every function they call is importable from a script,
which is what a training run or an MPC back test will do.

`regrow_features.py` reads through `regrow_query.py`, so the causal rule and
the column normalisation are defined once rather than twice.

---

## Requirements

```
python >= 3.10
pandas >= 2.0
pyarrow
numpy
marimo          # only needed for the two notebooks
```

## Setup

Point `REGROW_ROOT` at the folder that contains `fused_data/parquet`. Both
notebooks use it as the default value of their folder box, and it keeps
machine-specific paths out of the repository.

```bash
# macOS / Linux
export REGROW_ROOT=/path/to/REGORW

# Windows PowerShell (persistent)
setx REGROW_ROOT "C:\path\to\REGORW"
```

The expected layout is:

```
REGORW/
├── fused_data/
│   └── parquet/            # one <node>_fused.parquet per node
├── node_load_data/         # optional, used by scan_stream_dirs()
└── ...
```

---

## The data model, in brief

Every row of a fused file carries two timestamps, and telling them apart is
the whole game:

| column | meaning |
| --- | --- |
| `predict_day` / `predict_time` | when the forecast was **issued** |
| `forecast_day` / `forecast_time` | the hour the forecast is **about** |
| `day_diff` / `hour_diff` | `forecast_time` − `predict_time` |

Forecasts were issued every six hours and each issuance rolls out 120 hours,
so any target hour is covered by roughly twenty issuances. That splits the
file into two kinds of column, which have to be read differently:

- **Actual values** — generation, load, NOAA / NSRDB / hub weather. One true
  value per target hour, copied onto all ~20 rows that mention it. They are
  collapsed. Skipping that step multiplies every total by about twenty.
- **Forecast weather** — genuinely different on every row, so exactly one row
  per target hour has to be *chosen*, and which one is chosen is a modelling
  decision.

Two conventions worth knowing before reading a column index back into a
meaning:

- Generation columns carry the node geohash in their name (`9mudw2_solar`),
  while every other column does not. `logical_name()` strips it, so anything
  that groups columns by name must group on the part *after* the geohash.
- `day_diff` counts **local calendar days in `America/Los_Angeles`** while the
  timestamps are UTC, so it is **not** `hour_diff // 24` — it agrees with that
  on only 43% of rows. Use `hour_diff`, which is a plain cumulative UTC offset.

---

## `regrow_query.py` — reading the fused files

### `Catalog(root)`

Indexes every node from the parquet footers only, so the whole catalog costs
about a second and no meaningful memory.

```python
import os
import regrow_query as rq

cat = rq.Catalog(os.environ["REGROW_ROOT"])
cat.describe()             # one row per node: rows, span, which streams
cat.column_availability()  # how empty each column is, grouped without the geohash
cat.dtype_outliers()       # columns whose dtype differs between node files
cat.with_stream("wind")    # nodes carrying a given stream
```

### `get_actuals(...)` — the realised values

One row per node per target hour. The ~20 copies of each hour are identical,
so one is kept; `check_duplicates=True` verifies that rather than assuming it.

```python
power = rq.get_actuals(
    cat,
    nodes=cat.node_names(),
    streams=["solar", "wind", "dist_solar"],
    start="2019-06-03 09:00",
    end="2019-06-06 09:00",
    layout="wide",          # one column per (node, stream) — what an OPF run wants
)
```

No vantage point constrains this read: an actual value is not a prediction,
so the window may be as long as the data set.

### `get_forecasts(...)` — the causal window

A forecast window `[start, end]` is a **future window seen from `start`**.
Only issuances made at or before `start` are admissible; among those, every
target hour takes the most recently issued forecast of itself, which is the
smallest `hour_diff`.

```python
fc = rq.get_forecasts(
    cat,
    nodes=["9x0gc5"],
    columns=["temperature", "wind_speed", "clouds"],
    actuals=["wind"],       # attaches the realised power over the same window
    start="2019-06-03 09:00",
    end="2019-06-06 09:00",
)
print(rq.validate_forecast_window(fc, "2019-06-03 09:00", "2019-06-06 09:00"))
```

Because every source rolls out the same 120 hours on the same cadence, one
issuance normally serves the whole window: `predict_time` comes back constant
and `hour_diff` climbs steadily. The per-hour minimisation is applied anyway,
since it is also the rule that covers mixed sources with different rollout
lengths, where an older but longer-reaching issuance has to serve the tail.

A window longer than 120 h cannot be completed. What exists is returned and an
`IncompleteWindowWarning` is raised rather than the shortfall passing
silently.

### Reporting rather than filling

Nothing is ever filled with zeros. A column a node does not have, a target
hour with no admissible issuance, an issuance older than one six-hour cycle —
each is recorded in `df.attrs["notes"]` and surfaced in the notebook.

---

## `regrow_features.py` — building X and Y

### The paradigm

The forecaster is never a one-step model iterated forward. Standing at an
**origin** time `T0`, it predicts the whole rollout at once, as a vector in
R<sup>H</sup>. So the target is a **matrix**:

```
Y[i, h-1] = actual value of the target at T0_i + h,   h = 1 … H
```

and every row of `X` holds only what was genuinely available at `T0_i`,
assembled from up to three switchable blocks:

| block | contents | direction | default |
| --- | --- | --- | --- |
| `forecast` | forecast weather for T0+1 … T0+H, from issuances made at or before T0 | forward | on |
| `autoregressive` | the actual target streams at T0, T0−1, … | backward | off |
| `observed` | actual weather at T0, T0−1, … | backward | off |

Forecast alone is the pure exogenous model; adding the autoregressive block
gives exogenous-plus-autoregressive. The blocks are stored separately so the
value of each can be measured by leaving it out.

Several streams at one node **share one X** and get **one Y each**.

### Usage

```python
import regrow_query as rq
import regrow_features as rf

cat = rq.Catalog(os.environ["REGROW_ROOT"])
origins = rf.make_origins("2019-01-01", "2019-12-31 23:00",
                          every="1h", tz=cat.time_tz)

m = rf.build_node_matrices(
    cat, "9x0gc5", targets=["wind"], origins=origins, horizon=48,
    forecast_columns=["temperature", "wind_speed", "wind_deg", "clouds"],
    ar_lags=[0, -1, -2, -24],          # omit for the pure exogenous model
)

X, Y = m.X[m.valid], m.Y["wind"][m.valid]   # fit on the complete rows
```

A full year of hourly origins on one node takes about a second.

Useful views:

```python
m.feature_layout()      # which columns of X hold which variable
m.forecast_frame(i)     # the forecast block of sample i, one row per step
m.target_frame(i)       # row i of Y, one row per step
m.lag_frame(i)          # the backward-looking features, with the hour each points at
m.features("forecast")  # X restricted to some blocks, for ablation
m.forecast_cube()       # (n_origins, horizon, n_forecast_columns) for sequence models
```

### Across nodes

Nodes are stacked along a leading axis rather than appended into ever wider
matrices, and a node → index map comes back with them. Any subset works.

```python
t = rf.build_tensor(cat, cat.with_stream("wind"), "wind", origins)
t.X.shape                 # (n_nodes, n_origins, n_features)
t.node_index["9x0gc5"]    # which slice is which node
t.skipped                 # nodes left out, and why
rf.describe_dimensions(t) # what every axis means, with its contents
```

A node enters the tensor only if it carries every requested target and
column; the rest are listed in `skipped` rather than padded.

### Causality

This is the property that would not announce itself if it broke: a leaking X
trains and scores perfectly well and is simply wrong. It is enforced in three
places.

1. Lags must be ≤ 0. A positive lag is refused with an explanation, since it
   would put the answer into the question.
2. Every forecast cell is asserted to have been issued at or before its
   origin.
3. `check_against_reference()` recomputes sampled rows the slow way, with
   `get_forecasts` / `get_actuals`, and compares them cell by cell. The
   forecast block is normally built by one vectorised as-of join for speed;
   this makes that fast path inherit the reference implementation's
   correctness instead of asking to be trusted.

```python
rf.check_against_reference(m, cat, n_samples=5)
```

### Missing data

Samples are never dropped. An origin too close to the start of the data for
its lags, too close to the end for its targets, or sitting on a gap in the
source file keeps its row, with `NaN` where the value does not exist, and is
marked `False` in `m.valid`. Dropping rows per node would leave the nodes
with different sample counts and make them impossible to stack.

`m.notes` names the block that fell short. This is how the NSRDB and hub
cut-off after 2021 surfaces: with `nsrdb_*` in the observed block, origins
from 2022 onward come back invalid.

### Memory

Check before allocating — a full network with every forecast column is
several gigabytes.

```python
rf.estimate_bytes(n_nodes=126, n_origins=8760,
                  n_features=rf.n_features_for(rq.FORECAST_WEATHER, 48),
                  horizon=48)                  # bytes
```

`dtype=np.float32` halves it.

### Saving

```python
path = m.save("wind_9x0gc5")       # .npz
d = rf.load_matrices(path)         # {"X", "Y", "valid", "origins", ...}
```

---

## The notebooks

```bash
marimo edit regrow_extractor.py    # extract actual or forecast series
marimo edit regrow_matrices.py     # build X and Y
```

Both open with a folder box defaulted to `$REGROW_ROOT`. Use `marimo run`
instead of `marimo edit` for a read-only view with the code cells hidden.

`regrow_matrices.py` in particular shows, for a chosen sample: the forecast
block step by step with the hour each step refers to and when that forecast
was published, the backward-looking lags with the hour each points at, and
the matching row of `Y` — the three kept in separate tables, because the
distinction that must never blur is which side of the origin a number came
from.

---

## Known data issues

These are properties of the fused files as of this writing, not of the code.
They are surfaced by the tools rather than worked around.

| Issue | Effect |
| --- | --- |
| No fused file carries a `{node_id}_load` column | Load is unavailable; the column name is already wired in and will work once it returns. |
| `nsrdb_*` and `hub_*` stop at the end of 2021 | Observed-weather features from those groups are `NaN` from 2022 onward. |
| `nsrdb_calc_cloud_coverage` is null at night | Expected, not a defect: it is derived from irradiance, which is zero in darkness. |
| `day_diff` uses local calendar days while timestamps are UTC | Use `hour_diff`. |
| A few nodes have a column stored as `int64` or `null` where the rest use `double` | Normalised on read; `Catalog.dtype_outliers()` names them. |

---

## Open questions

1. With a shared `X`, whose lags should the autoregressive block carry — every
   target, or one set per target? Currently it defaults to every target, so
   that `X` stays genuinely shared.
2. Should origins be spaced every hour, or every 6 hours to match the issuance
   cycle? Hourly matches how often a receding-horizon controller re-plans, but
   six consecutive origins then share one issuance.
3. Should the forecast block also carry step `h = 0`, so that a
   forecast-minus-observed bias feature becomes possible? The forecast of the
   origin hour itself is legitimately available at the origin, but is not
   currently included.
