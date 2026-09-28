"""A query layer over the REGROW fused node data.

--------------------------------------------------------------------------
The two timestamps, and why everything here turns on them
--------------------------------------------------------------------------

Every row of a fused node file carries two times, and confusing them is the
single easiest way to produce a wrong answer from this dataset:

    predict_day / predict_time   when the forecast was ISSUED
    forecast_day / forecast_time the hour the forecast is ABOUT (the target)
    day_diff / hour_diff         forecast_time minus predict_time

Historical forecasts were issued every six hours and each issuance rolls out
120 hours, so any given target hour is covered by roughly twenty different
issuances. The relationship is many-to-one: many forecasts of one reality.

That has two consequences, and they pull in opposite directions:

  * The measured and actual columns - generation, load, NOAA/NSRDB/hub
    weather - are the reality. There is exactly one true value per target
    hour, and the fusion pipeline copies it onto every one of the ~20 rows
    that mention that hour. Reading them without collapsing the duplicates
    inflates every series roughly twentyfold.

  * The forecast weather columns are the predictions. They genuinely differ
    from row to row, so collapsing them would be destroying real
    information. Instead exactly one of them has to be chosen, and which one
    is chosen is a modelling decision with consequences (see below).

So this module offers two reads rather than one:

    get_actuals()    collapses the duplicates - generation, load, measured
                     weather. One row per node per target hour.
    get_forecasts()  keeps them and applies the causal rule below -
                     forecast weather, optionally joined to the actual power
                     over the same window so the result can train a
                     forecaster.

--------------------------------------------------------------------------
The causal rule for forecast windows
--------------------------------------------------------------------------

A forecast window [start, end] is a FUTURE window seen from `start`. Asking
for one means standing at `start` and asking what the next end - start
hours were expected to look like, which is the whole point of a back test.
So:

    1. Only issuances made at or before `start` are admissible. An issuance
       made INSIDE the window did not exist yet from that vantage point, and
       returning one leaks the future into the back test.

    2. Among the admissible issuances, every target hour takes the most
       recently issued forecast of itself - the smallest `hour_diff`.
       The fused files carry that column so this lookup is an index search
       rather than arithmetic, so that is what is used here; no timestamps
       are differenced.

    3. `end` decides which target hours are returned. It plays no part in
       choosing between issuances.

In the data as it stands the minimisation resolves to one issuance for the
whole window, because every source rolls out the same 120 hours on the same
six-hour cadence, so the newest admissible issuance reaches further than
any older one. `predict_time` therefore comes back CONSTANT and `hour_diff`
climbs steadily from the window start. That is the signature of a healthy
result.

The per-hour form of the minimisation is written anyway. It costs nothing,
and it is the rule that also covers the harder case: two forecast sources
with different rollout lengths and cadences, where the newest issuance may
stop short of the window and an older, longer-reaching one has to serve the
tail. That case is in the raw upstream data even if the fusion pipeline
standardised it away here - and the same rule solves both, so a constant predict_time comes
out as a result rather than being assumed.

Two consequences of causality, both reported rather than hidden:

  * A window longer than MAX_HORIZON_HOURS cannot be completed, because no
    admissible issuance reaches that far. What exists is returned and an
    IncompleteWindowWarning is raised.
  * Where `hour_diff` climbs faster than the hours elapsed since `start`,
    the issuance that should have served those hours is absent from the
    source file and an older rollout had to be used instead.

None of this constrains get_actuals. A realised value is not a prediction,
so no vantage point applies to it and a window there may be as long as the
data set.

Typical use:

    import regrow_query as rq

    cat = rq.Catalog(os.environ["REGROW_ROOT"])   # or a literal path

    # 72-hour study starting 09:00 on 3 June 2019
    s, e = rq.window("2019-06-03 09:00", hours=72)

    # the freshest forecast of every hour in the window, plus the actual
    # power those forecasts were trying to predict
    train = rq.get_forecasts(
        cat, nodes=["9mudw2"],
        columns=["temperature", "wind_speed", "clouds"],
        actuals=["wind", "solar"],
        start=s, end=e,
    )
    print(rq.validate_forecast_window(train, s, e))

    # power time series for a power-flow run over the same window
    power = rq.get_actuals(
        cat, nodes=cat.node_names(),
        streams=["solar", "wind", "dist_solar"],
        start=s, end=e, layout="wide",
    )

--------------------------------------------------------------------------
Known data problems this module works around
--------------------------------------------------------------------------

  1. Generation columns carry the node geohash in their name - `9mudw2_solar`
     rather than `solar` - while every other column does not. Two nodes
     therefore cannot be concatenated without renaming. _normalise strips the
     prefix; _resolve_columns puts it back when deciding what to read. Any
     analysis that groups columns by name must group on the part AFTER the
     geohash, or every generation column looks like a one-node column.
  2. The same column is `double` in most nodes, `int64` in a few where the
     values happen to be whole numbers, and `null` where the column is empty
     for every row. Concatenating those either raises or silently upcasts.
  3. No fused file contains a `{node_id}_load` column, although the Primer
     defines one. Load has to come from the node_load_data folder instead;
     see scan_stream_dirs. This is reported, never filled with zeros.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

__all__ = [
    "Catalog",
    "NodeInfo",
    "window",
    "get_actuals",
    "get_forecasts",
    "validate_forecast_window",
    "assert_no_leakage",
    "IncompleteWindowWarning",
    "scan_stream_dirs",
    "export",
    "GEN_STREAMS",
    "ACTUAL_GROUPS",
    "FORECAST_WEATHER",
    "HRRR_COLS",
    "WWD_COLS",
    "TIME_COL",
    "ISSUE_COL",
    "MAX_HORIZON_HOURS",
]

# Columns 0-6 of the Primer. Present in every node, never null.
INDEX_COLS = [
    "node_id", "predict_day", "predict_time",
    "forecast_day", "forecast_time", "day_diff", "hour_diff",
]

# The target hour - the hour a row is ABOUT. Everything is keyed against it.
TIME_COL = "forecast_time"

# When the forecast was issued.
ISSUE_COL = "predict_time"

# forecast_time - predict_time, in hours. Cumulative, so it is sufficient on
# its own; day_diff only exists to make "how far ahead" readable.
OFFSET_COL = "hour_diff"
DAY_OFFSET_COL = "day_diff"

# Provenance kept on forecast results so the selection can be audited.
# day_diff is deliberately not here: it counts local calendar days in
# America/Los_Angeles while the timestamps are UTC, so it is not hour_diff
# // 24 and showing it next to a UTC offset invites misreading. hour_diff is
# the cumulative offset and is sufficient on its own.
PROVENANCE_COLS = [ISSUE_COL, OFFSET_COL]

# How far a single issuance rolls out. A property of the source data, not a
# tunable: forecasts were generated every 6 h and each run covers 120 h.
MAX_HORIZON_HOURS = 120
ISSUE_INTERVAL_HOURS = 6

# Generation streams, as they appear once the node prefix is stripped.
GEN_STREAMS = ["solar", "wind", "dist_solar"]

# Actual power, including load. Load is listed because it is meant to be
# here; it is currently absent from every fused file and that absence is
# reported rather than hidden.
POWER_STREAMS = GEN_STREAMS + ["load"]

# Measured or modelled observations, shared by every node. These repeat
# across the overlapping forecast windows exactly like generation does.
ACTUAL_GROUPS = {
    "noaa": ["a_temperature", "a_cloud_cover", "a_wind_speed", "a_wind_dir"],
    "nsrdb": [
        "nsrdb_temp_air", "nsrdb_dhi", "nsrdb_dni", "nsrdb_ghi",
        "nsrdb_wind_speed", "nsrdb_air_pressure", "nsrdb_calc_cloud_coverage",
    ],
    "hub": ["hub_air_temp", "hub_wind_speed", "hub_wind_dir"],
}
OBSERVED_WEATHER = [c for group in ACTUAL_GROUPS.values() for c in group]

# Forecast weather. These genuinely differ per issuance, so they are the
# columns the as-of selection exists for.
FORECAST_WEATHER = [
    "temperature", "dew_point", "pressure", "ground_pressure", "humidity",
    "clouds", "wind_speed", "wind_deg", "rain", "snow", "ice", "fr_rain",
    "convective", "snow_depth", "accumulated", "rate", "probability",
]
HRRR_COLS = [
    "hb_temp_dew_point", "hb_pressure", "hb_humidity", "hb_cloud_cover",
    "hb_temp_ambient", "hb_wind_speed", "hb_wind_dir",
]
WWD_COLS = [
    "wwd_avg_air_temp", "wwd_avg_surface_air_pressure",
    "wwd_avg_wind_speed", "wwd_avg_wind_dir",
]

class IncompleteWindowWarning(UserWarning):
    """The window reaches past what any admissible issuance can cover.

    Raised, not silenced: the caller asked for N hours and is getting fewer,
    and the shortfall is a property of the forecast horizon rather than a
    gap in the file. What exists is still returned.
    """


_GEN_RE = re.compile(r"^(?P<node>[0-9a-z]+)_(?P<stream>dist_solar|solar|wind|load)$")


# ---------------------------------------------------------------------------
# time handling
# ---------------------------------------------------------------------------

def _ts(value, tz: str | None) -> pd.Timestamp:
    """Coerce anything date-like to a Timestamp matching the files' tz.

    Accepts dates, datetimes, strings and pandas Timestamps, so a window can
    start at an arbitrary hour rather than at midnight. The tz is taken from
    the parquet schema instead of being assumed, because a naive/aware
    mismatch is a TypeError at filter time, not a wrong answer - but only if
    it is handled in one place.
    """
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def window(start, hours: float, tz: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convenience: turn "09:00 on the 3rd, 72 hours out" into (start, end).

    This exists so callers can express a study window the way it is spoken
    without the horizon becoming an argument to the query itself. The
    returned end is inclusive, so `hours=72` yields 73 hourly points.

    Pass tz=catalog.time_tz to get bounds that compare directly against the
    timestamps in a returned frame.
    """
    s = _ts(start, tz)
    return s, s + pd.Timedelta(hours=hours)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    node: str
    path: Path
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    nulls: dict[str, int]
    size_mb: float
    streams: list[str] = field(default_factory=list)
    has_wind_weather: bool = False
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


class Catalog:
    """An index of what exists, built from parquet footers only.

    Reading footers means the whole catalog for 126 nodes costs about a
    second and no meaningful memory, so a caller can ask what is available
    before deciding what to load.
    """

    def __init__(self, root: str | Path, parquet_subdir: str = "fused_data/parquet"):
        self.root = Path(root)
        self.parquet_dir = self.root / parquet_subdir
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(
                f"no parquet folder at {self.parquet_dir}. "
                "Pass parquet_subdir if it lives somewhere else."
            )
        self.nodes: dict[str, NodeInfo] = {}
        self.time_tz: str | None = None
        for path in sorted(self.parquet_dir.glob("*.parquet")):
            info, tz = self._read_footer(path)
            self.nodes[info.node] = info
            if self.time_tz is None:
                self.time_tz = tz
        if not self.nodes:
            raise FileNotFoundError(f"no .parquet files in {self.parquet_dir}")

    @staticmethod
    def _read_footer(path: Path) -> tuple[NodeInfo, str | None]:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        md = pf.metadata
        cols = list(schema.names)

        tz = None
        if TIME_COL in schema.names:
            ftype = schema.field(TIME_COL).type
            if pa.types.is_timestamp(ftype):
                tz = ftype.tz

        nulls = {c: 0 for c in cols}
        span_min = span_max = None
        for rg in range(md.num_row_groups):
            group = md.row_group(rg)
            for i in range(group.num_columns):
                col_md = group.column(i)
                name = col_md.path_in_schema.split(".")[0]
                stats = col_md.statistics
                if stats is None:
                    continue
                if stats.null_count is not None and name in nulls:
                    nulls[name] += stats.null_count
                # Min/max on the target timestamp gives the covered span
                # without touching the data.
                if name == TIME_COL and stats.has_min_max:
                    lo, hi = stats.min, stats.max
                    span_min = lo if span_min is None else min(span_min, lo)
                    span_max = hi if span_max is None else max(span_max, hi)

        node = path.stem.split("_")[0]
        streams = []
        for c in cols:
            m = _GEN_RE.match(c)
            if m and m.group("node") == node:
                streams.append(m.group("stream"))

        info = NodeInfo(
            node=node,
            path=path,
            rows=md.num_rows,
            columns=cols,
            dtypes={f.name: str(f.type) for f in schema},
            nulls=nulls,
            size_mb=path.stat().st_size / 1e6,
            streams=sorted(streams),
            has_wind_weather=any(c.startswith(("hb_", "wwd_")) for c in cols),
            start=pd.Timestamp(span_min) if span_min is not None else None,
            end=pd.Timestamp(span_max) if span_max is not None else None,
        )
        return info, tz

    # -- convenience ------------------------------------------------------

    def node_names(self) -> list[str]:
        return list(self.nodes)

    def with_stream(self, stream: str) -> list[str]:
        return [n for n, i in self.nodes.items() if stream in i.streams]

    def span(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Earliest and latest target hour across every node."""
        starts = [i.start for i in self.nodes.values() if i.start is not None]
        ends = [i.end for i in self.nodes.values() if i.end is not None]
        return (min(starts) if starts else None, max(ends) if ends else None)

    def describe(self) -> pd.DataFrame:
        """One row per node: what it holds, how long it covers, how empty."""
        rows = []
        for n, i in self.nodes.items():
            rows.append(
                {
                    "node": n,
                    "rows": i.rows,
                    "columns": len(i.columns),
                    "size_mb": round(i.size_mb, 1),
                    "start": i.start,
                    "end": i.end,
                    "solar": "solar" in i.streams,
                    "wind": "wind" in i.streams,
                    "dist_solar": "dist_solar" in i.streams,
                    "load": "load" in i.streams,
                    "wind_weather": i.has_wind_weather,
                }
            )
        return pd.DataFrame(rows).sort_values("node").reset_index(drop=True)

    def column_availability(self) -> pd.DataFrame:
        """Per logical column: how many nodes have it, and how empty it is.

        Generation columns are grouped by the name AFTER the geohash prefix,
        so `9mudw2_solar` and `9mupsy_solar` count as one column present in
        two nodes rather than two columns present in one node each. Grouping
        on the raw name is what made the earlier availability report show a
        node count of 1 for every generation stream.
        """
        seen: dict[str, list[float]] = {}
        raw_names: dict[str, set[str]] = {}
        for i in self.nodes.values():
            if not i.rows:
                continue
            for c in i.columns:
                logical = logical_name(c, i.node)
                seen.setdefault(logical, []).append(i.nulls.get(c, 0) / i.rows)
                raw_names.setdefault(logical, set()).add(c)
        rows = [
            {
                "column": c,
                "nodes": len(v),
                "prefixed": len(raw_names[c]) > 1 or c in POWER_STREAMS,
                "mean_null": sum(v) / len(v),
                "min_null": min(v),
                "max_null": max(v),
            }
            for c, v in seen.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("mean_null", ascending=False)
            .reset_index(drop=True)
        )

    def dtype_outliers(self) -> pd.DataFrame:
        """Columns whose parquet dtype is not the same in every node file.

        Returns one row per (column, dtype) with the nodes involved, so a
        one-off int64 or all-null column can be named rather than counted.
        """
        seen: dict[tuple[str, str], list[str]] = {}
        for i in self.nodes.values():
            for c, t in i.dtypes.items():
                seen.setdefault((logical_name(c, i.node), t), []).append(i.node)
        by_col: dict[str, set[str]] = {}
        for (c, t) in seen:
            by_col.setdefault(c, set()).add(t)
        rows = [
            {
                "column": c,
                "dtype": t,
                "n_nodes": len(nodes),
                "nodes": ", ".join(sorted(nodes)[:8]) + ("..." if len(nodes) > 8 else ""),
            }
            for (c, t), nodes in seen.items()
            if len(by_col[c]) > 1
        ]
        if not rows:
            return pd.DataFrame(columns=["column", "dtype", "n_nodes", "nodes"])
        return (
            pd.DataFrame(rows)
            .sort_values(["column", "n_nodes"], ascending=[True, True])
            .reset_index(drop=True)
        )


def logical_name(column: str, node: str) -> str:
    """Strip the node geohash from a generation column name.

    Generation columns - and only generation columns - were prepended with
    the node geohash by the fusion pipeline. Every grouping, availability
    count and schema comparison has to be done on this name, not the raw one.
    """
    m = _GEN_RE.match(column)
    if m and m.group("node") == node:
        return m.group("stream")
    return column


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def _normalise(table: pa.Table, node: str) -> pa.Table:
    """Rename node-prefixed columns and make dtypes comparable across nodes.

    Without this, concatenating two nodes produces a sparse frame of
    one-node columns, and any column that is int64 in one file and double in
    another either raises or gets upcast without anyone being told. The
    index columns are left alone so hour_diff stays an integer offset.
    """
    names, cols = [], []
    for field_ in table.schema:
        col = table.column(field_.name)
        name = logical_name(field_.name, node)

        if name not in INDEX_COLS:
            if pa.types.is_integer(field_.type):
                col = pc.cast(col, pa.float64())
            elif pa.types.is_null(field_.type):
                # An all-empty column. Give it the type the other nodes use
                # so it lines up instead of blocking the concatenation.
                col = pc.cast(col, pa.float64())

        names.append(name)
        cols.append(col)
    return pa.Table.from_arrays(cols, names=names)


def _time_filter(start, end, tz):
    """Row-group predicate on the target hour, so unwanted groups are never
    decompressed."""
    preds = []
    if start is not None:
        preds.append((TIME_COL, ">=", _ts(start, tz)))
    if end is not None:
        preds.append((TIME_COL, "<=", _ts(end, tz)))
    return preds or None


def _forecast_filter(start, end, tz):
    """The target-hour window plus the causality cut, pushed to the reader.

    predict_time <= start is the cut that makes a forecast window a back
    test, and pushing it down is also what keeps the query cheap: roughly
    twenty issuances cover every target hour and all but one are discarded,
    so discarding them before decompression is the difference between
    reading a window and reading the file.
    """
    preds = _time_filter(start, end, tz) or []
    preds.append((ISSUE_COL, "<=", _ts(start, tz)))
    return preds


def _resolve_columns(info: NodeInfo, wanted: list[str]) -> tuple[list[str], list[str]]:
    """Map requested logical names onto the columns this node actually has.

    Returns (columns to read, names that are missing here). A name that is
    missing is reported to the caller, never filled with zeros.
    """
    read, missing = [], []
    for w in wanted:
        if w in POWER_STREAMS:
            prefixed = f"{info.node}_{w}"
            if prefixed in info.columns:
                read.append(prefixed)
            else:
                missing.append(w)
        elif w in info.columns:
            read.append(w)
        else:
            missing.append(w)
    return read, missing


def _dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_wide(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One column per (node, variable), indexed by target hour.

    This is the shape a solver wants: a rectangular block of node columns on
    a single time index.
    """
    wide = df.pivot(index=TIME_COL, columns="node_id", values=value_cols)
    wide.columns = [f"{node}_{var}" for var, node in wide.columns]
    return wide.sort_index().reset_index()


# ---------------------------------------------------------------------------
# read 1: the actual values
# ---------------------------------------------------------------------------

def get_actuals(
    catalog: Catalog,
    nodes: list[str],
    streams: list[str],
    start=None,
    end=None,
    layout: str = "long",
    check_duplicates: bool = False,
) -> pd.DataFrame:
    """Generation, load and measured weather, one row per node per hour.

    Rows in the fused files are keyed by forecast_time and the forecast
    windows overlap, so each target hour appears about twenty times with the
    same actual values copied onto each. Those duplicates are collapsed
    here; skipping that step inflates every total roughly twentyfold.

    That the copies really are identical was checked against a full node
    file: of the 43,938 distinct target hours, none had more than one value
    for any observed column, while the forecast columns differed in 43,926
    of them. Pass check_duplicates=True to re-run that check on whatever is
    being loaded now - it costs a group-by, so it is off by default.

    Note that no as-of rule applies here and none is needed. An actual
    value is not a prediction, so which issuance the row came from is
    irrelevant; every copy carries the same number. Missing forecast
    issuances therefore do not lose actual data, as long as at least one
    surviving issuance covered that hour.

    start and end may carry a time of day; they are not rounded to midnight.

    layout="long" returns node_id, forecast_time, then one column per stream.
    layout="wide" pivots to one column per (node, stream).
    """
    if not nodes:
        raise ValueError("no nodes selected")
    if not streams:
        raise ValueError("no streams selected")
    if layout not in ("long", "wide"):
        raise ValueError(f"unknown layout {layout!r}, use 'long' or 'wide'")

    tz = catalog.time_tz
    frames, notes = [], []
    for node in nodes:
        info = catalog.nodes.get(node)
        if info is None:
            notes.append(f"{node}: not in the catalog")
            continue

        read_cols, missing = _resolve_columns(info, streams)
        if missing:
            notes.append(f"{node}: no {', '.join(missing)}")
        if not read_cols:
            continue

        table = pq.read_table(
            info.path,
            columns=_dedupe([TIME_COL, OFFSET_COL] + read_cols),
            filters=_time_filter(start, end, tz),
        )
        if table.num_rows == 0:
            continue

        df = _normalise(table, node).to_pandas()

        # The row-group predicate prunes groups, it does not trim rows at the
        # edges of a surviving group, so the boundary is enforced here too.
        if start is not None:
            df = df[df[TIME_COL] >= _ts(start, tz)]
        if end is not None:
            df = df[df[TIME_COL] <= _ts(end, tz)]
        if df.empty:
            continue

        if check_duplicates:
            # hour_diff is the selection key, not data: it varies within a
            # group by construction and must not be checked for constancy.
            value_cols = [
                c for c in df.columns if c not in (TIME_COL, OFFSET_COL)
            ]
            varying = [
                c for c in value_cols
                if (df.groupby(TIME_COL)[c].nunique(dropna=True) > 1).any()
            ]
            if varying:
                raise AssertionError(
                    f"{node}: {', '.join(varying)} is not constant within a "
                    "forecast_time group, so collapsing the duplicates would "
                    "discard real variation"
                )

        # One row per target hour. The copies are identical, so any of them
        # would do; taking the smallest hour_diff is simply a rule that
        # always picks exactly one, and it is the same primitive the
        # forecast read uses. It is also robust to missing rollouts: if the
        # freshest issuance of an hour is absent, an older copy of the same
        # realised value is still there and still correct.
        keep = df.groupby(TIME_COL, sort=False)[OFFSET_COL].idxmin()
        df = df.loc[keep].sort_values(TIME_COL)
        df = df.drop(columns=[OFFSET_COL])
        df.insert(0, "node_id", node)
        frames.append(df)

    if not frames:
        raise ValueError(
            "nothing matched. " + ("; ".join(notes) if notes else "check the date range")
        )

    out = pd.concat(frames, ignore_index=True, sort=False)
    notes += _coverage_notes(out, start, end, tz)

    if layout == "wide":
        value_cols = [c for c in out.columns if c not in ("node_id", TIME_COL)]
        out = _to_wide(out, value_cols)

    out.attrs["notes"] = notes
    out.attrs["read"] = "actuals"
    return out


# ---------------------------------------------------------------------------
# read 2: the forecasts
# ---------------------------------------------------------------------------

def get_forecasts(
    catalog: Catalog,
    nodes: list[str],
    columns: list[str] | None = None,
    start=None,
    end=None,
    actuals: list[str] | None = None,
    layout: str = "long",
    keep_provenance: bool = True,
) -> pd.DataFrame:
    """The forecast that was available at `start`, for each hour of the
    window [start, end].

    The window is a FUTURE window seen from `start`: asking for it means
    standing at `start` and asking what the next end - start hours were
    expected to look like. That makes the read a back test, and it makes
    causality the governing constraint:

      * only issuances made at or before `start` are admissible, because an
        issuance made inside the window did not exist yet from that vantage
        point;
      * among those, each target hour takes the most recently issued
        forecast of itself, which is the smallest `hour_diff`;
      * `end` decides which target hours are returned and plays no part in
        choosing between issuances.

    Both bounds may be arbitrary times of day - 09:00 on the 3rd is a legal
    start even though issuances only happen at 00/06/12/18 - which is the
    point: a controller needs an input every hour while forecasts arrive
    every six.

    In this data the minimisation resolves to a single issuance for the
    whole window, so `predict_time` comes back constant and `hour_diff`
    climbs steadily; every source rolls out 120 hours on the same cadence,
    so the newest admissible issuance reaches further than any older one.
    The per-hour form is written anyway because it costs nothing and is the
    rule that also covers mixed sources with different rollout lengths,
    where an older, longer-reaching issuance has to serve the tail. A
    constant predict_time is then an observed result rather than an
    assumption.

    Selection uses the `hour_diff` index rather than arithmetic on
    timestamps. The fused files carry that column precisely so this is an
    index lookup. Where issuances are missing from the source file the
    minimum simply lands further out, which is the visible symptom of a gap
    and is reported in the notes rather than hidden.

    A window longer than MAX_HORIZON_HOURS cannot be completed, since no
    admissible issuance reaches that far. What exists is returned and an
    IncompleteWindowWarning is raised rather than the shortfall passing
    silently.

    `actuals` attaches the actual values for the same window, which is what
    turns the result from forecaster INPUT into forecaster TRAINING data. No
    join is needed: the fusion pipeline copies the actual value of a target
    hour onto every row that mentions it, so once the row is selected the
    actual power is already on it. Pass e.g. ["wind", "solar"].

    Returns one row per node per target hour, with predict_time / hour_diff
    kept as provenance so the selection can be audited.
    """
    if not nodes:
        raise ValueError("no nodes selected")
    if start is None or end is None:
        raise ValueError(
            "a forecast window needs both start and end; together they "
            "delimit which target hours are returned."
        )
    if layout not in ("long", "wide"):
        raise ValueError(f"unknown layout {layout!r}, use 'long' or 'wide'")

    tz = catalog.time_tz
    t0, t1 = _ts(start, tz), _ts(end, tz)
    if t1 < t0:
        raise ValueError(f"end ({t1}) is before start ({t0})")

    cols = list(columns) if columns else list(FORECAST_WEATHER)
    want_actuals = list(actuals) if actuals else []

    frames, notes = [], []

    for node in nodes:
        info = catalog.nodes.get(node)
        if info is None:
            notes.append(f"{node}: not in the catalog")
            continue

        read_cols, missing = _resolve_columns(info, cols)
        if missing:
            notes.append(f"{node}: no forecast column {', '.join(missing)}")
        actual_cols, missing_actual = _resolve_columns(info, want_actuals)
        if missing_actual:
            notes.append(f"{node}: no actual column {', '.join(missing_actual)}")
        if not read_cols and not actual_cols:
            continue

        table = pq.read_table(
            info.path,
            columns=_dedupe(
                [TIME_COL, ISSUE_COL, OFFSET_COL]
                + read_cols + actual_cols
            ),
            filters=_forecast_filter(t0, t1, tz),
        )
        if table.num_rows == 0:
            notes.append(f"{node}: no rows in the window at all")
            continue

        df = _normalise(table, node).to_pandas()

        # Re-apply both cuts on the rows themselves. The parquet predicate
        # prunes row groups; it does not trim inside a surviving group, and
        # the causality cut is the one that must not be approximate.
        df = df[(df[TIME_COL] >= t0) & (df[TIME_COL] <= t1) & (df[ISSUE_COL] <= t0)]
        if df.empty:
            notes.append(f"{node}: nothing was issued at or before {t0}")
            continue

        # The selection itself: among the admissible issuances, each target
        # hour keeps the row with the smallest offset - the most recently
        # issued forecast of that hour. Applied per hour rather than once so
        # that a shorter newest issuance can hand the tail of the window
        # back to an older, longer-reaching one.
        keep = df.groupby(TIME_COL, sort=False)[OFFSET_COL].idxmin()
        df = df.loc[keep].sort_values(TIME_COL)

        df.insert(0, "node_id", node)
        frames.append(df)

    if not frames:
        raise ValueError(
            "nothing matched. " + ("; ".join(notes) if notes else "check the window")
        )

    out = pd.concat(frames, ignore_index=True, sort=False)

    ordered = (
        ["node_id", TIME_COL]
        + [c for c in PROVENANCE_COLS if c in out.columns]
        + [c for c in out.columns
           if c not in ("node_id", TIME_COL) and c not in PROVENANCE_COLS]
    )
    out = out[ordered]

    assert_no_leakage(out, t0)
    notes += _coverage_notes(out, t0, t1, tz)
    notes += _staleness_notes(out, t0)
    notes += _horizon_warning(out, t0, t1, tz)

    if layout == "wide":
        value_cols = [
            c for c in out.columns
            if c not in ("node_id", TIME_COL) and c not in PROVENANCE_COLS
        ]
        out = _to_wide(out, value_cols)
        notes.append(
            "wide layout drops predict_time / hour_diff, which are per row "
            "and cannot be pivoted; use the long layout to audit selection."
        )
    elif not keep_provenance:
        out = out.drop(columns=[c for c in PROVENANCE_COLS if c in out.columns])

    _incomplete = out.attrs.get("incomplete")
    out.attrs["notes"] = notes
    out.attrs["read"] = "forecasts"
    out.attrs["as_of"] = t0
    out.attrs["window"] = (t0, t1)
    if _incomplete:
        out.attrs["incomplete"] = _incomplete
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def assert_no_leakage(df: pd.DataFrame, cutoff) -> None:
    """Fail loudly if any row was issued after the moment it claims to know.

    A forecast window is a back test: the caller is pretending to stand at
    `cutoff` and see only what existed then. A row issued inside the window
    breaks that pretence, and it is the one error here that would not
    announce itself - a leaking dataset trains and scores perfectly well, it
    is simply wrong. So it is asserted rather than assumed, on every read.
    """
    if cutoff is None or df.empty or ISSUE_COL not in df.columns:
        return
    bad = df[df[ISSUE_COL] > cutoff]
    if len(bad):
        raise AssertionError(
            f"{len(bad):,} rows were issued after {cutoff}, which is inside "
            f"the window. Latest offending issuance: {bad[ISSUE_COL].max()}"
        )


def _coverage_notes(df: pd.DataFrame, start, end, tz) -> list[str]:
    """Say which target hours of the requested window did not come back."""
    if start is None or end is None or df.empty:
        return []
    expected = pd.date_range(_ts(start, tz), _ts(end, tz), freq="h")
    notes = []
    for node, part in df.groupby("node_id", sort=True):
        gaps = expected.difference(pd.DatetimeIndex(part[TIME_COL]))
        if len(gaps):
            notes.append(
                f"{node}: {len(gaps)} of {len(expected)} target hours absent "
                f"({gaps[0]} to {gaps[-1]})"
            )
    return notes


def _staleness_notes(df: pd.DataFrame, t0: pd.Timestamp) -> list[str]:
    """Report how fresh the forecasts available at the window start were.

    Every admissible issuance predates `start`, and a newer issuance reaches
    further than an older one, so a healthy window is served end to end by
    one issuance made within the last six hours. Two things spoil that, and
    both are defects in the source file rather than faults in the selection:

      * the newest issuance before `start` is much older than six hours,
        because the issuances in between are absent;
      * some hours fall back on an older issuance, because the newest
        rollout has holes in it or stops short of the window - the mixed
        source case the per-hour minimisation exists to handle.
    """
    if df.empty or ISSUE_COL not in df.columns:
        return []
    notes = []
    for node, part in df.groupby("node_id", sort=True):
        newest = part[ISSUE_COL].max()
        age = (t0 - newest) / pd.Timedelta(hours=1)
        if age > ISSUE_INTERVAL_HOURS:
            notes.append(
                f"{node}: the freshest issuance available at {t0} was "
                f"{age:.0f} h old. Issuances are normally at most "
                f"{ISSUE_INTERVAL_HOURS} h apart, so issuances are missing "
                "from the source file around this window."
            )
        fallback = int((part[ISSUE_COL] < newest).sum())
        if fallback:
            notes.append(
                f"{node}: {fallback} of {len(part)} hour(s) fall back on an "
                f"issuance older than {newest}, so that rollout either has "
                "holes or stops short of the window."
            )
    return notes


def _horizon_warning(df, t0, t1, tz) -> list[str]:
    """Raise IncompleteWindowWarning when the window outruns the rollout.

    The caller asked for a span no forecast made at or before `start` can
    cover. What exists is returned, but silently handing back fewer hours
    than were asked for is how a truncated back test gets mistaken for a
    complete one, so this says so out loud.
    """
    if df.empty:
        return []
    asked = len(pd.date_range(_ts(t0, tz), _ts(t1, tz), freq="h"))
    span_hours = (_ts(t1, tz) - _ts(t0, tz)) / pd.Timedelta(hours=1)
    if span_hours <= MAX_HORIZON_HOURS:
        return []

    got = int(df.groupby("node_id")[TIME_COL].nunique().min())
    message = (
        f"asked for {span_hours:.0f} h of forecast from {t0}, but a single "
        f"issuance only rolls out {MAX_HORIZON_HOURS} h, so this window "
        f"cannot be completed. Returning the {got} of {asked} target hours "
        "that exist; the rest could not have been known at the window start."
    )
    warnings.warn(message, IncompleteWindowWarning, stacklevel=3)
    df.attrs["incomplete"] = message
    return [message]


def validate_forecast_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """A pass/fail table for a forecast extraction, one row per check.

    Written to be shown rather than trusted: this is what to put on screen
    when demonstrating that the window logic is right. The check that
    matters most is causality - a frame carrying forecasts published inside
    the window trains and scores perfectly well and is simply wrong.
    """
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    if df.empty:
        return pd.DataFrame([{"check": "non-empty", "pass": False, "detail": "no rows"}])

    tz = df[TIME_COL].dt.tz if hasattr(df[TIME_COL], "dt") else None
    if tz is not None:
        t0 = t0.tz_localize(tz) if t0.tzinfo is None else t0.tz_convert(tz)
        t1 = t1.tz_localize(tz) if t1.tzinfo is None else t1.tz_convert(tz)

    expected = pd.date_range(t0, t1, freq="h")
    rows = []

    per_node = df.groupby("node_id")[TIME_COL]
    dup = int((per_node.value_counts() > 1).sum()) if len(df) else 0
    rows.append({
        "check": "one row per node per target hour",
        "pass": dup == 0,
        "detail": f"{dup} duplicated (node, hour) pairs",
    })

    inside = bool((df[TIME_COL] >= t0).all() and (df[TIME_COL] <= t1).all())
    rows.append({
        "check": "every row inside the window",
        "pass": inside,
        "detail": f"{df[TIME_COL].min()} to {df[TIME_COL].max()}",
    })

    if ISSUE_COL in df.columns:
        late = int((df[ISSUE_COL] > t0).sum())
        rows.append({
            "check": f"no issuance from inside the window (after {t0})",
            "pass": late == 0,
            "detail": f"{late} anti-causal rows; latest issuance "
                      f"{df[ISSUE_COL].max()}",
        })

        used = int(df.groupby("node_id")[ISSUE_COL].nunique().max())
        rows.append({
            "check": "one issuance serves the whole window",
            "pass": used == 1,
            "detail": f"{used} issuance(s) used; more than one means the "
                      "newest rollout has holes or stops short",
        })

    if OFFSET_COL in df.columns:
        off = df[OFFSET_COL].astype("int64")
        rows.append({
            "check": f"offsets within the {MAX_HORIZON_HOURS} h rollout",
            "pass": bool(off.max() <= MAX_HORIZON_HOURS),
            "detail": f"hour_diff spans {int(off.min())} to {int(off.max())}",
        })

    covered = df.groupby("node_id")[TIME_COL].nunique()
    span_hours = (t1 - t0) / pd.Timedelta(hours=1)
    rows.append({
        "check": f"all {len(expected)} target hours present for every node",
        "pass": bool((covered == len(expected)).all()),
        "detail": ", ".join(f"{n}: {int(c)}" for n, c in covered.items())
                  + (f" (window is {span_hours:.0f} h, past the "
                     f"{MAX_HORIZON_HOURS} h rollout)"
                     if span_hours > MAX_HORIZON_HOURS else ""),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the stream folders (path A)
# ---------------------------------------------------------------------------

def scan_stream_dirs(root: str | Path) -> pd.DataFrame:
    """Report what is inside the node_* folders, without assuming a layout.

    Path A - the four pre-split stream files - is the other half of the
    timing comparison, and it is the only place load exists at all, since no
    fused file currently carries a load column. The layout of those folders
    has not been confirmed, so this inspects them and reports rather than
    guessing.
    """
    root = Path(root)
    rows = []
    for d in sorted(root.glob("node_*")):
        if not d.is_dir():
            continue
        files = [p for p in sorted(d.iterdir()) if p.is_file()]
        rows.append(
            {
                "folder": d.name,
                "files": len(files),
                "total_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
                "extensions": ", ".join(sorted({p.suffix or "(none)" for p in files})),
                "example": files[0].name if files else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def _stringify(value):
    """Render an attrs value in a form json can encode."""
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def export(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    fmt: str = "parquet",
) -> Path:
    """Write the result and return the path.

    parquet keeps the dtypes and is roughly ten times smaller; csv is there
    for handing the numbers to someone who will open them in a spreadsheet.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "extract"

    if fmt == "parquet":
        path = out_dir / f"{safe}.parquet"
        # pandas JSON-encodes df.attrs into the parquet metadata, and the
        # window bounds stored there are Timestamps, which it cannot encode.
        # Write a copy whose attrs are strings rather than dropping the
        # provenance: knowing the as-of moment of an exported file matters.
        out = df.copy(deep=False)
        out.attrs = {k: _stringify(v) for k, v in df.attrs.items()}
        out.to_parquet(path, compression="zstd", index=False)
    elif fmt == "csv":
        path = out_dir / f"{safe}.csv"
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unknown format {fmt!r}, use 'parquet' or 'csv'")
    return path
