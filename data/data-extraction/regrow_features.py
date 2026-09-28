"""Feature and target matrices for training forecasters on REGROW data.

This module turns the fused node files into the two arrays a statistical
forecaster is fitted on. It does no fitting and no train/test splitting; it
only builds X and Y, correctly and causally. Everything it reads goes through
regrow_query, so the causal forecast rule and the column normalisation are
the ones already validated there rather than a second copy of them.

--------------------------------------------------------------------------
The paradigm
--------------------------------------------------------------------------

A forecaster here is never a one-step model iterated forward. It predicts
the whole rollout at once: standing at an origin time T0, it outputs the
next H hours as a single vector in R^H. So the target is a MATRIX:

    Y[i, h-1] = realised value of the target at T0_i + h,   h = 1 .. H

and every row of X holds only what was genuinely available at T0_i.

X is assembled from up to three blocks, each switchable, so that the value
of each kind of information can be tested by leaving it out:

    forecast        Forecast weather for T0+1 .. T0+H, taken from the
                    issuances made at or before T0 (the causal rule of
                    regrow_query.get_forecasts). Forward-looking, but
                    generated in the past. On by default.

    autoregressive  The realised target streams themselves at T0+lag, for
                    lag <= 0. Lag 0 is "now". Off by default.

    observed        Realised (measured or modelled-actual) weather at
                    T0+lag, for lag <= 0. Same causal window as the
                    autoregressive block. Off by default.

With only the forecast block this is the pure exogenous model; adding the
autoregressive block gives the exogenous-plus-autoregressive model.

--------------------------------------------------------------------------
Shared X, separate Y
--------------------------------------------------------------------------

Several power streams at one node share one feature matrix and get one
target matrix each: Y["wind"], Y["solar"], Y["dist_solar"], and later
Y["load"]. When the autoregressive block is on, X carries the lags of every
stream in `ar_streams`, which defaults to the targets, so that X is still
shared across them.

--------------------------------------------------------------------------
Causality
--------------------------------------------------------------------------

This is the property that would not announce itself if it were broken: a
leaking X trains and scores perfectly well and is simply wrong. So it is
enforced rather than assumed, in three places:

  * lags must be <= 0; a positive lag would put the target into its own
    features, and is refused outright;
  * every forecast cell is checked to have been issued at or before its
    origin;
  * check_against_reference() recomputes sampled rows with the already
    validated get_forecasts / get_actuals and compares them cell by cell.

--------------------------------------------------------------------------
Missing data
--------------------------------------------------------------------------

Samples are never dropped. An origin too close to the start of the data for
its lags, too close to the end for its targets, or sitting on a gap in the
source file keeps its row, with NaN where the value does not exist, and is
marked False in the `valid` mask. Dropping rows per node would leave the
nodes with different sample counts and make them impossible to stack, so
the mask is the one consistent way to handle it: fit on X[valid], Y[valid].

--------------------------------------------------------------------------
Typical use
--------------------------------------------------------------------------

    import regrow_query as rq
    import regrow_features as rf

    cat = rq.Catalog(os.environ["REGROW_ROOT"])   # or a literal path
    origins = rf.make_origins("2019-01-01", "2019-12-31", tz=cat.time_tz)

    # the building block: one node, one or more streams sharing X
    m = rf.build_node_matrices(
        cat, "9x0gc5", targets=["wind"], origins=origins, horizon=48,
        forecast_columns=["temperature", "wind_speed", "wind_deg", "clouds"],
        ar_lags=[0, -1, -2, -24],
    )
    X, Y = m.X[m.valid], m.Y["wind"][m.valid]

    # the same thing stacked across nodes, with a node -> index map
    t = rf.build_tensor(cat, cat.with_stream("wind"), ["wind"], origins)
    t.X.shape            # (n_nodes, n_origins, n_features)
    t.node_index["9x0gc5"]
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import regrow_query as rq

__all__ = [
    "make_origins",
    "build_node_matrices",
    "build_tensor",
    "stack_nodes",
    "check_against_reference",
    "describe_dimensions",
    "estimate_bytes",
    "n_features_for",
    "load_matrices",
    "NodeMatrices",
    "MatrixTensor",
    "DEFAULT_HORIZON",
]

HOUR = pd.Timedelta(hours=1)

# Two days, the rollout used as the running example throughout.
DEFAULT_HORIZON = 48

# Block names, in the order they appear along the feature axis.
BLOCKS = ("forecast", "autoregressive", "observed")


# ---------------------------------------------------------------------------
# origins
# ---------------------------------------------------------------------------

def make_origins(start, end, every: str = "1h", tz: str | None = None) -> pd.DatetimeIndex:
    """The decision moments to build one sample for, from start to end.

    Hourly by default, which is how often a receding-horizon controller
    re-plans. every="6h" instead gives one origin per issuance cycle. Both
    bounds are inclusive and must fall on the hour, since the data is
    hourly; pass tz=catalog.time_tz so they compare directly with the files.
    """
    idx = pd.date_range(rq._ts(start, tz), rq._ts(end, tz), freq=every)
    return _check_origins(idx, tz)


def _check_origins(origins, tz) -> pd.DatetimeIndex:
    idx = _to_ns(origins, tz)
    if len(idx) == 0:
        raise ValueError("no origins given")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        idx = idx.unique().sort_values()
    off_hour = idx[(idx.minute != 0) | (idx.second != 0) | (idx.nanosecond != 0)
                   | (idx.microsecond != 0)]
    if len(off_hour):
        raise ValueError(
            f"origins must fall on the hour, since the data is hourly; "
            f"{len(off_hour)} do not, the first being {off_hour[0]}"
        )
    return idx


def _to_ns(values, tz) -> pd.DatetimeIndex:
    """A DatetimeIndex in nanoseconds and in the files' timezone.

    Parquet hands back millisecond timestamps and pandas builds nanosecond
    ones, and merge_asof refuses to align the two, so every timestamp that
    takes part in a join is normalised here first.
    """
    idx = pd.DatetimeIndex(values)
    if tz is None:
        idx = idx.tz_localize(None) if idx.tz is not None else idx
    else:
        idx = idx.tz_localize(tz) if idx.tz is None else idx.tz_convert(tz)
    return idx.as_unit("ns")


def _check_lags(lags, name) -> tuple[int, ...]:
    """Lags as a sorted tuple, newest first; any positive lag is refused."""
    if lags is None:
        return ()
    out = sorted({int(x) for x in lags}, reverse=True)
    future = [x for x in out if x > 0]
    if future:
        raise ValueError(
            f"{name} {future} point into the future relative to the origin. "
            "A positive lag would put values from the forecast window into "
            "the features, which is exactly the leak this module exists to "
            "prevent. Use lags <= 0; lag 0 is the origin itself."
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read(info, columns, filters, node) -> pd.DataFrame:
    table = pq.read_table(info.path, columns=rq._dedupe(columns), filters=filters)
    return rq._normalise(table, node).to_pandas()


def _forecast_block(info, node, columns, origins, horizon, tz):
    """Forecast features for every (origin, step) pair in one vectorised pass.

    For each origin T0 and step h, the cell is the forecast of hour T0 + h
    from the most recent issuance made at or before T0. That is the same
    rule get_forecasts applies per window, and for a fixed target hour the
    most recent admissible issuance is exactly the smallest admissible
    hour_diff. Computing it once per window would re-read the file once per
    origin; here all origins are answered by a single as-of join, which is
    what makes a year of hourly origins practical.

    Returns the values as {column: (n, H)}, plus the hour_diff and the
    issuance age actually used for every cell, for auditing.
    """
    read_cols, missing = rq._resolve_columns(info, columns)
    if missing:
        raise ValueError(f"{node}: no forecast column {', '.join(missing)}")

    t_lo = origins[0] + HOUR
    t_hi = origins[-1] + horizon * HOUR
    cutoff = origins[-1]
    df = _read(
        info,
        [rq.TIME_COL, rq.ISSUE_COL, rq.OFFSET_COL] + read_cols,
        [(rq.TIME_COL, ">=", t_lo), (rq.TIME_COL, "<=", t_hi),
         (rq.ISSUE_COL, "<=", cutoff)],
        node,
    )
    df[rq.TIME_COL] = _to_ns(df[rq.TIME_COL], tz)
    df[rq.ISSUE_COL] = _to_ns(df[rq.ISSUE_COL], tz)
    df = df[(df[rq.TIME_COL] >= t_lo) & (df[rq.TIME_COL] <= t_hi)
            & (df[rq.ISSUE_COL] <= cutoff)]
    right = df.sort_values(rq.ISSUE_COL, kind="stable")

    n, H = len(origins), horizon
    as_of = origins.repeat(H)
    target = as_of + pd.to_timedelta(np.tile(np.arange(1, H + 1), n), unit="h")
    left = pd.DataFrame({"_row": np.arange(n * H), "_as_of": as_of,
                         rq.TIME_COL: _to_ns(target, tz)})
    left = left.sort_values("_as_of", kind="stable")

    joined = pd.merge_asof(
        left, right,
        left_on="_as_of", right_on=rq.ISSUE_COL,
        by=rq.TIME_COL, direction="backward",
    ).sort_values("_row")

    issued = joined[rq.ISSUE_COL]
    present = issued.notna().to_numpy()
    if (issued[present].to_numpy() > joined.loc[present, "_as_of"].to_numpy()).any():
        raise AssertionError(
            f"{node}: a forecast issued after its origin reached the "
            "features. This should be impossible and means the as-of join "
            "is misconfigured."
        )

    values = {c: joined[c].to_numpy(dtype=float).reshape(n, H) for c in columns}
    offsets = joined[rq.OFFSET_COL].to_numpy(dtype=float).reshape(n, H)
    age = ((joined["_as_of"] - issued) / HOUR).to_numpy(dtype=float).reshape(n, H)
    return values, offsets, age


def _realised_frame(info, node, columns, t_lo, t_hi, tz) -> pd.DataFrame:
    """One row per hour of the actual values, on a complete hourly grid.

    The copies of an actual value are identical across the ~20 rows that
    mention its hour, so any of them would do; the smallest hour_diff is
    simply the rule that always picks exactly one, as in get_actuals.
    Hours absent from the file come back as NaN rather than being skipped,
    so that positional lookups stay aligned.
    """
    read_cols, missing = rq._resolve_columns(info, columns)
    if missing:
        raise ValueError(f"{node}: no actual column {', '.join(missing)}")

    df = _read(
        info,
        [rq.TIME_COL, rq.OFFSET_COL] + read_cols,
        [(rq.TIME_COL, ">=", t_lo), (rq.TIME_COL, "<=", t_hi)],
        node,
    )
    df = df[(df[rq.TIME_COL] >= t_lo) & (df[rq.TIME_COL] <= t_hi)]
    grid = _to_ns(pd.date_range(t_lo, t_hi, freq="h"), tz)
    if df.empty:
        return pd.DataFrame(index=grid, columns=list(columns), dtype=float)

    keep = df.groupby(rq.TIME_COL, sort=False)[rq.OFFSET_COL].idxmin()
    df = df.loc[keep]
    df.index = _to_ns(df[rq.TIME_COL], tz)
    return df[list(columns)].astype(float).reindex(grid)


def _at(frame: pd.DataFrame, column: str, origins, offsets_h) -> np.ndarray:
    """frame[column] at origin + offset, as an (n_origins, n_offsets) array."""
    n, k = len(origins), len(offsets_h)
    times = origins.repeat(k) + pd.to_timedelta(np.tile(offsets_h, n), unit="h")
    return frame[column].reindex(times).to_numpy(dtype=float).reshape(n, k)


def estimate_bytes(
    n_nodes: int,
    n_origins: int,
    n_features: int,
    horizon: int,
    n_targets: int = 1,
    dtype=np.float64,
) -> int:
    """How much memory X and Y will occupy, before building them.

    A year of hourly origins over the whole network is not small: 126 nodes
    with every forecast column comes to several gigabytes in float64. Worth
    knowing before allocating rather than after.
    """
    item = np.dtype(dtype).itemsize
    return int(n_nodes) * int(n_origins) * (
        int(n_features) + int(n_targets) * int(horizon)
    ) * item


def n_features_for(
    forecast_columns, horizon, ar_streams=(), ar_lags=(), observed_columns=(), observed_lags=()
) -> int:
    """The width of X for a given configuration, without building it."""
    n = len(forecast_columns) * int(horizon)
    n += len(ar_streams) * len(ar_lags)
    n += len(observed_columns) * len(observed_lags)
    return n


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class NodeMatrices:
    """X and Y for one node, sharing one feature matrix across its targets.

    X          (n_origins, n_features)
    Y[target]  (n_origins, horizon), Y[t][i, h-1] is the target at origin+h
    valid      (n_origins,) every feature and every target present
    offsets    (n_origins, horizon) hour_diff of each forecast cell used
    age        (n_origins, horizon) hours between issuance and origin
    blocks     where each block sits along the feature axis
    """

    node: str
    origins: pd.DatetimeIndex
    X: np.ndarray
    Y: dict[str, np.ndarray]
    feature_names: list[str]
    blocks: dict[str, slice]
    horizon: int
    valid: np.ndarray
    offsets: np.ndarray
    age: np.ndarray
    notes: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def targets(self) -> list[str]:
        return list(self.Y)

    def target_names(self, target: str) -> list[str]:
        return [f"{target}@h+{h}" for h in range(1, self.horizon + 1)]

    def valid_for(self, target: str) -> np.ndarray:
        """Rows where every feature and this particular target are present.

        Wider than `valid` when a node's streams have gaps in different
        places, which matters when fitting the targets one at a time.
        """
        return ~np.isnan(self.X).any(axis=1) & ~np.isnan(self.Y[target]).any(axis=1)

    def features(self, *blocks: str) -> tuple[np.ndarray, list[str]]:
        """X restricted to some blocks, with matching names.

        This is how the value of a block is tested: build once with every
        block on, then compare fits on features("forecast") against
        features("forecast", "observed").
        """
        unknown = [b for b in blocks if b not in self.blocks]
        if unknown:
            raise KeyError(f"no block {unknown}; have {list(self.blocks)}")
        cols = np.concatenate([np.arange(self.X.shape[1])[self.blocks[b]]
                               for b in blocks]) if blocks else np.arange(0)
        return self.X[:, cols], [self.feature_names[c] for c in cols]

    def forecast_cube(self) -> tuple[np.ndarray, list[str]]:
        """The forecast block as (n_origins, horizon, n_forecast_columns).

        The same numbers as X[:, blocks["forecast"]], arranged step by step,
        which is the layout a sequence model wants.
        """
        cols = self.config["forecast_columns"]
        flat = self.X[:, self.blocks["forecast"]]
        cube = flat.reshape(len(self.origins), len(cols), self.horizon)
        return cube.transpose(0, 2, 1), list(cols)

    def step_times(self, i: int) -> pd.DatetimeIndex:
        """The hours sample i is about: origin + 1 … origin + horizon."""
        return pd.DatetimeIndex(
            self.origins[i] + pd.to_timedelta(np.arange(1, self.horizon + 1), unit="h")
        )

    def issue_times(self, i: int) -> pd.DatetimeIndex:
        """When the forecast behind each step of sample i was issued.

        NaT where no admissible issuance covered that step. Always at or
        before the origin, which is the property the whole module exists to
        guarantee, so it is worth being able to read it directly.
        """
        return pd.DatetimeIndex(
            self.origins[i] - pd.to_timedelta(self.age[i], unit="h")
        )

    def forecast_frame(self, i: int) -> pd.DataFrame:
        """The forecast block of sample i, one row per step.

        This is part of X, not of Y: every number here was available at the
        origin. `forecast issued` is when it was published and is always at
        or before the origin, which is the property worth being able to read
        straight off the screen.
        """
        H = self.horizon
        out = pd.DataFrame({
            "h": np.arange(1, H + 1),
            "target time": self.step_times(i),
        })
        cols = list(self.config["forecast_columns"])
        if not cols:
            return out
        out["forecast issued"] = self.issue_times(i)
        out["lead (h)"] = self.age[i]
        flat = self.X[i, self.blocks["forecast"]]
        cube = np.asarray(flat, dtype=float).reshape(len(cols), H).T
        for k, c in enumerate(cols):
            out[f"fc:{c}"] = cube[:, k]
        return out

    def target_frame(self, i: int, targets: list[str] | None = None) -> pd.DataFrame:
        """Row i of Y, one row per step - what actually happened.

        Deliberately a separate table from the features. Mixing the two in
        one view makes it easy to forget which side of the origin each
        number came from, and that is the one distinction that must never
        blur.
        """
        out = pd.DataFrame({
            "h": np.arange(1, self.horizon + 1),
            "target time": self.step_times(i),
        })
        for t in (targets or self.targets):
            out[f"Y:{t}"] = self.Y[t][i]
        return out

    def sample_frame(self, i: int, targets: list[str] | None = None) -> pd.DataFrame:
        """Features and targets of sample i side by side, for export.

        The union of forecast_frame and target_frame. Convenient to write
        out or eyeball in a script; for showing someone what a sample is,
        the two halves separately are clearer.
        """
        left = self.forecast_frame(i)
        right = self.target_frame(i, targets).drop(columns=["h", "target time"])
        return pd.concat([left, right], axis=1)

    def feature_layout(self) -> pd.DataFrame:
        """Which columns of X hold which variable.

        The blocks alone say where the forecast block ends; this says where
        each variable inside it starts, which is what you need to read a raw
        column index back into a meaning.
        """
        rows, H = [], self.horizon

        def add(block, names, width):
            start = self.blocks[block].start
            for name in names:
                stop = start + width - 1
                rows.append({
                    "block": block,
                    "variable": name,
                    "columns": f"{start}" if width == 1 else f"{start}\u2013{stop}",
                    "features": width,
                    "meaning": {
                        "forecast": f"h = 1 \u2026 {H}, i.e. T0+1 \u2026 T0+{H}",
                        "autoregressive": "lags " + ", ".join(
                            f"{l:+d}" for l in self.config["ar_lags"]),
                        "observed": "lags " + ", ".join(
                            f"{l:+d}" for l in self.config["observed_lags"]),
                    }[block],
                })
                start += width

        add("forecast", self.config["forecast_columns"], H)
        if self.config["ar_lags"]:
            add("autoregressive", self.config["ar_streams"],
                len(self.config["ar_lags"]))
        if self.config["observed_columns"]:
            add("observed", self.config["observed_columns"],
                len(self.config["observed_lags"]))
        return pd.DataFrame(
            rows, columns=["block", "variable", "columns", "features", "meaning"]
        )

    def lag_frame(self, i: int) -> pd.DataFrame:
        """The backward-looking features of sample i, with their hours.

        Rebuilt in the order the blocks were assembled, so the name, the lag
        and the column line up by construction rather than by parsing the
        feature name back apart.
        """
        rows = []
        plan = [
            ("autoregressive", self.config["ar_streams"], self.config["ar_lags"]),
            ("observed", self.config["observed_columns"], self.config["observed_lags"]),
        ]
        for block, names, lags in plan:
            start = self.blocks[block].start
            k = 0
            for name in names:
                for lag in lags:
                    rows.append({
                        "block": block,
                        "feature": self.feature_names[start + k],
                        "time": self.origins[i] + lag * HOUR,
                        "value": float(self.X[i, start + k]),
                    })
                    k += 1
        return pd.DataFrame(rows, columns=["block", "feature", "time", "value"])

    def to_frame(self, target: str | None = None) -> pd.DataFrame:
        """X and one target as a labelled DataFrame, one row per origin."""
        frames = [pd.DataFrame(self.X, index=self.origins, columns=self.feature_names)]
        for t in ([target] if target else self.targets):
            frames.append(pd.DataFrame(self.Y[t], index=self.origins,
                                       columns=self.target_names(t)))
        out = pd.concat(frames, axis=1)
        out.index.name = "origin"
        out.insert(0, "valid", self.valid)
        return out

    def save(self, path) -> Path:
        """Write the arrays and their labels to a compressed .npz file."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            valid=self.valid,
            offsets=self.offsets,
            origins=self.origins.asi8,
            feature_names=np.array(self.feature_names),
            **{f"Y__{t}": y for t, y in self.Y.items()},
            meta=np.array(json.dumps({
                "kind": "node", "node": self.node, "horizon": self.horizon,
                "tz": str(self.origins.tz) if self.origins.tz else None,
                "blocks": {k: [v.start, v.stop] for k, v in self.blocks.items()},
                "config": self.config, "notes": self.notes,
            })),
        )
        return path


@dataclass
class MatrixTensor:
    """X and Y stacked across nodes, with a node -> index mapping.

    X          (n_nodes, n_origins, n_features)
    Y[target]  (n_nodes, n_origins, horizon)
    valid      (n_nodes, n_origins)

    Every node shares the same origins and the same feature layout, so the
    node axis can be indexed, sliced or flattened freely. Nodes missing a
    requested target or feature column are left out rather than padded, and
    the reason is recorded in `skipped`.
    """

    nodes: list[str]
    node_index: dict[str, int]
    origins: pd.DatetimeIndex
    X: np.ndarray
    Y: dict[str, np.ndarray]
    feature_names: list[str]
    blocks: dict[str, slice]
    horizon: int
    valid: np.ndarray
    skipped: dict[str, str] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def node(self, name: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """(X, Y, valid) for one node, looked up by name."""
        i = self.node_index[name]
        return self.X[i], {t: y[i] for t, y in self.Y.items()}, self.valid[i]

    def save(self, path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            valid=self.valid,
            origins=self.origins.asi8,
            nodes=np.array(self.nodes),
            feature_names=np.array(self.feature_names),
            **{f"Y__{t}": y for t, y in self.Y.items()},
            meta=np.array(json.dumps({
                "kind": "tensor", "horizon": self.horizon,
                "tz": str(self.origins.tz) if self.origins.tz else None,
                "blocks": {k: [v.start, v.stop] for k, v in self.blocks.items()},
                "skipped": self.skipped, "config": self.config,
            })),
        )
        return path


def load_matrices(path) -> dict:
    """Read a file written by NodeMatrices.save or MatrixTensor.save.

    Returns a dict of plain arrays plus the metadata, with origins restored
    as a DatetimeIndex and Y collected under "Y".
    """
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        out = {k: z[k] for k in z.files if not k.startswith("Y__") and k != "meta"}
        out["Y"] = {k[3:]: z[k] for k in z.files if k.startswith("Y__")}
    origins = pd.DatetimeIndex(out["origins"].astype("datetime64[ns]"))
    out["origins"] = origins.tz_localize("UTC").tz_convert(meta["tz"]) if meta.get("tz") else origins
    out["feature_names"] = out["feature_names"].tolist()
    out["blocks"] = {k: slice(*v) for k, v in meta["blocks"].items()}
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# the building block
# ---------------------------------------------------------------------------

def build_node_matrices(
    catalog: rq.Catalog,
    node: str,
    targets: list[str] | str,
    origins,
    horizon: int = DEFAULT_HORIZON,
    forecast_columns: list[str] | None = None,
    ar_lags=(),
    ar_streams: list[str] | None = None,
    observed_columns: list[str] | None = None,
    observed_lags=(0,),
    dtype=np.float64,
) -> NodeMatrices:
    """The feature matrix and one target matrix per stream, for one node.

    targets           power streams to predict: "solar", "wind",
                      "dist_solar", "load". One Y each, all sharing X.
    origins           decision moments, one sample each (see make_origins).
    horizon           length of the rollout; Y has this many columns.
    forecast_columns  forecast weather for steps 1..horizon. Defaults to
                      every OpenWeather forecast column; pass [] to omit the
                      block.
    ar_lags           lags (<= 0) of the realised streams. Empty = off,
                      which is the pure exogenous model.
    ar_streams        whose lags to include. Defaults to the targets.
    observed_columns  actual weather to include at observed_lags. None =
                      off.
    dtype             storage type of X and Y. float64 is exact and is the
                      default; float32 halves the memory, which matters once
                      the node axis is added - see estimate_bytes.

    Raises ValueError when the node lacks a requested target or column.
    """
    tz = catalog.time_tz
    info = catalog.nodes.get(node)
    if info is None:
        raise ValueError(f"{node}: not in the catalog")

    targets = [targets] if isinstance(targets, str) else list(targets)
    if not targets:
        raise ValueError("no targets given")
    bad = [t for t in targets if t not in rq.POWER_STREAMS]
    if bad:
        raise ValueError(f"targets must be power streams {rq.POWER_STREAMS}; got {bad}")

    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError(
            f"dtype must be a floating type so that missing values can be "
            f"NaN; got {dtype}"
        )

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if horizon > rq.MAX_HORIZON_HOURS:
        warnings.warn(
            f"horizon {horizon} h exceeds the {rq.MAX_HORIZON_HOURS} h rollout, "
            "so the later steps can never be filled and every sample will be "
            "marked invalid.",
            rq.IncompleteWindowWarning, stacklevel=2,
        )

    origins = _check_origins(origins, tz)
    fc_cols = list(rq.FORECAST_WEATHER) if forecast_columns is None else list(forecast_columns)
    ar_lags = _check_lags(ar_lags, "ar_lags")
    ar_streams = list(targets) if ar_streams is None else list(ar_streams)
    if not ar_lags:
        ar_streams = []
    obs_cols = list(observed_columns) if observed_columns else []
    obs_lags = _check_lags(observed_lags, "observed_lags") if obs_cols else ()
    bad = [s for s in ar_streams if s not in rq.POWER_STREAMS]
    if bad:
        raise ValueError(f"ar_streams must be power streams; got {bad}")

    n, H = len(origins), horizon
    blocks_out, names, pieces = {}, [], []
    notes: list[str] = []

    # -- forecast block ------------------------------------------------------
    if fc_cols:
        fc, offsets, age = _forecast_block(info, node, fc_cols, origins, H, tz)
        for c in fc_cols:
            pieces.append(fc[c])
            names += [f"fc:{c}@h+{h}" for h in range(1, H + 1)]
    else:
        offsets = np.full((n, H), np.nan)
        age = np.full((n, H), np.nan)
    blocks_out["forecast"] = slice(0, len(names))

    # -- everything realised, read in one pass --------------------------------
    realised_cols = rq._dedupe(targets + ar_streams + obs_cols)
    all_lags = list(ar_lags) + list(obs_lags) + [0]
    t_lo = origins[0] + min(all_lags) * HOUR
    t_hi = origins[-1] + H * HOUR
    frame = _realised_frame(info, node, realised_cols, t_lo, t_hi, tz)

    start = len(names)
    for s in ar_streams:
        pieces.append(_at(frame, s, origins, list(ar_lags)))
        names += [f"ar:{s}@t{lag:+d}" for lag in ar_lags]
    blocks_out["autoregressive"] = slice(start, len(names))

    start = len(names)
    for c in obs_cols:
        pieces.append(_at(frame, c, origins, list(obs_lags)))
        names += [f"obs:{c}@t{lag:+d}" for lag in obs_lags]
    blocks_out["observed"] = slice(start, len(names))

    X = (np.hstack(pieces) if pieces else np.empty((n, 0))).astype(dtype, copy=False)
    Y = {t: _at(frame, t, origins, list(range(1, H + 1))).astype(dtype, copy=False)
         for t in targets}

    x_ok = ~np.isnan(X).any(axis=1)
    y_ok = np.logical_and.reduce([~np.isnan(y).any(axis=1) for y in Y.values()])
    valid = x_ok & y_ok

    # -- say why rows are invalid, per block ---------------------------------
    def _bad(block):
        sl = blocks_out[block]
        return int(np.isnan(X[:, sl]).any(axis=1).sum()) if sl.stop > sl.start else 0

    if not valid.all():
        notes.append(f"{node}: {int((~valid).sum())} of {n} origins incomplete")
        for block in BLOCKS:
            k = _bad(block)
            if k:
                notes.append(f"{node}:   {block} block missing in {k} origins")
        for t, y in Y.items():
            k = int(np.isnan(y).any(axis=1).sum())
            if k:
                notes.append(f"{node}:   target {t} missing in {k} origins")

    # A healthy origin is served by an issuance made within one cycle.
    stale = np.nanmax(age, axis=1) > rq.ISSUE_INTERVAL_HOURS - 1 if fc_cols else np.zeros(n, bool)
    stale &= ~np.isnan(age).all(axis=1)
    if stale.any():
        notes.append(
            f"{node}: {int(stale.sum())} origins had no issuance within the "
            f"preceding {rq.ISSUE_INTERVAL_HOURS} h, so their forecasts are "
            "staler than usual; those issuances are missing from the file"
        )

    config = {
        "targets": targets, "horizon": H,
        "forecast_columns": fc_cols,
        "ar_streams": ar_streams, "ar_lags": list(ar_lags),
        "observed_columns": obs_cols, "observed_lags": list(obs_lags),
        "origin_first": str(origins[0]), "origin_last": str(origins[-1]),
        "n_origins": n, "dtype": dtype.name,
    }
    return NodeMatrices(
        node=node, origins=origins, X=X, Y=Y, feature_names=names,
        blocks=blocks_out, horizon=H, valid=valid, offsets=offsets, age=age,
        notes=notes, config=config,
    )


# ---------------------------------------------------------------------------
# across nodes
# ---------------------------------------------------------------------------

def build_tensor(
    catalog: rq.Catalog,
    nodes: list[str],
    targets: list[str] | str,
    origins,
    horizon: int = DEFAULT_HORIZON,
    **kwargs,
) -> MatrixTensor:
    """build_node_matrices for each node, stacked along a leading node axis.

    Any subset of nodes works - all of them, three, one. A node enters the
    tensor only if it carries every target and every requested column; the
    rest are listed in `skipped` with the reason, so that no slot in the
    tensor is filled with something that is not there.
    """
    targets = [targets] if isinstance(targets, str) else list(targets)
    built, skipped = [], {}
    for node in nodes:
        try:
            built.append(
                build_node_matrices(catalog, node, targets, origins, horizon, **kwargs)
            )
        except ValueError as exc:
            skipped[node] = str(exc)
    return stack_nodes(built, skipped)


def stack_nodes(
    matrices: list[NodeMatrices],
    skipped: dict[str, str] | None = None,
) -> MatrixTensor:
    """Stack already-built node matrices along a leading node axis.

    build_tensor is this plus the loop. It is exposed separately so a caller
    that wants to keep the per-node objects - to inspect one node's forecast
    cube, say - can build them once and stack them, instead of building
    twice. The nodes must share origins, horizon, targets and feature
    layout, which they do whenever they came from the same arguments.
    """
    skipped = dict(skipped or {})
    if not matrices:
        raise ValueError(
            "no node carried every target and column requested. "
            + "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:5])
        )
    first = matrices[0]
    for m in matrices[1:]:
        if (not m.origins.equals(first.origins) or m.horizon != first.horizon
                or m.feature_names != first.feature_names
                or list(m.Y) != list(first.Y)):
            raise ValueError(
                f"{m.node} was built with different arguments from "
                f"{first.node}; only matrices sharing origins, horizon, "
                "targets and features can be stacked"
            )
    built = matrices
    notes = {m.node: m.notes for m in built if m.notes}
    targets = list(first.Y)
    return MatrixTensor(
        nodes=[m.node for m in built],
        node_index={m.node: i for i, m in enumerate(built)},
        origins=first.origins,
        X=np.stack([m.X for m in built]),
        Y={t: np.stack([m.Y[t] for m in built]) for t in targets},
        feature_names=first.feature_names,
        blocks=first.blocks,
        horizon=first.horizon,
        valid=np.stack([m.valid for m in built]),
        skipped=skipped,
        notes=notes,
        config=first.config,
    )


# ---------------------------------------------------------------------------
# describing the result
# ---------------------------------------------------------------------------

def describe_dimensions(obj) -> pd.DataFrame:
    """What every axis of X and Y means, with its actual contents.

    A shape like (2, 744, 201) says nothing on its own. This spells out that
    axis 0 is the node and which node is which, that axis 1 is the origin
    and which hours those are, and that axis 2 is the feature and which
    block occupies which columns. Works for a single node's matrices and for
    a stacked tensor; the single-node case simply has no node axis.
    """
    is_tensor = isinstance(obj, MatrixTensor)
    origins = obj.origins
    n = len(origins)
    step = (
        (origins[1] - origins[0]) / pd.Timedelta(hours=1) if n > 1 else float("nan")
    )
    blocks = " \u00b7 ".join(
        f"{b} {sl.start}\u2013{sl.stop - 1}"
        for b, sl in obj.blocks.items() if sl.stop > sl.start
    )
    target = next(iter(obj.Y))
    n_feat = obj.X.shape[-1]
    H = obj.horizon

    rows = []
    if is_tensor:
        who = ", ".join(f"{k} \u2192 {v}" for k, v in list(obj.node_index.items())[:6])
        if len(obj.node_index) > 6:
            who += ", \u2026"
        rows.append({"array": "X", "axis": 0, "size": len(obj.nodes),
                     "meaning": "node", "contents": who})
    rows.append({
        "array": "X", "axis": 1 if is_tensor else 0, "size": n,
        "meaning": "origin (T0) \u2014 one sample each",
        "contents": f"{origins[0]} \u2192 {origins[-1]}, every {step:.0f} h",
    })
    rows.append({
        "array": "X", "axis": 2 if is_tensor else 1, "size": n_feat,
        "meaning": "feature", "contents": blocks,
    })
    if is_tensor:
        rows.append({"array": "Y", "axis": 0, "size": len(obj.nodes),
                     "meaning": "node", "contents": "same order as X"})
    rows.append({
        "array": "Y", "axis": 1 if is_tensor else 0, "size": n,
        "meaning": "origin (T0)", "contents": "same origins as X",
    })
    rows.append({
        "array": "Y", "axis": 2 if is_tensor else 1, "size": H,
        "meaning": "forecast step",
        "contents": f"h = 1 \u2026 {H}, i.e. T0+1 \u2026 T0+{H} (Y[\"{target}\"])",
    })
    return pd.DataFrame(rows, columns=["array", "axis", "size", "meaning", "contents"])


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def check_against_reference(
    m: NodeMatrices,
    catalog: rq.Catalog,
    n_samples: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Recompute sampled rows with get_forecasts / get_actuals and compare.

    The forecast block above is computed by a vectorised as-of join rather
    than by calling get_forecasts once per origin, for speed. This checks
    that the two agree cell for cell on a random sample of valid origins,
    so the fast path inherits the reference implementation's correctness
    instead of asking to be trusted. Autoregressive cells are checked the
    same way against get_actuals.
    """
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(m.valid)
    if len(candidates) == 0:
        return pd.DataFrame(columns=["origin", "part", "cells", "max_abs_diff", "match"])
    picks = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)
    H = m.horizon
    fc_cols = m.config["forecast_columns"]
    rows = []

    for i in sorted(picks):
        t0 = m.origins[i]

        if fc_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", rq.IncompleteWindowWarning)
                ref = rq.get_forecasts(catalog, [m.node], fc_cols,
                                       start=t0, end=t0 + H * HOUR)
            ref = ref.set_index(rq.TIME_COL)
            ref.index = _to_ns(ref.index, catalog.time_tz)
            steps = _to_ns(t0 + pd.to_timedelta(np.arange(1, H + 1), unit="h"),
                           catalog.time_tz)
            want = np.concatenate([ref[c].reindex(steps).to_numpy(float) for c in fc_cols])
            got = m.X[i, m.blocks["forecast"]]
            diff = np.nanmax(np.abs(want - got)) if len(got) else 0.0
            rows.append({"origin": t0, "part": "forecast", "cells": len(got),
                         "max_abs_diff": float(diff),
                         "match": bool(np.allclose(want, got, equal_nan=True))})

        lags = m.config["ar_lags"]
        for s in m.config["ar_streams"]:
            lo, hi = t0 + min(lags) * HOUR, t0 + max(lags) * HOUR
            act = rq.get_actuals(catalog, [m.node], [s], start=lo, end=hi)
            act = act.set_index(rq.TIME_COL)[s]
            act.index = _to_ns(act.index, catalog.time_tz)
            times = _to_ns(t0 + pd.to_timedelta(list(lags), unit="h"), catalog.time_tz)
            want = act.reindex(times).to_numpy(float)
            cols = [m.feature_names.index(f"ar:{s}@t{lag:+d}") for lag in lags]
            got = m.X[i, cols]
            rows.append({"origin": t0, "part": f"ar:{s}", "cells": len(got),
                         "max_abs_diff": float(np.nanmax(np.abs(want - got))),
                         "match": bool(np.allclose(want, got, equal_nan=True))})

        for t in m.targets:
            act = rq.get_actuals(catalog, [m.node], [t],
                                 start=t0 + HOUR, end=t0 + H * HOUR)
            act = act.set_index(rq.TIME_COL)[t]
            act.index = _to_ns(act.index, catalog.time_tz)
            steps = _to_ns(t0 + pd.to_timedelta(np.arange(1, H + 1), unit="h"),
                           catalog.time_tz)
            want = act.reindex(steps).to_numpy(float)
            got = m.Y[t][i]
            rows.append({"origin": t0, "part": f"Y:{t}", "cells": len(got),
                         "max_abs_diff": float(np.nanmax(np.abs(want - got))),
                         "match": bool(np.allclose(want, got, equal_nan=True))})

    return pd.DataFrame(rows)
