import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="REGROW data extractor")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # REGROW data extractor

        Pick nodes, pick columns, pick a window, get one merged table out and
        write it to a file.

        The logic lives in `regrow_query.py` next to this notebook, so the same
        functions can be imported into a script or another notebook. This page
        is the interactive front end for them, not a copy of them.

        ## The two timestamps

        Every row of a fused file carries two times, and telling them apart is
        the whole game:

        | column | meaning |
        |---|---|
        | `predict_day` / `predict_time` | when the forecast was **issued** |
        | `forecast_day` / `forecast_time` | the hour the forecast is **about** |
        | `day_diff` / `hour_diff` | `forecast_time` − `predict_time` |

        Forecasts were issued every six hours and each issuance rolls out 120
        hours, so any target hour is covered by roughly twenty issuances. The
        relationship is **many forecasts to one reality**, and that splits the
        file into two kinds of column that have to be read differently:

        - **Actual values** — generation, load, NOAA / NSRDB / hub weather.
          One true value per target hour, copied onto all ~20 rows that mention
          it. They are collapsed here. Skipping that multiplies every total by
          about twenty. No selection rule applies: an actual number is not a
          prediction, so a missing issuance does not lose it, as long as some
          surviving issuance covered that hour.
        - **Forecast weather** — genuinely different on every row, so exactly
          one row per target hour has to be *chosen*, and which one is chosen
          is a modelling decision.


        ## The causal rule

        A forecast window `[start, end]` is a **future window seen from
        `start`**. Asking for it means standing at `start` and asking what
        the next `end − start` hours were expected to look like — which is
        the whole point of a back test. So:

        1. Only issuances made **at or before `start`** are admissible. One
           made *inside* the window did not exist yet from that vantage
           point, and returning it leaks the future into the back test.
        2. Among those, each target hour takes the **most recently issued
           forecast of itself** — the smallest `hour_diff`.
        3. `end` decides which target hours are returned. It plays **no
           part** in choosing between issuances.

        In this data every source rolls out 120 h on the same six-hour
        cadence, so the newest admissible issuance reaches further than any
        older one and one issuance serves the whole window:

        | forecast_time | predict_time | hour_diff |
        |---|---|---|
        | 01-01 12:00 | 01-01 12:00 | 0 |
        | 01-01 13:00 | 01-01 12:00 | 1 |
        | 01-01 14:00 | 01-01 12:00 | 2 |
        | … | … | … |
        | 01-03 12:00 | 01-01 12:00 | 48 |

        `predict_time` comes back **constant** and `hour_diff` **climbs**.
        The minimisation is still applied per hour, because it costs nothing
        and it is the rule that also covers mixed sources with different
        rollout lengths — where the newest issuance stops short and an
        older, longer-reaching one has to serve the tail. A constant
        `predict_time` is then a *result*, not an assumption.

        Both bounds carry a time of day. A window may start at 09:00 even
        though issuances only happen at 00 / 06 / 12 / 18 — which is the
        point, since a controller needs an input every hour while forecasts
        arrive every six.

        **Two things to watch for.** A window longer than 120 h cannot be
        completed, because no admissible issuance reaches that far; what
        exists is returned with a warning. And more than one `predict_time`
        in the result means the newest rollout has holes or stops short — a
        defect to report, not a fault in the selection.

        None of this constrains the actual read: an actual value is not a
        prediction, so a window there may be as long as the data set.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    import datetime as dt

    import pandas as pd

    import regrow_query as rq

    return dt, pd, rq


@app.cell(hide_code=True)
def _(mo):
    import os

    root_input = mo.ui.text(
        value=os.environ.get("REGROW_ROOT", ""),
        label="REGROW folder  (defaults to $REGROW_ROOT)",
        full_width=True,
    )
    root_input
    return (root_input,)


@app.cell(hide_code=True)
def _(mo, root_input, rq):
    try:
        catalog = rq.Catalog(root_input.value.strip('"'))
        load_error = None
    except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
        catalog = None
        load_error = str(exc)

    mo.stop(
        catalog is None,
        mo.callout(
            mo.md(
                f"Could not build the catalog.\n\n```\n{load_error}\n```\n\n"
                "The folder should be the one holding `fused_data/parquet`."
            ),
            kind="danger",
        ),
    )

    catalog_df = catalog.describe()
    span_lo, span_hi = catalog.span()

    mo.callout(
        mo.md(
            f"**{len(catalog_df)} nodes** indexed from "
            f"`{catalog.parquet_dir}` — "
            f"{catalog_df['size_mb'].sum()/1024:.2f} GB, "
            f"{catalog_df['rows'].sum():,} rows, covering "
            f"{span_lo:%Y-%m-%d %H:%M} to {span_hi:%Y-%m-%d %H:%M} "
            f"({catalog.time_tz or 'tz-naive'})"
        ),
        kind="success",
    )
    return catalog, catalog_df, span_hi, span_lo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. What is available
    """)
    return


@app.cell(hide_code=True)
def _(catalog_df, mo):
    availability = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="utility solar", value=f"{int(catalog_df['solar'].sum())}"),
                    mo.stat(label="utility wind", value=f"{int(catalog_df['wind'].sum())}"),
                    mo.stat(label="distributed solar", value=f"{int(catalog_df['dist_solar'].sum())}"),
                    mo.stat(label="load", value=f"{int(catalog_df['load'].sum())}"),
                ],
                justify="start",
                gap=2,
            ),

        ]
    )
    availability
    return


@app.cell(hide_code=True)
def _(catalog, mo):
    _dtypes = catalog.dtype_outliers()
    schema_view = mo.vstack(
        [
            mo.md(
                "**Columns whose dtype is not the same in every node file.** "
                "Grouped by the name *after* the geohash, so `9mudw2_solar` and "
                "`9mupsy_solar` count as one column. An `int64` where every other "
                "node has `double` usually means that file holds only whole "
                "numbers — often all zeros — and a `null` type means the column is "
                "empty for every row. Both are named here rather than counted, so "
                "the offending geohash can go straight into the report."
            ),
            mo.ui.table(_dtypes, selection=None, page_size=10)
            if len(_dtypes)
            else mo.callout(mo.md("Every column has one dtype across all nodes."), kind="success"),
        ]
    )
    schema_view
    return


@app.cell(hide_code=True)
def _(catalog, mo):
    _avail = catalog.column_availability()
    emptiness_view = mo.vstack(
        [
            mo.md(
                "**How empty each column is, averaged over the nodes that have "
                "it.** `nodes` is the number of node files containing the column; "
                "`mean_null` / `min_null` / `max_null` are the share of rows where "
                "it is null. A column that should be everywhere and shows "
                "`max_null` well above zero is a defect to report, not a column to "
                "fill with zeros."
            ),
            mo.ui.table(_avail, selection=None, page_size=12),
        ]
    )
    emptiness_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Choose what to extract
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    read_kind = mo.ui.radio(
        options={
            "Actual (generation, load, measured weather)": "actuals",
            "Forecast weather (as known at the window start)": "forecasts",
        },
        value="Actual (generation, load, measured weather)",
        label="What kind of data",
    )
    asset_filter = mo.ui.multiselect(
        options={
            "has utility solar": "solar",
            "has utility wind": "wind",
            "has distributed solar": "dist_solar",
        },
        value=[],
        label="Narrow the node list to nodes that have any of",
    )
    mo.hstack([read_kind, asset_filter], justify="start", gap=3)
    return asset_filter, read_kind


@app.cell(hide_code=True)
def _(asset_filter, catalog_df, mo):
    _wanted = list(asset_filter.value)
    if not _wanted:
        # Nothing ticked means no filter at all rather than no nodes, so the
        # picker opens on the full list instead of an empty one.
        eligible = list(catalog_df["node"])
        _caption = (
            f"*All {len(catalog_df)} nodes. Tick an asset above to narrow "
            "the list.*"
        )
    else:
        # Union, not intersection: ticking solar and wind gives the nodes
        # carrying either, which is the set a run covering both needs.
        _mask = catalog_df[_wanted].any(axis=1)
        eligible = list(catalog_df.loc[_mask, "node"])
        _caption = (
            f"*Nodes carrying **any** of {', '.join(_wanted)}. A node with "
            "only one of them is included, and the columns it lacks are "
            "reported under the result.*"
        )

    node_picker = mo.ui.multiselect(
        options=eligible,
        value=eligible[:2],
        label=f"Nodes ({len(eligible)} available)",
    )
    mo.vstack(
        [
            node_picker,
            mo.md(_caption),
            mo.md(
                "*A column a node does not have is reported under the result "
                "rather than filled with zeros.*"
            ),
        ]
    )
    return (node_picker,)


@app.cell(hide_code=True)
def _(mo, read_kind, rq):
    if read_kind.value == "actuals":
        _options = (
            rq.GEN_STREAMS
            + ["load"]
            + rq.ACTUAL_GROUPS["noaa"]
            + rq.ACTUAL_GROUPS["nsrdb"]
            + rq.ACTUAL_GROUPS["hub"]
        )
        _default = ["solar", "wind"]
    else:
        _options = rq.FORECAST_WEATHER + rq.HRRR_COLS + rq.WWD_COLS
        _default = ["temperature", "wind_speed", "clouds"]

    stream_picker = mo.ui.multiselect(
        options=_options,
        value=[s for s in _default if s in _options],
        label="Forecast columns" if read_kind.value == "forecasts" else "Columns",
    )

    # Only meaningful for the forecast read: the actual values over the same
    # window, which is what turns forecaster INPUT into TRAINING data.
    target_picker = mo.ui.multiselect(
        options=rq.GEN_STREAMS + ["load"] + rq.OBSERVED_WEATHER,
        value=["wind", "solar"],
        label="Actual columns to attach (training targets)",
    )

    columns_view = (
        stream_picker
        if read_kind.value == "actuals"
        else mo.vstack(
            [
                stream_picker,
                target_picker,
                mo.md(
                    "*No join is needed for the targets: the fusion pipeline "
                    "copies the actual value of a target hour onto every row "
                    "that mentions it, so once the causal rule has picked the "
                    "row, the actual power is already sitting on it.*"
                ),
            ]
        )
    )
    columns_view
    return stream_picker, target_picker


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The window
    """)
    return


@app.cell(hide_code=True)
def _(dt, mo, pd, span_hi, span_lo):
    def _wall(ts):
        """Drop the tz for the widget; the query puts it back."""
        if ts is None:
            return None
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.to_pydatetime()

    _lo = _wall(span_lo)
    _hi = _wall(span_hi)
    _wanted = dt.datetime(2019, 6, 3, 9, 0)
    _default_start = min(max(_wanted, _lo), _hi) if _lo and _hi else _wanted
    _default_end = min(_default_start + dt.timedelta(hours=72), _hi) if _hi else _default_start

    # precision="minute" exposes year, month, day, hour and minute as separate
    # editable fields. The window is not snapped to midnight or to an issuance
    # time; both bounds are free.
    start_picker = mo.ui.datetime(
        value=_default_start, start=_lo, stop=_hi,
        precision="minute", label="Window starts at",
    )
    end_picker = mo.ui.datetime(
        value=_default_end, start=_lo, stop=_hi,
        precision="minute", label="Window ends at",
    )

    mo.vstack(
        [
            mo.hstack([start_picker, end_picker], justify="start", gap=3),
            mo.md(
                "*These two are the whole specification of the window: the start "
                "is also the moment the forecast is being made, and the horizon is "
                "`end - start`. The data is hourly, so minutes other than :00 "
                "simply move which hour the window opens on. The resolved call is "
                "shown below.*"
            ),
        ]
    )
    return end_picker, start_picker


@app.cell(hide_code=True)
def _(catalog, end_picker, mo, pd, read_kind, rq, start_picker):
    win_start = pd.Timestamp(start_picker.value)
    win_end = pd.Timestamp(end_picker.value)

    _tz = catalog.time_tz
    if _tz is not None:
        win_start = win_start.tz_localize(_tz)
        win_end = win_end.tz_localize(_tz)

    _hours = (win_end - win_start) / pd.Timedelta(hours=1)
    _bad = win_end <= win_start

    if read_kind.value == "forecasts":
        _call = (
            "rq.get_forecasts(\n"
            "    catalog, nodes=[...], columns=[...], actuals=[...],\n"
            f"    start='{win_start}',\n"
            f"    end='{win_end}',\n"
            ")"
        )
        _extra = (
            f"\n\nStanding at **{win_start}**, looking {_hours:.0f} h ahead. "
            "Only issuances made at or before that moment are admissible, so "
            "`predict_time` should come back constant and `hour_diff` should "
            "climb steadily."
            + (
                f"\n\n**Longer than the {rq.MAX_HORIZON_HOURS} h rollout.** "
                "No issuance made at or before the start reaches the end of "
                f"this window, so the last "
                f"{_hours - rq.MAX_HORIZON_HOURS:.0f} h cannot come back. "
                "The query returns what exists and raises a warning."
                if _hours > rq.MAX_HORIZON_HOURS
                else ""
            )
        )
    else:
        _call = (
            "rq.get_actuals(\n"
            "    catalog, nodes=[...], streams=[...],\n"
            f"    start='{win_start}',\n"
            f"    end='{win_end}',\n"
            ")"
        )
        _extra = f"\n\n{_hours:.0f} h, {_hours + 1:.0f} hourly points."

    window_view = mo.callout(
        mo.md(f"```python\n{_call}\n```{_extra}"),
        kind="danger" if _bad else "info",
    )
    window_view
    return win_end, win_start


@app.cell(hide_code=True)
def _(mo, read_kind):
    layout_picker = mo.ui.radio(
        options={"long (node_id column)": "long", "wide (one column per node)": "wide"},
        value="long (node_id column)",
        label="Shape",
    )
    check_toggle = mo.ui.checkbox(
        value=False,
        label="Verify that the collapsed duplicates really were identical (slower)",
    )
    run_button = mo.ui.run_button(label="Extract")

    mo.vstack(
        [
            mo.hstack(
                [layout_picker] + ([check_toggle] if read_kind.value == "actuals" else []),
                justify="start",
                gap=3,
            ),
            run_button,
        ]
    )
    return check_toggle, layout_picker, run_button


@app.cell(hide_code=True)
def _(
    catalog,
    check_toggle,
    layout_picker,
    mo,
    node_picker,
    read_kind,
    rq,
    run_button,
    stream_picker,
    target_picker,
    win_end,
    win_start,
):
    mo.stop(
        not run_button.value,
        mo.md("*Set the options above, then press **Extract**.*"),
    )

    import time as _time

    _t0 = _time.perf_counter()
    result = None
    result_error = None
    try:
        if read_kind.value == "actuals":
            result = rq.get_actuals(
                catalog,
                nodes=list(node_picker.value),
                streams=list(stream_picker.value),
                start=win_start,
                end=win_end,
                layout=layout_picker.value,
                check_duplicates=check_toggle.value,
            )
        else:
            result = rq.get_forecasts(
                catalog,
                nodes=list(node_picker.value),
                columns=list(stream_picker.value),
                actuals=list(target_picker.value),
                start=win_start,
                end=win_end,
                layout=layout_picker.value,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
        result_error = str(exc)
    elapsed = _time.perf_counter() - _t0
    return elapsed, result, result_error


@app.cell(hide_code=True)
def _(elapsed, mo, result, result_error, rq):
    mo.stop(
        result_error is not None,
        mo.callout(mo.md(f"Extraction failed:\n\n```\n{result_error}\n```"), kind="danger"),
    )

    notes = result.attrs.get("notes", [])
    time_col = rq.TIME_COL if rq.TIME_COL in result.columns else result.columns[0]

    summary = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="rows", value=f"{len(result):,}"),
                    mo.stat(label="columns", value=f"{result.shape[1]}"),
                    mo.stat(
                        label="nodes",
                        value=f"{result['node_id'].nunique() if 'node_id' in result else '-'}",
                    ),
                    mo.stat(label="seconds", value=f"{elapsed:.2f}"),
                ],
                justify="start",
                gap=2,
            ),
            mo.md(f"**{result[time_col].min()}** to **{result[time_col].max()}**"),
            mo.callout(
                mo.md(
                    "**The window could not be completed.** "
                    + result.attrs["incomplete"]
                ),
                kind="danger",
            )
            if result.attrs.get("incomplete")
            else mo.md(""),
            mo.callout(
                mo.md(
                    "Worth knowing about this result:\n\n"
                    + "\n".join(f"- {n}" for n in notes)
                ),
                kind="warn",
            )
            if notes
            else mo.md(""),
            mo.ui.table(result.head(200), selection=None, page_size=10),
        ]
    )
    summary
    return (time_col,)


@app.cell(hide_code=True)
def _(mo, read_kind):
    mo.md(
        """
        ### Does the window logic hold up

        All of these are cheap, so they are checked rather than assumed. The
        one that matters most is causality — it is the only error here that
        would not announce itself, because a frame carrying forecasts
        published inside the window trains and scores perfectly well and is
        simply wrong.
        """
    ) if read_kind.value == "forecasts" else mo.md("")
    return


@app.cell(hide_code=True)
def _(layout_picker, mo, read_kind, result, rq, win_end, win_start):
    mo.stop(read_kind.value != "forecasts" or layout_picker.value != "long", mo.md(""))

    _checks = rq.validate_forecast_window(result, win_start, win_end)
    _all_pass = bool(_checks["pass"].all())

    validation = mo.vstack(
        [
            mo.callout(
                mo.md(
                    "All checks passed."
                    if _all_pass
                    else "**Something in the window is not what it should be.**"
                ),
                kind="success" if _all_pass else "danger",
            ),
            mo.ui.table(_checks, selection=None),
        ]
    )
    validation
    return


@app.cell(hide_code=True)
def _(layout_picker, mo, pd, read_kind, result, rq):
    mo.stop(read_kind.value != "forecasts" or layout_picker.value != "long", mo.md(""))

    _rows = []
    for _node, _part in result.groupby("node_id", sort=True):
        _offsets = _part[rq.OFFSET_COL].astype("int64")
        _rows.append(
            {
                "node": _node,
                "hours": len(_part),
                "issuances used": _part[rq.ISSUE_COL].nunique(),
                "first issuance": _part[rq.ISSUE_COL].min(),
                "last issuance": _part[rq.ISSUE_COL].max(),
                "smallest offset": int(_offsets.min()),
                "largest offset": int(_offsets.max()),
                "hours on an older rollout": int(
                    (_part[rq.ISSUE_COL] < _part[rq.ISSUE_COL].max()).sum()
                ),
            }
        )

    provenance = mo.vstack(
        [
            mo.md(
                "**Which issuance the window actually came from.** A healthy "
                "window is served end to end by a single issuance made within "
                f"the last {rq.ISSUE_INTERVAL_HOURS} h, so `issuances used` "
                "is 1 and the smallest offset is 0–"
                f"{rq.ISSUE_INTERVAL_HOURS - 1}. A smallest offset of 15 or 24 "
                "means the issuances in between are absent from the file; "
                "hours on an older rollout mean the newest rollout has holes "
                "or stops short of the window. Both are defects to report, "
                "not faults in the selection."
            ),
            mo.ui.table(pd.DataFrame(_rows), selection=None),
        ]
    )
    provenance
    return


@app.cell(hide_code=True)
def _(mo, result, time_col):
    _num = [
        c for c in result.columns
        if c not in ("node_id", time_col) and result[c].dtype.kind == "f"
    ]
    quality = (
        mo.vstack(
            [
                mo.md("**How complete is what came back**"),
                mo.ui.table(
                    result[_num]
                    .isna()
                    .mean()
                    .rename("share missing")
                    .to_frame()
                    .assign(**{"non-null rows": result[_num].notna().sum()})
                    .reset_index()
                    .rename(columns={"index": "column"}),
                    selection=None,
                ),
            ]
        )
        if _num
        else mo.md("")
    )
    quality
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Save it
    """)
    return


@app.cell(hide_code=True)
def _(mo, node_picker, read_kind, stream_picker, win_end, win_start):
    _suggested = (
        f"{read_kind.value}_"
        f"{'-'.join(list(node_picker.value)[:3])}"
        f"{'_etc' if len(node_picker.value) > 3 else ''}_"
        f"{'-'.join(list(stream_picker.value)[:3])}_"
        f"{win_start:%Y%m%dT%H%M}_{win_end:%Y%m%dT%H%M}"
    )
    name_input = mo.ui.text(value=_suggested, label="File name", full_width=True)
    fmt_picker = mo.ui.radio(
        options={"parquet (keeps dtypes, ~10x smaller)": "parquet",
                 "csv (opens in Excel)": "csv"},
        value="parquet (keeps dtypes, ~10x smaller)",
        label="Format",
    )
    subdir_input = mo.ui.text(value="exports", label="Subfolder of REGORW")
    save_button = mo.ui.run_button(label="Save file")

    mo.vstack(
        [
            name_input,
            mo.hstack([fmt_picker, subdir_input], justify="start", gap=3),
            save_button,
        ]
    )
    return fmt_picker, name_input, save_button, subdir_input


@app.cell(hide_code=True)
def _(
    catalog,
    fmt_picker,
    mo,
    name_input,
    result,
    rq,
    save_button,
    subdir_input,
):
    mo.stop(not save_button.value, mo.md("*Press **Save file** to write it out.*"))

    out_dir = catalog.root / subdir_input.value.strip("/\\ ")
    try:
        written = rq.export(result, out_dir, name_input.value, fmt=fmt_picker.value)
        saved_view = mo.callout(
            mo.md(
                f"Written to `{written}`\n\n"
                f"{written.stat().st_size/1e6:.2f} MB, "
                f"{len(result):,} rows x {result.shape[1]} columns"
            ),
            kind="success",
        )
    except Exception as exc:  # noqa: BLE001
        written = None
        saved_view = mo.callout(
            mo.md(f"Could not write the file:\n\n```\n{exc}\n```"), kind="danger"
        )
    saved_view
    return


if __name__ == "__main__":
    app.run()
