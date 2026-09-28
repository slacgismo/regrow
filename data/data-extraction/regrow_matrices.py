import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="REGROW training matrices")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # REGROW training matrices

        Builds the **feature matrix X** and the **target matrix Y** a forecaster
        is fitted on. No fitting and no train/test split happen here \u2014 this
        page only constructs the two arrays, correctly and causally. The logic
        lives in `regrow_features.py`, which reads through `regrow_query.py`.

        ## One sample = one decision moment

        Standing at an origin time **T0**, the forecaster predicts the next
        **H** hours in one go, as a vector in R\u1d34 \u2014 never a one-step model
        iterated forward. So Y is a **matrix**:

        `Y[i, h\u22121]` = actual value of the target at `T0\u1d62 + h`, for h = 1 \u2026 H

        and each row of X holds only what was genuinely available at T0.

        | block | contents | direction | default |
        |---|---|---|---|
        | **forecast** | forecast weather for T0+1 \u2026 T0+H, from issuances made at or before T0 | forward | on |
        | **autoregressive** | the actual target streams at T0, T0\u22121, \u2026 | backward | off |
        | **observed** | actual weather at T0, T0\u22121, \u2026 | backward | off |

        Forecast alone is the **pure exogenous** model. Adding the
        autoregressive block gives **exogenous + autoregressive**. The blocks
        are kept separate so the value of each can be tested by leaving it out.

        Several streams at one node **share X** and get **one Y each**. Across
        nodes the result is a **tensor** with a node \u2192 index map.

        Samples are never dropped: an origin missing any value keeps its row,
        with NaN where the value does not exist, and is marked False in the
        **valid** mask. Fit on `X[valid]`, `Y[valid]`.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    import datetime as dt

    import numpy as np
    import pandas as pd

    import regrow_features as rf
    import regrow_query as rq

    return dt, np, pd, rf, rq


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
        _err = None
    except Exception as exc:  # noqa: BLE001 - shown to the user
        catalog, _err = None, str(exc)
    mo.stop(
        catalog is None,
        mo.callout(mo.md(f"Could not build the catalog.\n\n```\n{_err}\n```"), kind="danger"),
    )
    span_lo, span_hi = catalog.span()
    mo.callout(
        mo.md(f"**{len(catalog.nodes)} nodes**, {span_lo:%Y-%m-%d} to {span_hi:%Y-%m-%d}"),
        kind="success",
    )
    return catalog, span_hi, span_lo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. What to predict, and from which nodes
    """)
    return


@app.cell(hide_code=True)
def _(mo, rq):
    target_picker = mo.ui.multiselect(
        options=rq.POWER_STREAMS,
        value=["wind"],
        label="Targets (one Y each, all sharing X)",
    )
    target_picker
    return (target_picker,)


@app.cell(hide_code=True)
def _(catalog, mo, target_picker):
    _wanted = list(target_picker.value)
    _carrying = [
        n for n, i in catalog.nodes.items()
        if all(t in i.streams for t in _wanted)
    ] if _wanted else []

    node_picker = mo.ui.multiselect(
        options=list(catalog.nodes),
        value=_carrying[:2],
        label=f"Nodes ({len(_carrying)} carry every selected target)",
    )
    mo.vstack([
        node_picker,
        mo.md(
            "*One node gives a matrix; several give a tensor. A node missing a "
            "target or a column is left out of the tensor and the reason is "
            "listed below the result.*"
        ),
    ])
    return (node_picker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. The samples

    An **origin**, written **T0**, is one decision moment: the instant at
    which the forecaster is asked *"what happens over the next H hours?"*.
    Each origin becomes **one row of X and one row of Y**, so the number of
    origins is the number of samples.

    The two pickers set the first and last T0; the spacing sets how often in
    between. Hourly over January 2019 gives 31 × 24 = **744 samples**.

    Spacing changes how many samples you get, not what a sample is. Hourly
    is how often a receding-horizon controller re-plans, and six consecutive
    origins then share one issuance. Six-hourly gives one sample per
    issuance instead, so every sample rests on a forecast published moments
    earlier.
    """)
    return


@app.cell(hide_code=True)
def _(dt, mo, pd, span_hi, span_lo):
    def _wall(ts):
        ts = pd.Timestamp(ts)
        return (ts.tz_localize(None) if ts.tzinfo else ts).to_pydatetime()

    _lo, _hi = _wall(span_lo), _wall(span_hi)
    origin_start = mo.ui.datetime(
        value=max(_lo, dt.datetime(2019, 1, 1, 0, 0)), start=_lo, stop=_hi,
        precision="hour", label="First origin (T0)",
    )
    origin_end = mo.ui.datetime(
        value=max(_lo, dt.datetime(2019, 1, 31, 23, 0)), start=_lo, stop=_hi,
        precision="hour", label="Last origin (T0)",
    )
    every_picker = mo.ui.radio(
        options={"every hour (how often a controller re-plans)": "1h",
                 "every 6 h (one per issuance cycle)": "6h"},
        value="every hour (how often a controller re-plans)",
        label="Spacing between origins",
    )
    horizon_input = mo.ui.number(start=1, stop=120, step=1, value=48,
                                 label="Horizon H (hours; Y has H columns)")
    dtype_picker = mo.ui.radio(
        options={"float64 (exact)": "float64", "float32 (half the memory)": "float32"},
        value="float64 (exact)",
        label="Storage type",
    )
    mo.vstack([
        mo.hstack([origin_start, origin_end], justify="start", gap=3),
        mo.hstack([every_picker, horizon_input, dtype_picker], justify="start", gap=3),
    ])
    return dtype_picker, every_picker, horizon_input, origin_end, origin_start


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. The features
    """)
    return


@app.cell(hide_code=True)
def _(mo, rq):
    fc_picker = mo.ui.multiselect(
        options=rq.FORECAST_WEATHER + rq.HRRR_COLS + rq.WWD_COLS,
        value=["temperature", "wind_speed", "wind_deg", "clouds"],
        label="Forecast block \u2014 columns (each contributes H features)",
    )
    ar_toggle = mo.ui.checkbox(value=False, label="Autoregressive block")
    ar_lags_input = mo.ui.text(value="0, -1, -2, -24", label="AR lags (\u2264 0; 0 = the origin)")
    obs_toggle = mo.ui.checkbox(value=False, label="Observed-weather block")
    obs_picker = mo.ui.multiselect(
        options=rq.OBSERVED_WEATHER,
        value=["a_temperature", "a_wind_speed"],
        label="Observed columns",
    )
    obs_lags_input = mo.ui.text(value="0, -1", label="Observed lags (\u2264 0)")

    mo.vstack([
        fc_picker,
        mo.hstack([ar_toggle, ar_lags_input], justify="start", gap=3),
        mo.hstack([obs_toggle, obs_picker, obs_lags_input], justify="start", gap=3),
        mo.md(
            "*With only the forecast block this is the pure exogenous model. "
            "When the autoregressive block is on, X carries the lags of every "
            "target, so X stays shared across them. A lag of 0 is the value at "
            "the origin itself; positive lags are refused, since they would put "
            "the answer into the question.*"
        ),
    ])
    return (
        ar_lags_input,
        ar_toggle,
        fc_picker,
        obs_lags_input,
        obs_picker,
        obs_toggle,
    )


@app.cell(hide_code=True)
def _(
    ar_lags_input,
    ar_toggle,
    dtype_picker,
    every_picker,
    fc_picker,
    horizon_input,
    mo,
    node_picker,
    obs_lags_input,
    obs_picker,
    obs_toggle,
    origin_end,
    origin_start,
    pd,
    rf,
    target_picker,
):
    def _lags(text):
        return [int(x) for x in text.replace(" ", "").split(",") if x != ""]

    def _lag_count(text, on):
        return len(set(_lags(text))) if on else 0

    _n_origins = len(pd.date_range(pd.Timestamp(origin_start.value),
                                   pd.Timestamp(origin_end.value),
                                   freq=every_picker.value))
    _H = int(horizon_input.value)
    _n_feat = rf.n_features_for(
        list(fc_picker.value), _H,
        list(target_picker.value) if ar_toggle.value else [],
        range(_lag_count(ar_lags_input.value, ar_toggle.value)),
        list(obs_picker.value) if obs_toggle.value else [],
        range(_lag_count(obs_lags_input.value, obs_toggle.value)),
    )
    _bytes = rf.estimate_bytes(max(len(node_picker.value), 1), _n_origins,
                               _n_feat, _H, max(len(target_picker.value), 1),
                               dtype_picker.value)
    _big = _bytes > 2e9

    _nodes = list(node_picker.value)
    _fn = "rf.build_tensor" if len(_nodes) > 1 else "rf.build_node_matrices"
    _who = repr(_nodes) if len(_nodes) > 1 else (repr(_nodes[0]) if _nodes else "[]")
    _call = (
        f"{_fn}(\n"
        f"    catalog, {_who}, targets={list(target_picker.value)!r},\n"
        f"    origins=rf.make_origins('{pd.Timestamp(origin_start.value)}', "
        f"'{pd.Timestamp(origin_end.value)}', every='{every_picker.value}'),\n"
        f"    horizon={_H},\n"
        f"    forecast_columns={list(fc_picker.value)!r},\n"
        + (f"    ar_lags={_lags(ar_lags_input.value)!r},\n" if ar_toggle.value else "")
        + (f"    observed_columns={list(obs_picker.value)!r},\n"
           f"    observed_lags={_lags(obs_lags_input.value)!r},\n"
           if obs_toggle.value else "")
        + f"    dtype='{dtype_picker.value}',\n)"
    )

    size_preview = mo.callout(
        mo.md(
            f"```python\n{_call}\n```\n\n"
            f"**{len(_nodes)} node(s) × {_n_origins:,} origins × "
            f"{_n_feat:,} features**, horizon {_H}. X and Y would occupy about "
            f"**{_bytes / 1e9:.2f} GB** in {dtype_picker.value}."
            + ("\n\nThat is large enough to be worth reconsidering: shorten the "
               "origin range, drop forecast columns, use 6 h spacing, or switch "
               "to float32." if _big else "")
        ),
        kind="warn" if _big else "info",
    )
    size_preview
    return


@app.cell(hide_code=True)
def _(mo):
    build_button = mo.ui.run_button(label="Build X and Y")
    build_button
    return (build_button,)


@app.cell(hide_code=True)
def _(
    ar_lags_input,
    ar_toggle,
    build_button,
    catalog,
    dtype_picker,
    every_picker,
    fc_picker,
    horizon_input,
    mo,
    node_picker,
    obs_lags_input,
    obs_picker,
    obs_toggle,
    origin_end,
    origin_start,
    rf,
    target_picker,
):
    mo.stop(not build_button.value, mo.md("*Set the options above, then press **Build X and Y**.*"))

    import time as _time

    def _lags(text):
        return [int(x) for x in text.replace(" ", "").split(",") if x != ""]

    built, skipped, build_error = [], {}, None
    _t0 = _time.perf_counter()
    try:
        origins = rf.make_origins(origin_start.value, origin_end.value,
                                  every=every_picker.value, tz=catalog.time_tz)
        _kwargs = dict(
            horizon=int(horizon_input.value),
            forecast_columns=list(fc_picker.value),
            ar_lags=_lags(ar_lags_input.value) if ar_toggle.value else (),
            observed_columns=list(obs_picker.value) if obs_toggle.value else None,
            observed_lags=_lags(obs_lags_input.value),
            dtype=dtype_picker.value,
        )
        for _node in node_picker.value:
            try:
                built.append(rf.build_node_matrices(
                    catalog, _node, list(target_picker.value), origins, **_kwargs))
            except ValueError as exc:
                skipped[_node] = str(exc)
        tensor = rf.stack_nodes(built, skipped) if len(built) > 1 else None
    except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
        build_error, tensor = str(exc), None
    elapsed = _time.perf_counter() - _t0
    by_node = {m.node: m for m in built}
    return build_error, built, by_node, elapsed, skipped, tensor


@app.cell(hide_code=True)
def _(build_error, built, elapsed, mo, np, pd, skipped, tensor):
    mo.stop(build_error is not None,
            mo.callout(mo.md(f"Build failed:\n\n```\n{build_error}\n```"), kind="danger"))
    mo.stop(not built, mo.callout(
        mo.md("No node carried every target and column.\n\n"
              + "\n".join(f"- {k}: {v}" for k, v in skipped.items())),
        kind="danger"))

    _m = built[0]
    _x_shape = tensor.X.shape if tensor is not None else _m.X.shape
    _y_shapes = ({t: tensor.Y[t].shape for t in tensor.Y} if tensor is not None
                 else {t: y.shape for t, y in _m.Y.items()})
    _valid = tensor.valid if tensor is not None else _m.valid
    _nbytes = (
        (tensor.X.nbytes + sum(y.nbytes for y in tensor.Y.values()))
        if tensor is not None
        else (_m.X.nbytes + sum(y.nbytes for y in _m.Y.values()))
    )

    _dims = rf.describe_dimensions(tensor if tensor is not None else _m)
    _layout = _m.feature_layout()
    _notes = [n for m in built for n in m.notes]

    mo.vstack([
        mo.hstack([
            mo.stat(label="X shape", value=" \u00d7 ".join(map(str, _x_shape))),
            mo.stat(label="Y shape (each)", value=" \u00d7 ".join(map(str, next(iter(_y_shapes.values()))))),
            mo.stat(label="valid samples", value=f"{int(np.sum(_valid)):,} / {_valid.size:,}"),
            mo.stat(label="seconds", value=f"{elapsed:.1f}"),
            mo.stat(label="memory", value=f"{_nbytes / 1e6:,.0f} MB"),
        ], justify="start", gap=2),
        mo.md(
            "**What each axis means.** A shape on its own says nothing, so "
            "this spells out which node is which, which hours the origins "
            "are, and which columns of X hold which block."
            + (f"\n\nTargets: {', '.join(f'`Y[\"{t}\"]`' for t in _y_shapes)}"
               if len(_y_shapes) > 1 else "")
        ),
        mo.ui.table(_dims, selection=None, page_size=8),
        mo.md(
            "**Which columns of X hold which variable.** Every forecast "
            f"variable contributes {_m.horizon} features — one per step — "
            "while every lag contributes exactly one, because a past hour "
            "has only one true value."
        ),
        mo.ui.table(_layout, selection=None, page_size=10),
        mo.callout(mo.md("Left out of the tensor:\n\n"
                         + "\n".join(f"- {k}: {v}" for k, v in skipped.items())),
                   kind="warn") if skipped else mo.md(""),
        mo.callout(mo.md("Worth knowing:\n\n" + "\n".join(f"- {n}" for n in _notes)),
                   kind="warn") if _notes else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Look at one sample

    One row of X and the matching row of Y, shown **separately**, because
    the distinction that must never blur is which side of the origin a
    number came from.

    - **X** is everything the model may look at: the forecast block, one row
      per step, plus the backward-looking lags.
    - **Y** is what actually happened over the same 48 hours — the answer,
      which never appears in X.

    In the forecast table, **`target time`** is the hour being predicted,
    **`forecast issued`** is when that prediction was published (always at
    or before the origin), and **`lead (h)`** is the gap between them, which
    stays between 0 and 5 because issuances arrive every six hours.
    """)
    return


@app.cell(hide_code=True)
def _(built, mo):
    mo.stop(not built, mo.md(""))
    inspect_node = mo.ui.dropdown(options=[m.node for m in built],
                                  value=built[0].node, label="Node")
    _first_valid = int(built[0].valid.argmax()) if built[0].valid.any() else 0
    sample_slider = mo.ui.slider(start=0, stop=len(built[0].origins) - 1,
                                 value=_first_valid, label="Origin index", show_value=True)
    round_toggle = mo.ui.checkbox(value=True, label="Round for display")
    mo.hstack([inspect_node, sample_slider, round_toggle], justify="start", gap=3)
    return inspect_node, round_toggle, sample_slider


@app.cell(hide_code=True)
def _(by_node, inspect_node, mo, np, round_toggle, sample_slider):
    _m = by_node[inspect_node.value]
    _i = int(sample_slider.value)
    _t0 = _m.origins[_i]
    _ok = bool(_m.valid[_i])

    _step = _m.forecast_frame(_i)
    _target = _m.target_frame(_i)
    _back = _m.lag_frame(_i)
    if round_toggle.value:
        # Only the numeric columns: rounding a frame that holds timestamps
        # is a no-op there and pandas warns about it.
        def _r(df):
            return df.round(dict.fromkeys(df.select_dtypes("number").columns, 2))

        _step, _target, _back = _r(_step), _r(_target), _r(_back)

    _ages = _m.age[_i]
    _issued = np.nanmax(_ages) if not np.isnan(_ages).all() else float("nan")
    _window = (
        f"predicted window **{_m.step_times(_i)[0]}** \u2192 "
        f"**{_m.step_times(_i)[-1]}**"
    )
    _lead = (
        "" if np.isnan(_issued)
        else f" \u00b7 forecast published {int(_issued)} h before the origin"
    )

    _nx = _m.X.shape[1]
    _fc_n = _m.blocks["forecast"].stop - _m.blocks["forecast"].start

    mo.vstack([
        mo.callout(
            mo.md(
                f"**Origin {_t0}** \u00b7 node `{_m.node}` \u00b7 "
                f"valid: **{_ok}**\n\n{_window}{_lead}"
            ),
            kind="success" if _ok else "warn",
        ),
        mo.md(
            f"### X \u2014 what the model may look at ({_nx} numbers)\n\n"
            f"**Forecast block** \u2014 {len(_m.config['forecast_columns'])} "
            f"variables \u00d7 {_m.horizon} steps = **{_fc_n}** of those numbers, "
            "published before the origin but about hours after it."
        ),
        mo.ui.table(_step, selection=None, page_size=12),
        mo.md(
            f"**Backward-looking block** \u2014 the remaining **{len(_back)}**. "
            "Every one sits at or before the origin, which is what makes it "
            "usable at all. One number per lag, because a past hour has only "
            "one true value."
        ) if len(_back) else mo.md(""),
        mo.ui.table(_back, selection=None, page_size=10) if len(_back) else mo.md(""),
        mo.md(
            f"### Y \u2014 what it has to predict ({_m.horizon} numbers per target)"
            "\n\nWhat actually happened over the same window. None of this was "
            "available at the origin, and none of it appears in X."
        ),
        mo.ui.table(_target, selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Check it against the reference

    The forecast block is computed by one vectorised as-of join for speed,
    not by calling `get_forecasts` once per origin. This recomputes a
    random sample of rows the slow way and compares them cell by cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    check_button = mo.ui.run_button(label="Run the check")
    check_button
    return (check_button,)


@app.cell(hide_code=True)
def _(built, by_node, catalog, check_button, inspect_node, mo, rf):
    mo.stop(not built or not check_button.value, mo.md(""))
    _r = rf.check_against_reference(by_node[inspect_node.value], catalog, n_samples=5)
    _ok = bool(_r["match"].all()) if len(_r) else False
    mo.vstack([
        mo.callout(mo.md("Every sampled cell matches the reference implementation."
                         if _ok else "**Some cells disagree with the reference.**"),
                   kind="success" if _ok else "danger"),
        mo.ui.table(_r, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Save it
    """)
    return


@app.cell(hide_code=True)
def _(built, mo, tensor):
    mo.stop(not built, mo.md(""))
    _stem = ("tensor_" + "-".join(tensor.nodes[:3]) if tensor is not None
             else "matrices_" + built[0].node)
    _stem += "_" + "-".join(built[0].targets)
    save_name = mo.ui.text(value=_stem, label="File name (.npz)", full_width=True)
    save_dir = mo.ui.text(value="training", label="Subfolder of REGORW")
    save_button = mo.ui.run_button(label="Save")
    mo.vstack([save_name, save_dir, save_button])
    return save_button, save_dir, save_name


@app.cell(hide_code=True)
def _(built, catalog, mo, save_button, save_dir, save_name, tensor):
    mo.stop(not built, mo.md(""))
    mo.stop(not save_button.value, mo.md("*Press **Save** to write the arrays out.*"))
    _obj = tensor if tensor is not None else built[0]
    _path = _obj.save(catalog.root / save_dir.value.strip("/\\ ") / save_name.value)
    mo.callout(
        mo.md(
            f"Written to `{_path}` ({_path.stat().st_size / 1e6:.2f} MB)\n\n"
            "Read it back with `rf.load_matrices(path)`."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
