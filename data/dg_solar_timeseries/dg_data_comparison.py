# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.1",
#     "marimo>=0.23.9",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
# ]
# ///

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd

    # Optional plotting backends
    import altair as alt
    import matplotlib.pyplot as plt

    return alt, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Timeseries diff explorer (two PV CSVs)

    Goal: investigate subtle differences between two large multivariate time series CSVs.
    - Index: timestamp (hourly or sub-hourly)
    - Columns: geohash locations
    - Values: PV generation

    This notebook loads both files, aligns them, computes differences, and provides interactive views:
    - Column-level summary (max abs diff, RMSE, correlation, etc.)
    - Time-window inspection for a selected geohash
    - Outlier/threshold search and event table
    """)
    return


@app.cell
def _(mo):
    # File inputs (paths or uploaded contents)
    original_path = mo.ui.text(
        value="original.csv",
        label="Original CSV path",
        full_width=True,
    )
    new_path = mo.ui.text(
        value="new.csv",
        label="New CSV path",
        full_width=True,
    )

    # Parsing controls
    timestamp_col = mo.ui.text(
        value="timestamp",
        label="Timestamp column name (leave as-is if your CSV already uses first column as index)",
        full_width=True,
    )
    index_in_first_col = mo.ui.checkbox(
        value=True,
        label="Timestamp is in first column (read as index)",
    )
    tz_mode = mo.ui.dropdown(
        options=["keep", "convert_to_utc", "drop_timezone"],
        value="keep",
        label="Timezone handling",
    )
    freq_hint = mo.ui.dropdown(
        options=["infer", "30min", "1h"],
        value="infer",
        label="Expected frequency (for diagnostics)",
    )

    load_btn = mo.ui.button(label="Load / Reload")

    ui_frame = mo.md(
        """
    ## Inputs
    {original_path}
    {new_path}

    ### Parsing
    {timestamp_col}
    {index_in_first_col}
    {tz_mode}
    {freq_hint}

    {load_btn}
    """
    ).batch(
        original_path=original_path,
        new_path=new_path,
        timestamp_col=timestamp_col,
        index_in_first_col=index_in_first_col,
        tz_mode=tz_mode,
        freq_hint=freq_hint,
        load_btn=load_btn,
    )
    ui_frame
    return (ui_frame,)


@app.cell
def _(pd, ui_frame):
    def _read_csv(path: str) -> pd.DataFrame:
        if ui_frame.value["index_in_first_col"]:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
        else:
            df = pd.read_csv(path)
            df[ui_frame.value["timestamp_col"]] = pd.to_datetime(df[ui_frame.value["timestamp_col"]], errors="coerce")
            df = df.set_index(ui_frame.value["timestamp_col"])

        # Ensure numeric where possible
        df = df.apply(pd.to_numeric, errors="coerce")

        # Timezone handling
        # - If timestamps are tz-aware, pandas keeps tz info in DatetimeIndex
        # - If they are strings with offset (e.g. ...-07:00), to_datetime will often produce tz-aware.
        if isinstance(df.index, pd.DatetimeIndex):
            if ui_frame.value["tz_mode"] == "convert_to_utc":
                if df.index.tz is None:
                    # leave naive as-is
                    pass
                else:
                    df.index = df.index.tz_convert("UTC")
            elif ui_frame.value["tz_mode"] == "drop_timezone":
                if df.index.tz is None:
                    pass
                else:
                    df.index = df.index.tz_convert("UTC").tz_localize(None)
            else:
                # keep
                pass

        # Drop rows with unparsed timestamps
        df = df.loc[~df.index.isna()].sort_index()
        return df

    # Load when button pressed or first run
    _ = ui_frame.value["load_btn"] # reactive dependency

    original_df = _read_csv(ui_frame.value["original_path"])
    new_df = _read_csv(ui_frame.value["new_path"])
    return new_df, original_df


@app.cell
def _(mo, new_df, original_df, pd, ui_frame):
    # Basic diagnostics
    def _freq(df: pd.DataFrame) -> str:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 3:
            return "n/a"
        try:
            if ui_frame.value["freq_hint"] == "infer":
                f = pd.infer_freq(df.index[:5000])
                return f or "unknown"
            return ui_frame.value["req_hint"]
        except Exception:
            return "unknown"

    info = {
        "original_shape": original_df.shape,
        "new_shape": new_df.shape,
        "original_start": str(original_df.index.min()) if len(original_df) else "n/a",
        "original_end": str(original_df.index.max()) if len(original_df) else "n/a",
        "new_start": str(new_df.index.min()) if len(new_df) else "n/a",
        "new_end": str(new_df.index.max()) if len(new_df) else "n/a",
        "original_inferred_freq": _freq(original_df),
        "new_inferred_freq": _freq(new_df),
        "original_columns": len(original_df.columns),
        "new_columns": len(new_df.columns),
        "columns_overlap": len(set(original_df.columns).intersection(set(new_df.columns))),
    }

    mo.md(
        f"""
    ## Loaded
    - Original: **{info["original_shape"]}**, columns={info["original_columns"]}, time=[{info["original_start"]} … {info["original_end"]}], freq={info["original_inferred_freq"]}
    - New: **{info["new_shape"]}**, columns={info["new_columns"]}, time=[{info["new_start"]} … {info["new_end"]}], freq={info["new_inferred_freq"]}
    - Column overlap: **{info["columns_overlap"]}**
    """
    )
    return


@app.cell
def _(new_df, np, original_df):
    # Align both datasets on shared index and columns
    common_cols = sorted(set(original_df.columns).intersection(set(new_df.columns)))
    common_idx = original_df.index.intersection(new_df.index)

    original = original_df.loc[common_idx, common_cols]
    new = new_df.loc[common_idx, common_cols]

    diff = new - original
    absdiff = diff.abs()

    # Convenience masks
    nonfinite_mask = (~np.isfinite(original.to_numpy())) | (~np.isfinite(new.to_numpy()))
    return absdiff, common_cols, common_idx, diff, new, original


@app.cell
def _(common_cols, common_idx, mo, new_df, original_df):
    missing_in_new = sorted(set(original_df.columns) - set(new_df.columns))
    missing_in_orig = sorted(set(new_df.columns) - set(original_df.columns))

    mo.md(
        f"""
    ## Alignment
    - Common columns: **{len(common_cols)}**
    - Common timestamps: **{len(common_idx)}**

    ### Column mismatches
    - Present only in original: **{len(missing_in_new)}**
    - Present only in new: **{len(missing_in_orig)}**
    """
    )
    return


@app.cell
def _(absdiff, diff, new, np, original, pd):
    # Per-column summary stats
    eps = 1e-12

    # Using pandas reductions (NaNs ignored)
    summary = pd.DataFrame(
        {
            "count": diff.count(),
            "mean_diff": diff.mean(),
            "max_abs_diff": absdiff.max(),
            "rmse": np.sqrt((diff**2).mean()),
            "mae": absdiff.mean(),
            "p99_abs_diff": absdiff.quantile(0.99),
            "orig_mean": original.mean(),
            "new_mean": new.mean(),
            "orig_max": original.max(),
            "new_max": new.max(),
        }
    )

    # Correlation (guard against constant series)
    corrs = []
    for _c in diff.columns:
        x = original[_c]
        y = new[_c]
        if x.std(skipna=True) < eps or y.std(skipna=True) < eps:
            corrs.append(np.nan)
        else:
            corrs.append(x.corr(y))
    summary["corr"] = corrs

    summary = summary.sort_values(["max_abs_diff", "rmse"], ascending=False)
    return (summary,)


@app.cell
def _(mo, summary):
    mo.md("## Column summary (sorted by max_abs_diff, rmse)")
    table = mo.ui.dataframe(summary.reset_index().rename(columns={"index": "geohash"}), page_size=20)
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(dd_ui):
    dd_ui.value["col"]
    return


@app.cell
def _(common_cols, mo, summary):
    # Column picker (default to worst offender)
    default_col = summary.index[0] if len(summary) else (common_cols[0] if len(common_cols) else "")
    col = mo.ui.dropdown(options=list(summary.index), value=default_col, label="Geohash column")

    # Thresholding
    abs_threshold = mo.ui.number(
        value=float(summary.loc[default_col, "p99_abs_diff"]) if default_col in summary.index else 1.0,
        label="Abs diff threshold (for events table)",
        start=0.0,
        step=1.0,
    )

    # Time window controls
    # Use date widgets if desired; text keeps it flexible with tz offsets.
    start_ts = mo.ui.text(value="", label="Start timestamp filter (optional)", full_width=True)
    end_ts = mo.ui.text(value="", label="End timestamp filter (optional)", full_width=True)

    dd_ui = mo.md(
        """
    ## Drill-down
    {col}

    ### Time window (optional)
    {start_ts}
    {end_ts}

    ### Event threshold
    {abs_threshold}
    """
    ).batch(col=col, start_ts=start_ts, end_ts=end_ts, abs_threshold=abs_threshold)
    dd_ui
    return abs_threshold, dd_ui


@app.cell
def _(capacity_df, dd_ui, new, original, plt):
    _o = original.groupby(original.index.to_period('M')).sum()
    _n = new.groupby(original.index.to_period('M')).sum()
    _xs = _o.index.to_timestamp()[:-1]
    _ys = _o[dd_ui.value["col"]].values[:-1] / 2 / 1000
    plt.plot(_xs, _ys, label='original')
    _xs = _n.index.to_timestamp()[:-1]
    _ys = _n[dd_ui.value["col"]].values[:-1] / 2 / 1000
    plt.plot(_xs, _ys, label='rescaled')
    _xs = _n.index.to_timestamp()[:-1]
    _ys = capacity_df[capacity_df['geohash'] == dd_ui.value["col"]]['Generation [MWh]'].values[:60]
    plt.plot(_xs, _ys, label='reported', ls='--')
    plt.legend()
    plt.title('month DG solar energy production true-up, ' + dd_ui.value["col"])
    plt.gca()
    return


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(original):
    original
    return


@app.cell
def _(new):
    new
    return


@app.cell
def _(absdiff, dd_ui, diff, new, original, pd):
    c = dd_ui.value["col"]

    s_orig = original[c].copy()
    s_new = new[c].copy()
    s_diff = diff[c].copy()
    s_abs = absdiff[c].copy()

    def _coerce_to_index_tz(ts: str, idx: pd.DatetimeIndex):
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t) or not isinstance(idx, pd.DatetimeIndex):
            return t
        if idx.tz is None:
            # index is tz-naive -> ensure t is tz-naive
            return t.tz_convert(None) if getattr(t, "tzinfo", None) is not None else t
        else:
            # index is tz-aware -> ensure t is tz-aware in same tz as index
            if getattr(t, "tzinfo", None) is None:
                return t.tz_localize(idx.tz)
            return t.tz_convert(idx.tz)

    # Optional time filtering
    if dd_ui.value["start_ts"].strip():
        t0 = _coerce_to_index_tz(pd.to_datetime(dd_ui.value["start_ts"], errors="coerce"), s_orig.index)
        if pd.notna(t0):
            s_orig = s_orig.loc[s_orig.index >= t0]
            s_new = s_new.loc[s_new.index >= t0]
            s_diff = s_diff.loc[s_diff.index >= t0]
            s_abs = s_abs.loc[s_abs.index >= t0]

    if dd_ui.value["end_ts"].strip():
        t1 = _coerce_to_index_tz(pd.to_datetime(dd_ui.value["end_ts"], errors="coerce"), s_orig.index)
        if pd.notna(t1):
            s_orig = s_orig.loc[s_orig.index <= t1]
            s_new = s_new.loc[s_new.index <= t1]
            s_diff = s_diff.loc[s_diff.index <= t1]
            s_abs = s_abs.loc[s_abs.index <= t1]

    view = pd.DataFrame(
        {
            "original": s_orig,
            "new": s_new,
            "diff_new_minus_original": s_diff,
            "abs_diff": s_abs,
        }
    )
    return c, view


@app.cell
def _(c, mo, view):
    mo.md(f"### Row-level view: `{c}`")
    df_view = mo.ui.dataframe(view.reset_index().rename(columns={"index": "timestamp"}), page_size=25)
    return (df_view,)


@app.cell
def _(df_view):
    df_view
    return


@app.cell
def _(new):
    new
    return


@app.cell
def _(dd_ui, diff):
    diff[dd_ui.value["col"]].plot()
    return


@app.cell
def _(dd_ui, original, plt):
    original[dd_ui.value["col"]].plot()
    plt.title('original data, ' + dd_ui.value["col"])
    return


@app.cell
def _(dd_ui, new, plt):
    new[dd_ui.value["col"]].plot()
    plt.title('new data, '+dd_ui.value["col"])
    return


@app.cell
def _(original):
    original.index.to_period('M')
    return


@app.cell
def _(dd_ui, new, original):
    new.groupby(original.index.to_period('M')).sum()[dd_ui.value["col"]] / 2 / 1000
    return


@app.cell
def _(capacity_df, dd_ui):
    capacity_df[capacity_df['geohash'] == dd_ui.value["col"]]
    return


@app.cell
def _(dd_ui, original):
    original[dd_ui.value["col"]]
    return


@app.cell
def _(capacity_df, dd_ui, plt):
    capacity_df[capacity_df['geohash'] == dd_ui.value["col"]].plot(y=['Capacity [MW]', 'Generation [MWh]'], secondary_y='Generation [MWh]')
    plt.title(dd_ui.value["col"]+' monthly capacity and generation values from EIA')
    return


@app.cell
def _(capacity_df, dd_ui):
    capacity_df[capacity_df['geohash'] == dd_ui.value["col"]]
    return


@app.cell
def _(pd):
    capacity_df = pd.read_csv("wecc_bus_dg_cap_and_gen_by_month.csv", index_col=0)
    capacity_df
    return (capacity_df,)


@app.cell
def _(alt, c, view):
    # Plot long-form
    _v = view.reset_index().rename(columns={"index": "timestamp"})
    long = _v.melt(id_vars=["timestamp"], value_vars=["original", "new"], var_name="series", value_name="value")

    lines = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("value:Q", title="PV"),
            color=alt.Color("series:N", title=""),
            tooltip=["timestamp:T", "series:N", "value:Q"],
        )
        .properties(title=f"{c}: original vs new", height=250, width=700)
    )

    diff_line = (
        alt.Chart(_v)
        .mark_line(color="#d62728")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("diff_new_minus_original:Q", title="new - original"),
            tooltip=["timestamp:T", "diff_new_minus_original:Q", "abs_diff:Q"],
        )
        .properties(title=f"{c}: difference", height=180, width=700)
    )

    abs_line = (
        alt.Chart(_v)
        .mark_line(color="#9467bd")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("abs_diff:Q", title="|new - original|"),
            tooltip=["timestamp:T", "abs_diff:Q"],
        )
        .properties(title=f"{c}: absolute difference", height=180, width=700)
    )

    chart = alt.vconcat(lines, diff_line, abs_line).resolve_scale(y="independent")
    return (chart,)


@app.cell
def _(abs_threshold, np, view):
    thr = float(abs_threshold.value)
    events = view[view["abs_diff"] >= thr].copy()
    events = events.sort_values("abs_diff", ascending=False)

    # Add a simple "relative" diff where original is nonzero
    denom = events["original"].abs()
    events["rel_diff"] = (events["diff_new_minus_original"] / denom.where(denom > 0)).replace([np.inf, -np.inf], np.nan)

    # Keep top N for display
    top_events = events.head(500).reset_index().rename(columns={"index": "timestamp"})
    return thr, top_events


@app.cell
def _(c, mo, thr, top_events):
    mo.md(f"### Events where |diff| ≥ {thr:g} for `{c}` (top 500 by |diff|)")
    events_table = mo.ui.dataframe(top_events, page_size=25)
    return (events_table,)


@app.cell
def _(events_table):
    events_table
    return


@app.cell
def _(mo):
    # Search / filter in summary with widgets
    min_max_abs = mo.ui.number(value=0.0, label="Min max_abs_diff", start=0.0, step=1.0)
    min_rmse = mo.ui.number(value=0.0, label="Min rmse", start=0.0, step=1.0)
    max_corr = mo.ui.number(value=1.0, label="Max corr (keep <=)", start=-1.0, stop=1.0, step=0.01)
    show_top_n = mo.ui.slider(10, 500, value=50, step=10, label="Show top N after filtering")

    mo.md(
        """
    ## Find columns with differences
    {min_max_abs} {min_rmse} {max_corr}  
    {show_top_n}
    """
    ).batch(min_max_abs=min_max_abs, min_rmse=min_rmse, max_corr=max_corr, show_top_n=show_top_n)
    return max_corr, min_max_abs, min_rmse, show_top_n


@app.cell
def _(max_corr, min_max_abs, min_rmse, show_top_n, summary):
    filt = summary.copy()

    filt = filt[filt["max_abs_diff"] >= float(min_max_abs.value)]
    filt = filt[filt["rmse"] >= float(min_rmse.value)]
    # If corr is NaN, keep it (often constant series); otherwise apply threshold
    mc = float(max_corr.value)
    filt = filt[filt["corr"].isna() | (filt["corr"] <= mc)]

    filt = filt.sort_values(["max_abs_diff", "rmse"], ascending=False).head(int(show_top_n.value))
    filt_view = filt.reset_index().rename(columns={"index": "geohash"})
    return (filt_view,)


@app.cell
def _(filt_view, mo):
    mo.md("### Filtered columns")
    filt_table = mo.ui.dataframe(filt_view, page_size=25)
    return (filt_table,)


@app.cell
def _(filt_table):
    filt_table
    return


@app.cell
def _(absdiff, diff, mo, np, original):
    # Global summary to quickly quantify scale of differences
    global_stats = {
        "rows": int(original.shape[0]),
        "cols": int(original.shape[1]),
        "cells": int(original.size),
        "mean_abs_diff_all_cells": float(absdiff.stack().mean()),
        "p99_abs_diff_all_cells": float(absdiff.stack().quantile(0.99)),
        "max_abs_diff_all_cells": float(absdiff.to_numpy(np.float64, copy=False, na_value=np.nan).max()),
        "nonzero_diff_cells": int((diff != 0).to_numpy().sum()),
        "nonzero_diff_fraction": float((diff != 0).to_numpy().sum() / max(1, original.size)),
    }

    mo.md(
        f"""
    ## Global difference summary
    - Size: **{global_stats["rows"]}** rows × **{global_stats["cols"]}** cols (**{global_stats["cells"]}** cells)
    - Mean |diff| (all cells): **{global_stats["mean_abs_diff_all_cells"]:.6g}**
    - P99 |diff| (all cells): **{global_stats["p99_abs_diff_all_cells"]:.6g}**
    - Max |diff| (all cells): **{global_stats["max_abs_diff_all_cells"]:.6g}**
    - Nonzero diff cells: **{global_stats["nonzero_diff_cells"]}** (**{global_stats["nonzero_diff_fraction"]:.3%}**)
    """
    )
    return


if __name__ == "__main__":
    app.run()
