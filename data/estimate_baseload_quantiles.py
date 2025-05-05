import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        This notebook reads the county-level geodata for total, heating, and cooling loads from the files in `geodata/counties`. It then computes the baseload as $baseload = total - heating - cooling$. The baseload data is then used to fit a smooth multi-periodic consistent quantile estimate for annual, weekly, and daily component of the load for the 50% and 98% quantiles. The resulting quantiles estimates for each county are stored with the actual data in `geodata/baseload`.

        Notes:

        1. If a quantile estimate file already exists in `geodata/baseload`, the results are simply loaded and the original county-level load data is not reprocessed.

        2. If a quantile estimate is already resident in memory, nothing is done.

        3. A 5% holdout test is performed on the estimate and reported when the data is processed. The result of holdout test are not recorded.

        4. The data viewer is provided to facilitate a visual check of the quantile estimates.
        """
    )
    return


@app.cell
def _(pd, tzinfo):
    # load data
    UTC = tzinfo.TZ("UTC",0,0)
    timezones = {x:tzinfo.TZ(x,*y) for x,y in {"EST":(-5,0),"CST":(-6,0),"MST":(-7,0),"PST":(-8,0)}.items()}

    cooling = pd.read_csv("geodata/counties/cooling.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True).dropna()

    heating = pd.read_csv("geodata/counties/heating.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True).dropna()

    baseload = (pd.read_csv("geodata/counties/total.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True) - heating - cooling).dropna()

    counties = pd.read_csv("counties.csv",index_col=["geocode"])
    return UTC, baseload, cooling, counties, heating, timezones


@app.cell
def _(UTC, baseload, counties, dt, np, timezones, tzinfo):
    #
    # Baseload model using truncated fourier
    #
    baseload_data = {}
    for _geocode in baseload.columns:

        # get hours
        _fips = f"{counties.loc[_geocode].fips:05.0f}"
        try:
            _tz = timezones[tzinfo.TIMEZONES[_fips][:3]]
        except:
            _tz = timezones[tzinfo.TIMEZONES[_fips[:2]][:3]]
        _year = dt.datetime(baseload.index[0].year,1,1,0,0,0,tzinfo=_tz).timestamp()
        baseload_hour = [int((x.timestamp()-_year)/3600) for x in baseload.index.tz_localize(_tz).tz_convert(UTC)]

        # get values
        _data = np.full(max(baseload_hour)+1,float('nan'))
        _data[baseload_hour] = baseload[_geocode].values

        baseload_data[_geocode] = _data
    return baseload_data, baseload_hour


@app.cell
def _(baseload_data, mo):
    mo.md(f"{len(baseload_data)} baseloads found")
    return


@app.cell
def _():
    baseload_fit = {}
    return (baseload_fit,)


@app.cell
def _(
    baseload,
    baseload_data,
    baseload_fit,
    counties,
    np,
    os,
    pd,
    qe,
    random,
    timezones,
    tzinfo,
):
    holdout_fraction = 0.05
    _timerange = pd.date_range(
        "2018-01-01 00:00:00+00:00", "2023-01-02 00:00:00+00:00", freq="1h"
    )
    _spq = qe.SmoothPeriodicQuantiles(
        num_harmonics=3,
        periods=[365.24 * 24, 7 * 24, 24],
        weight=0.1,
        quantiles=[0.5, 0.98],
    )
    total_time = 0
    _dir = os.path.join(*"geodata/baseload".split("/"))
    os.makedirs(_dir, exist_ok=True)
    count_found = 0
    count_loaded = 0
    count_processed = 0
    baseloads = dict(
        zip(
            [
                f"geodata/counties/baseload_q{x * 100:.0f}.csv"
                for x in _spq.quantiles
            ],
            [[]] * 2,
        )
    )
    baseloads["geodata/counties/baseload_data.csv"] = baseload.round(3)

    for _geocode in baseload.columns:
        _file = os.path.join(_dir, f"{_geocode}.csv.gz")
        if os.path.exists(_file):  # do not reprocess previously obtained results
            if (
                _geocode not in baseload_fit
            ):  # reload only if not already in memory
                print(
                    f"Loading {counties.loc[_geocode].county} {counties.loc[_geocode].usps} ({_geocode})",
                    flush=True,
                )
                baseload_fit[_geocode] = pd.read_csv(
                    _file, index_col=[0], parse_dates=[0], low_memory=True
                )
                count_loaded += 1
            else:
                count_found += 1
        else:
            _fips = f"{counties.loc[_geocode].fips:05.0f}"
            try:
                _tz = timezones[tzinfo.TIMEZONES[_fips][:3]]
            except:
                _tz = timezones[tzinfo.TIMEZONES[_fips[:2]][:3]]
            print(
                f"Processing {counties.loc[_geocode].county} {counties.loc[_geocode].usps} ({_geocode} {_tz})",
                end="...",
                flush=True,
            )
            _hours = np.array(
                [
                    int((x.timestamp() - _timerange[0].timestamp()) / 3600)
                    for x in _timerange
                ]
            )

            # prepare full data range (include prediction range)
            _data = np.full(max(_hours) + 1, float("nan"))
            _data[: len(baseload_data[_geocode])] = baseload_data[_geocode]

            # generate data holdout for testing
            _holdout = random.sample(
                range(len(baseload_data[_geocode])),
                int(len(baseload_data[_geocode]) * holdout_fraction),
            )

            # isolate training data
            _train = np.copy(_data)
            _train[_holdout] = float("nan")

            # train
            _spq.fit(_train)

            # collect predictions
            _df = pd.DataFrame(
                data=_spq.fit_quantiles[_hours, :],
                index=_timerange,
                columns=[f"q{x * 100:.0f}" for x in _spq.quantiles],
            ).round(3)

            # add data
            _df["data"] = _data.round(3)

            # name index
            _df.index.name = "timestamp"

            # save to file
            _df.to_csv(_file, header=True, index=True)

            # save to memory
            baseload_fit[_geocode] = _df

            # record time required
            total_time += _spq.fit_time

            # test performance using holdout data
            _test = _df.iloc[_holdout].dropna()
            print(
                f"Testing on {len(_test)} holdouts ({holdout_fraction * 100:.0f})"
            )
            for _column, _quantile in [
                (f"q{x * 100:.0f}", x) for x in _spq.quantiles
            ]:
                _frac = len(_test[_test["data"] < _test[_column]]) / len(_test)
                print(
                    f"  {_quantile * 100:.0f}% quantile test: {_frac * 100:.1f}%"
                )

            # report progress
            print(f"done in {_spq.fit_time:.1f} seconds")
            count_processed += 1

        # collate baseloads
        for _n, _geodata, _quantile in [
            (n, f"q{x * 100:.0f}", f"geodata/counties/baseload_q{x * 100:.0f}.csv")
            for n, x in enumerate(_spq.quantiles)
        ]:
            baseloads[_quantile].append(
                pd.DataFrame(
                    data=baseload_fit[_geocode][_geodata].values,
                    index=baseload_fit[_geocode][_geodata].index,
                    columns=[_geocode],
                )
            )

    for _file, _data in baseloads.items():
        if not os.path.exists(_file):
            print(f"Collating {_file}", end="...", flush=True)
            if isinstance(_data,list):
                _data = pd.concat(_data, axis=1).dropna()
            _data.to_csv(_file, header=True, index=True)
            print("ok")

    print(f"{count_found} already loaded")
    print(f"{count_loaded} reloaded from file")
    print(f"{count_processed} processed in {total_time:.1f} seconds")
    return (
        baseloads,
        count_found,
        count_loaded,
        count_processed,
        holdout_fraction,
        total_time,
    )


@app.cell
def _(counties, mo):
    state_ui = mo.ui.dropdown(label="State:",options=counties.usps.unique(),value=counties.usps.unique()[0])
    return (state_ui,)


@app.cell
def _(counties, mo, state_ui):
    _counties = counties[counties.usps==state_ui.value]
    _options = dict(zip(_counties.county,_counties.index))
    county_ui = mo.ui.dropdown(label="County:",options=_options,value=_counties.county.iloc[0])
    return (county_ui,)


@app.cell
def _(county_ui, mo, state_ui):
    mo.hstack([state_ui,county_ui],justify='start')
    return


@app.cell
def _(baseload_fit, county_ui, mo, state_ui, total_time):
    total_time # wait for processing to complete
    mo.stop(not county_ui.value in baseload_fit,f"Data for {county_ui.selected_key} {state_ui.value} ({county_ui.value}) not in baseload_fit results")
    week_ui = mo.ui.slider(label="Week:",start=0,stop=int(len(baseload_fit[county_ui.value])/7/24),value=0,show_value=True)
    days_ui = mo.ui.slider(label="Days:",steps=[7,14,21,28,92,184,365],value=7,show_value=True)
    mo.hstack([week_ui,days_ui],justify='start')
    return days_ui, week_ui


@app.cell
def _(baseload_fit, county_ui, days_ui, np, week_ui):
    import matplotlib.pyplot as plt

    _geocode = county_ui.value
    _week = week_ui.value
    _days = days_ui.value
    _index = np.s_[
        24 * 7 * _week : min(
            24 * (7 * _week + _days), len(baseload_fit[_geocode])
        )
    ]

    baseload_fit[_geocode][_index].plot(
        figsize=(15, 7),
        alpha=0.5,
        style=["-", "-", ":"],
        label=baseload_fit[_geocode].columns,
        grid=True,
        color=["b", "r", "k"],
    )
    return (plt,)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import datetime as dt
    import pandas as pd
    import numpy as np
    import tzinfo
    import spcqe as qe
    import random
    return dt, mo, np, os, pd, qe, random, sys, tzinfo


if __name__ == "__main__":
    app.run()
