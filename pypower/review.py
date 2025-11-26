import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is used to review the 2011 WECC 240 model with optional 2020 and REGROW data extensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following model options are available.
    """)
    return


@app.cell
def _(mo):
    scheduling_ui = mo.ui.checkbox(label="2020 model")
    hifld_ui = mo.ui.checkbox(label="HIFLD", disabled=True)
    loads_ui = mo.ui.checkbox(label="Loads", disabled=True)
    renewables_ui = mo.ui.checkbox(label="Renewables", disabled=True)
    mo.hstack(
        [
            mo.md("**WECC240 Data Options**:"),
            scheduling_ui,
            hifld_ui,
            loads_ui,
            renewables_ui,
        ],
        justify="start",
    )
    return hifld_ui, scheduling_ui


@app.cell
def _(PPData, hifld_ui, mo, pp, scheduling_ui, wecc240):
    _options = {
        scheduling_ui.value: "SCHEDULING",
        hifld_ui.value: "HIFLD",
    }
    options = [y for x,y in _options.items() if x]
    model = pp.PPModel("wecc240",case=wecc240(options))
    _data = PPData(model)
    _data.set_input("bus","PD","tests/load.csv",scale=10)
    _data.set_input("bus","QD","tests/load.csv",scale=1)
    _data.set_output("bus","VM","results/bus_vm.csv",formatting=".3f")
    _data.set_output("bus","VA","results/bus_va.csv",formatting=".4f")
    _data.set_output("bus","PD","results/bus_pd.csv",formatting=".4f")
    _data.set_output("bus","QD","results/bus_qd.csv",formatting=".4f")

    _data.set_recorder("results/cost.csv","cost",["cost"],
        scale=model.case['baseMVA'],formatting=".2f")
    _data.set_recorder("results/cost.csv","cost_pumva",["cost"],
        formatting=".2f")
    get_model,set_model = mo.state(model)
    return model, options


@app.cell
def _(mo, model, pd):
    info_ui = mo.ui.table(
        pd.DataFrame(
            model.get_info().items(), columns=["Attribute", "Value"]
        ).set_index("Attribute"),
        page_size=99,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        text_justify_columns={"Value": "right"},
    ).left()
    return (info_ui,)


@app.cell
def _(mo, model, options, result):
    options
    result
    data_model_ui = mo.ui.tabs(
            {
                n: mo.ui.table(
                    data=x,
                    show_data_types=False,
                    selection=None,
                    text_justify_columns={y: "right" for y in x.columns},
                    _internal_preload=False,
                )
                for n, x in {
                    z: model.get_data(z)
                    for z in [
                        "bus",
                        "branch",
                        "gen",
                        "gencost",
                        "dcline",
                        "dclinecost",
                        "gis",
                    ]
                }.items()
            },lazy=True)
    return (data_model_ui,)


@app.cell
def _(model, options, result):
    options
    result
    inputs = {"/".join(x):y["data"] for x,y in model.inputs.items()}
    return (inputs,)


@app.cell
def _(model, options, pd, result):
    options
    result
    outputs = {}
    for _x in model.outputs.keys():
        try:
            _data = pd.read_csv(
                _x,
                index_col=[0],
                parse_dates=[0],
                dtype=float,
                low_memory=False,
            )
        except Exception as error:
            _data = f"ERROR: {error}"
        outputs[_x] = _data
    return (outputs,)


@app.cell
def _(model, options, pd, result):
    options
    result
    recorders = {}
    for _x in model.recorders.keys():
        try:
            _data = pd.read_csv(
                _x,
                index_col=[0],
                parse_dates=[0],
                dtype=float,
                low_memory=False,
            )
        except Exception as error:
            _data = f"ERROR: {error}"
        recorders[_x] = _data
    return (recorders,)


@app.cell
def _(inputs, mo):
    _tabs = {
        x: mo.ui.table(
            y,
            show_data_types=False,
            selection=None,
            text_justify_columns={y: "right" for y in y.columns},
            _internal_preload=False,
        ).left()
        for x, y in inputs.items()
    }
    data_inputs_ui = mo.ui.tabs(_tabs, lazy=True)
    return (data_inputs_ui,)


@app.cell
def _(mo, outputs, pd):
    _tabs = {}
    for _x, _y in outputs.items():
        try:
            _data = pd.read_csv(_x, low_memory=False)
        except Exception as _err:
            _data = f"ERROR: {_err}"
        _tabs[_x] = _y if isinstance(_y,str) else (
            _data
            if isinstance(_data, str)
            else mo.ui.table(
                _data,
                show_data_types=False,
                selection=None,
                text_justify_columns={y: "right" for y in _data.columns},
                _internal_preload=False,
            )
        )
    data_outputs_ui = mo.ui.tabs(_tabs,lazy=True)
    return (data_outputs_ui,)


@app.cell
def _(mo, recorders):
    _tabs = {}
    for _x,_y in recorders.items():
        _tabs[_x] = (
            None
            if _y is None
            else mo.ui.table(
                _y,
                show_data_types=False,
                selection=None,
                text_justify_columns={y: "right" for y in _y.columns},
                _internal_preload=False,
            ).left()
        )
    data_recorders_ui = mo.ui.tabs(_tabs,lazy=True)
    return (data_recorders_ui,)


@app.cell
def _(data_inputs_ui, data_model_ui, data_outputs_ui, data_recorders_ui, mo):
    data_ui = mo.ui.tabs(
        {
            "Model": data_model_ui,
            "Inputs": data_inputs_ui,
            "Outputs": data_outputs_ui,
            "Recorders": data_recorders_ui,
        }
    )
    return (data_ui,)


@app.cell
def _(inputs, mo, model, outputs, pg, recorders, result):
    result
    graph_ui = mo.ui.tabs(
        {
            "Voltage": pg.PPPlots(model).voltage().gca(),
            "Generation": pg.PPPlots(model).generation().gca(),
            "Load": pg.PPPlots(model).load().gca(),
            "Inputs": mo.ui.tabs({x:y if isinstance(y,str) else y.plot(figsize=(15,8),grid=True) for x,y in inputs.items()},lazy=True),
            "Outputs": mo.ui.tabs({x:y if isinstance(y,str) else y.plot(figsize=(15,8),grid=True) for x,y in outputs.items()},lazy=True),
            "Recorders": mo.ui.tabs({x:y if isinstance(y,str) else y.plot(figsize=(15,8),grid=True) for x,y in recorders.items()},lazy=True),
        },
        lazy=True,
    )
    return (graph_ui,)


@app.cell
def _(data_ui, graph_ui, info_ui, mo):
    mo.accordion(
        {
            "**Overview**": info_ui,
            "**Data**": data_ui,
            "**Plots**": graph_ui,
        },
        lazy=True,
    )
    return


@app.cell
def _(continue_ui, end_ui, mo, opf_ui, run_ui, start_ui, verbose_ui):
    mo.hstack([start_ui,end_ui,run_ui,opf_ui,continue_ui,verbose_ui],justify='start')
    return


@app.cell
def _(mo):
    start_ui = mo.ui.date(start="2018-01-01",stop="2023-01-01",value="2020-08-01",label="Start date:")
    end_ui = mo.ui.date(start="2018-01-01",stop="2023-01-01",value="2020-08-01",label="End date:")
    opf_ui = mo.ui.checkbox(label="Use AC OPF")
    continue_ui = mo.ui.checkbox(label="Ignore failures")
    verbose_ui = mo.ui.checkbox(label="Verbose output (edit mode only)")
    return continue_ui, end_ui, opf_ui, start_ui, verbose_ui


@app.cell
def _(mo, set_ready):
    run_ui = mo.ui.button(label="Run",on_click=lambda x:set_ready(True))
    return (run_ui,)


@app.cell
def _():
    # mo.md(f"<font color=blue>HINT: Click {run_ui} to start simulation</font>") if model.profile is None else None
    return


@app.cell
def _(mo):
    get_ready,set_ready = mo.state(False)
    return get_ready, set_ready


@app.cell
def _(
    continue_ui,
    dt,
    end_ui,
    get_ready,
    mo,
    model,
    opf_ui,
    pd,
    ps,
    pytz,
    run_ui,
    set_ready,
    start_ui,
    verbose_ui,
):
    run_ui
    result = {"stdout":None,"stderr":None}
    if get_ready():
        model.options["OUT_ALL"] = 1 if verbose_ui.value else 0
        _start = dt.datetime.combine(
            start_ui.value, dt.time(0, 0, 0, tzinfo=pytz.UTC)
        )
        _end = dt.datetime.combine(end_ui.value, dt.time(0, 0, 0, tzinfo=pytz.UTC))
        _freq = "1h"
        _total = len(pd.date_range(_start, _end, freq=_freq))
        with mo.status.progress_bar(
            total=_total,
            title="Running WECC240 model",
            remove_on_exit=True,
        ) as _bar:
            with mo.capture_stdout() as _stdout:
                with mo.capture_stdout() as _stderr:
                    solver = ps.PPSolver(model)
                    solver.run_timeseries(
                        _start,
                        _end,
                        freq=_freq,
                        progress=lambda x: _bar.update(subtitle=x,increment=1),
                        call_on_fail=None,
                        use_acopf=opf_ui.value,
                        stop_on_fail=not continue_ui.value,
                    )
                result["stderr"] = _stderr.getvalue()
            result["stdout"] = _stdout.getvalue()
        set_ready(False)
    return (result,)


@app.cell
def _(mo, model, run_ui):
    run_ui
    if model.profile:
        _result = mo.accordion(
            {
                f"<font color={'red' if model.errors else 'blue'}>**{(len(model.errors))} error{'' if len(model.errors) == 1 else 's'} reported**</font>": mo.md(
                    "\n\n".join(
                        [f"{n+1}. {m}" for n, m in enumerate(model.errors)]
                    )
                ),
            }
        )
    else:
        _result = None
    _result
    return


@app.cell
def _(mo, model, pd, result):
    _result = {
        "**Performance profile**": None if model.profile is None else mo.ui.table(
            pd.DataFrame(model.profile.items(), columns=["Metric", "Value"]).set_index("Metric"),
            selection=None,
            page_size=99,
            text_justify_columns={"Value": "right"},
            ).left()
        }
    if result["stdout"]:
        _result["**Solver output**"] = result["stdout"]
    if result["stderr"]:
        _result["**Solver errors**"] = result["stderr"]
    mo.accordion(_result) if model.profile else mo.md("---")
    return


@app.cell
def _(mo):
    mo.md(f"""
    Marimo version {mo.__version__}
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import datetime as dt
    import pytz
    import pandas as pd
    import ppmodel as pp
    import ppplots as pg
    import ppsolver as ps
    from ppdata import PPData
    from wecc240 import wecc240
    return PPData, dt, mo, pd, pg, pp, ps, pytz, wecc240


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
