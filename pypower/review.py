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
    scheduling_ui = mo.ui.checkbox(label='2020 model data')
    hifld_ui = mo.ui.checkbox(label="HIFLD generator data")
    mo.hstack([mo.md("**WECC240 Data Options**:"),scheduling_ui,hifld_ui],justify='start')
    return hifld_ui, scheduling_ui


@app.cell
def _(hifld_ui, mo, pp, scheduling_ui, wecc240):
    _options = {
        scheduling_ui.value: "SCHEDULING",
        hifld_ui.value: "HIFLD",
    }
    options = [y for x,y in _options.items() if x]
    get_model,set_model = mo.state(pp.PPModel("wecc240",case=wecc240(options)))
    return (get_model,)


@app.cell
def _(get_model):
    model = get_model()
    return (model,)


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
def _(mo, model):
    data_ui = mo.ui.tabs(
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
        }
    )
    return (data_ui,)


@app.cell
def _(mo, model, pg):
    graph_ui = mo.ui.tabs({
        "Voltage": pg.PPPlots(model).voltage().gca(),
        "Generation": pg.PPPlots(model).generation().gca(),
        "Load": pg.PPPlots(model).load().gca(),
    })
    return (graph_ui,)


@app.cell
def _(data_ui, graph_ui, info_ui, mo):
    mo.accordion(
        {
            "**Overview**": info_ui,
            "**Data**": data_ui,
            "**Plots**": graph_ui,
        }
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
def _(mo, model, run_ui):
    mo.md(f"<font color=blue>HINT: Click {run_ui} to start simulation</font>") if model.profile is None else None
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
                        call_on_fail=False,
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
    import sys
    import datetime as dt
    import pytz
    import marimo as mo
    import pandas as pd
    import ppmodel as pp
    import ppplots as pg
    import ppsolver as ps
    return dt, mo, pd, pg, pp, ps, pytz


@app.cell
def _():
    from wecc240 import wecc240
    return (wecc240,)


if __name__ == "__main__":
    app.run()
