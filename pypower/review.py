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


@app.cell(hide_code=True)
def _(mo):
    scheduling_ui = mo.ui.checkbox(label='2020 model data')
    hifld_ui = mo.ui.checkbox(label="HIFLD generator data")
    mo.hstack([mo.md("**WECC240 Data Options**:"),scheduling_ui,hifld_ui],justify='start')
    return hifld_ui, scheduling_ui


@app.cell(hide_code=True)
def _(hifld_ui, pp, scheduling_ui, wecc240):
    _options = {
        scheduling_ui.value: "SCHEDULING",
        hifld_ui.value: "HIFLD",
    }
    options = [y for x,y in _options.items() if x]
    model = pp.PPModel("wecc240",case=wecc240(options))
    return (model,)


@app.cell(hide_code=True)
def _(mo, model, pd):
    _info = mo.ui.table(
                pd.DataFrame(
                    model.get_info().items(), columns=["Attribute", "Value"]
                ).set_index("Attribute"),
                page_size=99,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                text_justify_columns={"Value": "right"},
            )
    _data = mo.ui.tabs(
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
    mo.accordion(
        {
            "Model information": _info.left(),
            "Model data": _data,
        }
    )
    return


@app.cell
def _(continue_ui, end_ui, mo, opf_ui, run_ui, start_ui):
    mo.hstack([start_ui,end_ui,run_ui,opf_ui,continue_ui],justify='start')
    return


@app.cell
def _(mo):
    start_ui = mo.ui.date(start="2018-01-01",stop="2023-01-01",value="2020-08-01",label="Start date:")
    end_ui = mo.ui.date(start="2018-01-01",stop="2023-01-01",value="2020-08-02",label="End date:")
    opf_ui = mo.ui.checkbox(label="Use AC OPF")
    continue_ui = mo.ui.checkbox(label="Ignore failures")
    return continue_ui, end_ui, opf_ui, start_ui


@app.cell
def _(mo, model, run_simulation):
    run_ui = mo.ui.button(label="Run",on_click=run_simulation)
    mo.md(f"<font color=blue>HINT: Click {run_ui} to start simulation</font>") if not model.profile else None
    return (run_ui,)


@app.cell
def _(mo):
    get_result,set_result = mo.state(None)
    get_profile,set_profile = mo.state(None)
    return get_profile, get_result, set_profile, set_result


@app.cell
def _(
    continue_ui,
    dt,
    end_ui,
    mo,
    model,
    opf_ui,
    pd,
    pytz,
    set_profile,
    set_result,
    start_ui,
):
    def run_simulation(*args):
        _start = dt.datetime.combine(start_ui.value,dt.time(0,0,0,tzinfo=pytz.UTC))
        _end = dt.datetime.combine(end_ui.value,dt.time(0,0,0,tzinfo=pytz.UTC))
        _freq = "1h"
        _total = len(pd.date_range(_start,_end,freq=_freq))
        with mo.status.progress_bar(total=_total, title="Running WECC240 model",remove_on_exit=True) as _bar:
            result = model.run_timeseries(
                _start,
                _end,
                freq=_freq,
                progress=lambda x: _bar.update(subtitle=x, increment=1),
                call_on_fail=False,
                use_acopf=opf_ui.value,
                stop_on_fail=not continue_ui.value,
            )
            if result is None:
                _bar.subtitle = "Done"
                set_result([])
            else:
                _bar.subtitle = result
                set_result(result)
            set_profile(model.profile)
    return (run_simulation,)


@app.cell
def _(get_profile, get_result, mo):
    if get_profile():
        _result = mo.accordion(
            {
                f"<font color={'red' if get_result() else 'blue'}>**{(len(get_result()))} error{'' if len(get_result()) == 1 else 's'} occurred**</font>": mo.md(
                    "\n\n".join(
                        [f"{n+1}. {m}" for n, m in enumerate(get_result())]
                    )
                ),
            }
        )
    else:
        _result = None
    _result
    return


@app.cell
def _(get_profile, mo, pd):
    if get_profile():
        _profile = pd.DataFrame(get_profile().items(), columns=["Metric", "Value"]).set_index("Metric")
        _result = mo.accordion({"**Performance profile**":
            mo.ui.table(
                _profile,
                selection=None,
                page_size=99,
                text_justify_columns={"Value": "right"},
            ).left()})
    else:
        _result = None
    _result
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
    from wecc240 import wecc240
    return dt, mo, pd, pp, pytz, wecc240


if __name__ == "__main__":
    app.run()
