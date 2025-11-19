import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is used to review the 2011 WECC 240 model and data extensions.
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
def _(hifld_ui, scheduling_ui):
    _options = {
        scheduling_ui.value: "SCHEDULING",
        hifld_ui.value: "HIFLD",
    }
    options = [y for x,y in _options.items() if x]
    return (options,)


@app.cell
def _(options, pp, wecc240):
    model = pp.PPModel("wecc240").set_case(wecc240(options))
    return (model,)


@app.cell
def _():
    return


@app.cell
def _(mo, model, pd):
    mo.ui.table(pd.DataFrame(model.get_info().items(),columns=["Attribute","Value"]).set_index("Attribute"),
                page_size=99,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
               )
    return


@app.cell
def _(mo, model):
    mo.ui.tabs({x:model.get_data(x) for x in ["bus","branch","gen","gencost","dcline","dclinecost","gis"]})
    return


@app.cell
def _():
    import sys
    import marimo as mo
    import pandas as pd
    import numpy as np
    import ppmodel as pp
    from psse import PSSE
    from psse2pp import PSSE2PP
    from wecc240 import wecc240
    sys.path.append("../data")
    import utils
    return mo, pd, pp, wecc240


if __name__ == "__main__":
    app.run()
