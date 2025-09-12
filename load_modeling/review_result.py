import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(pd):
    errors = pd.read_csv("wecc240_errors.csv", index_col=[0]).sort_index()
    errors
    return (errors,)


@app.cell
def _(mo):
    timezone_ui = mo.ui.dropdown(
        label="Timezone:",
        options=[
            "UTC",
            "America/Los_Angeles",
            "America/Phoenix",
            "America/Denver",
        ],
        value="UTC",
    )
    return (timezone_ui,)


@app.cell
def _(errors, mo, pd, timezone_ui):
    loads = pd.read_csv("wecc240_loads.csv", index_col=[0],parse_dates=[0]).sort_index()/1000
    loads.index = loads.index.tz_localize("UTC").tz_convert(timezone_ui.value)
    nodes = pd.read_csv("../data/nodes.csv", index_col=[0]).sort_index()
    _nodes = nodes.join(loads)[["Bus  Name","Bus  Number"]].dropna()
    _nodes["name"] = [f"{x['Bus  Name']} ({x['Bus  Number']} @ {n})" for n,x in _nodes.iterrows()]
    _nodes = _nodes.drop(["Bus  Name","Bus  Number"],axis=1).sort_values("name").to_dict('index')
    _options = {y["name"]:x for x,y in _nodes.items() if x in errors.index.values}
    loads_ui = mo.ui.dropdown(label="Node:",options=_options, value=list(_options)[0])
    return loads, loads_ui


@app.cell
def _(loads, loads_ui, mo, px, timezone_ui):
    _plot = px.line(
        loads[loads_ui.value], 
        labels={
            "index": f"Date/Time [{loads.index.tz}]",
            "value": "Load [MW]"},
    )
    _plot.update_layout(showlegend=False)
    mo.vstack(
        [
            mo.hstack([loads_ui,timezone_ui]),
            mo.ui.plotly(_plot),
        ]
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px


if __name__ == "__main__":
    app.run()
