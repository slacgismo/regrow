import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(errors_ui, graph_prediction, graph_ui, loads_ui, mo, timezone_ui):
    mo.vstack([
        mo.hstack([loads_ui,timezone_ui]),
        mo.ui.tabs({
            "Prediction": mo.vstack([
                graph_ui,
                graph_prediction,
            ]),
            "Errors": errors_ui,
            "Data": mo.md("TODO"),
            "Validation": mo.md("TODO"),
        })])
    return


@app.cell
def _(pd):
    errors = pd.read_csv("wecc240_errors.csv", index_col=[0]).sort_index()
    errors["Delta_MAPE"] = errors["old_MAPE"] - errors["new_MAPE"]
    errors["Delta_MPED"] = errors["old_MPED"] - errors["new_MPED"]
    return (errors,)


@app.cell
def _(errors, loads_ui, mo):
    errors_ui = mo.ui.table(
        errors,
        initial_selection=[
            n for n, x in enumerate(errors.index) if x == loads_ui.value
        ],
        selection="single",
        pagination=False,
    )
    return (errors_ui,)


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
def _(mo):
    graph_ui = mo.ui.checkbox(label="August 2020 only")
    return (graph_ui,)


@app.cell
def _(graph_ui, loads, loads_ui, px):
    graph_prediction = px.line(
        loads[loads_ui.value], 
        labels={
            "index": f"Date/Time [{loads.index.tz}]",
            "value": "Load [MW]"},
        title = loads_ui.selected_key,
        range_x = ["2020-08-01","2020-09-01"] if graph_ui.value else None,
    ).update_layout(showlegend=False);
    return (graph_prediction,)


@app.cell
def _(px):
    graph_data = px.line(
    
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
