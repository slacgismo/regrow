import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(
    errors,
    errors_ui,
    graph_holdout,
    graph_prediction,
    graph_training,
    graph_ui,
    loads_ui,
    mo,
    pd,
    timezone_ui,
):
    mo.vstack([
        mo.hstack([loads_ui,timezone_ui]),
        mo.ui.tabs({
            "Prediction": mo.vstack([
                graph_ui,
                graph_prediction,
                pd.DataFrame(errors.loc[loads_ui.value]).T,
            ]),
            "Errors": errors_ui,
            "Training": graph_training,
            "Holdout": graph_holdout,
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
def _(pd):
    training = (pd.read_csv("../data/geodata/total.csv",index_col=[0],parse_dates=[0])/1000).round(3)
    print(training["9wdb95"])
    weather = (pd.read_csv("../data/geodata/temperature.csv",index_col=[0],parse_dates=[0])*9/5+32).round(1)
    return training, weather


@app.cell
def _(errors, mo, pd, timezone_ui):
    prediction = pd.read_csv("wecc240_loads.csv", index_col=[0],parse_dates=[0]).sort_index()/1000
    prediction.index = prediction.index.tz_localize("UTC").tz_convert(timezone_ui.value)
    nodes = pd.read_csv("../data/nodes.csv", index_col=[0]).sort_index()
    _nodes = nodes.join(prediction)[["Bus  Name","Bus  Number"]].dropna()
    _nodes["name"] = [f"{x['Bus  Name']} ({x['Bus  Number']} @ {n})" for n,x in _nodes.iterrows()]
    _nodes = _nodes.drop(["Bus  Name","Bus  Number"],axis=1).sort_values("name").to_dict('index')
    _options = {y["name"]:x for x,y in _nodes.items() if x in errors.index.values}
    loads_ui = mo.ui.dropdown(label="Node:",options=_options, value=list(_options)[0])
    return loads_ui, prediction


@app.cell
def _(loads_ui, prediction):
    print(prediction.loc["2018-12",loads_ui.value])
    return


@app.cell
def _(mo):
    graph_ui = mo.ui.checkbox(label="August 2020 only")
    return (graph_ui,)


@app.cell
def _(graph_ui, loads_ui, prediction, px):
    graph_prediction = px.line(
        prediction.loc["2020-08",loads_ui.value] if graph_ui.value else prediction[loads_ui.value], 
        labels={
            "index": f"Date/Time [{prediction.index.tz}]",
            "value": "Load [MW]"},
        title = loads_ui.selected_key,
        # range_x = ["2020-08-01","2020-09-01"] if graph_ui.value else None,
    ).update_layout(showlegend=False);
    return (graph_prediction,)


@app.cell
def _(loads_ui, prediction, px, training):
    graph_training = px.line(
        training.loc[:"2018-12",loads_ui.value],
        labels={
            "index": f"Date/Time [{prediction.index.tz}]",
            "value": "Load [MW]"},
        title = loads_ui.selected_key,
    ).update_layout(showlegend=False);
    return (graph_training,)


@app.cell
def _(loads_ui, pd, prediction, px, training, weather):
    _training = pd.DataFrame(training.loc["2018-12",loads_ui.value])
    _training.columns = ["Training"]
    print(_training)
    _prediction = pd.DataFrame(prediction.loc["2018-12",loads_ui.value])
    _prediction.columns = ["Prediction"]
    print(_prediction)
    _weather = pd.DataFrame(weather.loc["2018-12",loads_ui.value])
    _weather.columns = ["Temperature"]
    _data = _training.join(_prediction)#.join(_weather)
    graph_holdout = px.line(
        _data,
        labels={
            "index": f"Date/Time [{prediction.index.tz}]",
            "value": "Load [MW]"},
        title = loads_ui.selected_key,
    ).update_layout();
    return (graph_holdout,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px


if __name__ == "__main__":
    app.run()
