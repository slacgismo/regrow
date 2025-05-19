import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(counties, mo):
    _options = {f"{y.county} {y.usps} ({x})":x for x,y in counties.data.set_index(["region","geocode"]).loc["WECC"].iterrows()}
    county_ui = mo.ui.dropdown(options=_options,value=list(_options)[0])
    return (county_ui,)


@app.cell
def _(county_ui):
    county_ui
    return


@app.cell
def _(county_ui, lm, mo):
    county = lm.County(county_ui.value)
    weather = lm.Weather(county)
    loads = lm.Loads(county)
    model = lm.NERCModel(county, loads, weather)
    model.holdout = [x for x in weather.data.index if x.timestamp() // (24 * 7) % 4 == 0]
    with mo.status.progress_bar(total=192, remove_on_exit=True) as _bar:

        def _update(x):
            if x.startswith("Fitting"):
                _bar.update(title=x, subtitle="")
            else:
                _bar.subtitle += f"{_bar.subtitle}\n{x}"

        model.fit(cutoff=1.0, progress=_update)
    return county, loads, model, weather


@app.cell
def _(loads, model, weather):
    result = model.predict(weather.data).join(
        loads.data, lsuffix="_predicted", rsuffix="_actual"
    )
    return (result,)


@app.cell
def _(county, model, result):
    scatter_plot = result.plot.scatter(
        "total[MW]_actual",
        "total[MW]_predicted",
        1,
        grid=True,
        title=f"{county} RMSE={model.results['RMSE [%]']}",
        figsize=(10,4),
    )
    return (scatter_plot,)


@app.cell
def _(county, model, result):
    timeseries_plot = result.plot(
        y=["total[MW]_actual", "total[MW]_predicted"],
        grid=True,
        title=f"{county} RMSE={model.results['RMSE [%]']}",
        figsize=(10,4),
    )
    return (timeseries_plot,)


@app.cell
def _(mo, result, scatter_plot, timeseries_plot):
    mo.ui.tabs({
        "Timeseries" : mo.mpl.interactive(timeseries_plot),
        "Scatter" : mo.mpl.interactive(scatter_plot),
        "Data" : result,
    })
    return


@app.cell
def _():
    import marimo as mo
    import load_models as lm
    import counties
    return counties, lm, mo


if __name__ == "__main__":
    app.run()
