import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(lm, mo):
    county = lm.County("9q9q1v")
    weather = lm.Weather(county)
    loads = lm.Loads(county)
    model = lm.NERCModel(county,loads,weather)
    with mo.status.progress_bar(total=192,remove_on_exit=True) as _bar:
        def _update(x):
            if x.startswith("Fitting"):
                _bar.update(title=x,subtitle="")
            else:
                _bar.subtitle += f"{_bar.subtitle}\n{x}"
        model.fit(cutoff=1.0,progress=_update)
    return county, loads, model, weather


@app.cell
def _(model):
    model.results
    return


@app.cell
def _(loads, model, weather):
    result = model.predict(weather.data).join(loads.data,lsuffix="_predicted",rsuffix="_actual")
    return (result,)


@app.cell
def _(county, model, result):
    result.plot.scatter("total[MW]_actual","total[MW]_predicted",1,grid=True,title=f"{county} RMSE={model.results['RMSE [%]']}")
    return


@app.cell
def _():
    import marimo as mo
    import load_models as lm
    return lm, mo


if __name__ == "__main__":
    app.run()
