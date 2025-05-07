

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _(os):
    datasets=[os.path.splitext(x)[0] for x in os.listdir("geodata/counties")]
    return (datasets,)


@app.cell
def _(mo, pd):
    _data = pd.read_csv(f"geodata/counties/total.csv",index_col=0)
    column_ui = mo.ui.dropdown(label="Geocode:",options=sorted(_data.columns),value=_data.columns[0])
    return (column_ui,)


@app.cell
def _(column_ui):
    column_ui
    return


@app.cell
def _(column_ui, mo, pd):
    counties = pd.read_csv("counties.csv",index_col="geocode")
    import tzinfo
    try:
        tz = tzinfo.TIMEZONES[f"{counties.loc[column_ui.value]["fips"]:05.0f}"]
    except:
        tz = tzinfo.TIMEZONES[f"{counties.loc[column_ui.value]["fips"]:05.0f}"[:2]]
    mo.hstack([mo.md(f"**{x.title()}**: {y}") for x,y in counties.loc[column_ui.value].to_dict().items()]+[mo.md(f"**Timezone**: {tz}")])
    return


@app.cell
def _(column_ui, datasets, mo, pd, px):
    data = {x:pd.read_csv(f"geodata/counties/{x}.csv",index_col=0)[column_ui.value] for x in datasets}
    data["baseload"] = data["total"] - data["heating"] - data["cooling"]
    tabs_ui = mo.ui.tabs({x:mo.ui.plotly(px.line(y)) for x,y in data.items()})
    return data, tabs_ui


@app.cell
def _(tabs_ui):
    tabs_ui
    return


@app.cell
def _(mo):
    load_selection = {x:mo.ui.checkbox(label=x,value=True) for x in ["total","heating","cooling","baseload"]}
    load_ui = mo.ui.array(load_selection.values())
    mo.hstack([mo.md("**Loads**:")]+list(load_ui),justify='start')
    return load_selection, load_ui


@app.cell
def _(data, load_selection, load_ui, pd, sp):
    summary = (
        pd.DataFrame(
            {
                "temperature": data["temperature"].values,
                "timestamp": data["temperature"].index,
            }
        )
        .set_index("timestamp")
        .join(
            pd.DataFrame(
                {
                    "heating": data["heating"].values,
                    "timestamp": data["heating"].index,
                }
            ).set_index("timestamp")
        )
        .join(
            pd.DataFrame(
                {
                    "cooling": data["cooling"].values,
                    "timestamp": data["cooling"].index,
                }
            ).set_index("timestamp")
        )
        .join(
            pd.DataFrame(
                {
                    "baseload": data["total"].values-data["heating"].values-data["cooling"].values,
                    "timestamp": data["cooling"].index,
                }
            ).set_index("timestamp")
        )
        .join(
            pd.DataFrame(
                {
                    "total": data["total"].values,
                    "timestamp": data["total"].index,
                }
            ).set_index("timestamp")
        )
    ).dropna()
    results = {"plot":summary.plot(x="temperature",y=[x for x,y in zip(load_selection.keys(),load_ui.value) if y],markersize=0.75,marker=".",linestyle="",grid=True,figsize=(15,7)),
     "heating":sp.linregress(summary["temperature"],summary["heating"])[:2],
     "cooling":sp.linregress(summary["temperature"],summary["cooling"])[:2],
    }
    return results, summary


@app.cell
def _(results):
    results["plot"]
    return


@app.cell
def _(summary):
    cooling = summary[summary["cooling"]>1]
    heating = summary[summary["heating"]>1]
    return cooling, heating


@app.cell
def _(cooling, heating, mo, sp):
    cooling_fit = sp.linregress(cooling["temperature"],cooling["cooling"])[:6]
    print(cooling_fit)
    _x = [cooling["temperature"].min(),cooling["temperature"].max()]
    _ax1 = cooling.plot(x="temperature",y="cooling",marker=".",linewidth=0.1,markersize=0.75)
    _ax1.plot(_x,[cooling_fit[0]*_x[0]+cooling_fit[1],cooling_fit[0]*_x[1]+cooling_fit[1]])
    _ax1.grid()
    _ax1.set_xlabel("Temperature ($^\\circ$C)")
    _ax1.set_ylabel("Load (MW)")
    _ax1.legend(["Load data",f"Fit: $L={cooling_fit[0]:.1f}T{cooling_fit[1]:+.1f}$"])
    _ax1.set_title("Cooling")
    _ax1.set_ylim([0,cooling["cooling"].max()])

    heating_fit = sp.linregress(heating["temperature"],heating["heating"])[:4]
    _x = [heating["temperature"].min(),heating["temperature"].max()]
    _ax2 = heating.plot(x="temperature",y="heating",marker=".",linewidth=0.1,markersize=0.75)
    _ax2.plot(_x,[heating_fit[0]*_x[0]+heating_fit[1],heating_fit[0]*_x[1]+heating_fit[1]])
    _ax2.grid()
    _ax2.set_xlabel("Temperature ($^\\circ$C)")
    _ax2.set_ylabel("Load (MW)")
    _ax2.legend(["Load data",f"Fit: $L={heating_fit[0]:.1f}T{heating_fit[1]:+.1f}$"])
    _ax2.set_title("Heating")
    _ax2.set_ylim([0,heating["heating"].max()])

    mo.hstack([_ax2,_ax1])
    return


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import scipy.stats as sp
    return mo, os, pd, px, sp


if __name__ == "__main__":
    app.run()
