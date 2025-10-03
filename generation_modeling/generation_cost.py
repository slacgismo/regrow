import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""This notebook is designed to review the generation cost data in the original NREL WECC240 model. The cost data is provided as piece-wise linear""")
    return


@app.cell
def _(pd):
    costs = pd.read_csv("generation_data.csv")
    return (costs,)


@app.cell
def _(costs, mo):
    _options = [str(x["busname"]) for n,x in costs.iterrows() if x["Cost3"] != 0.0]
    busname_ui = mo.ui.dropdown(
        options=_options, value=_options[0], label="Bus name:"
    )
    return (busname_ui,)


@app.cell
def _(busname_ui, costs, mo):
    _options = costs[costs.busname.astype(str) == busname_ui.value]
    genname_ui = mo.ui.dropdown(
        options=dict(zip(_options.Gen_Type, _options.genname)),
        value="Gas",
        label="Generator type:",
    )
    return (genname_ui,)


@app.cell
def _(mo):
    order_ui = mo.ui.number(start=0, stop=10, value=2, label="Fit order:")
    return (order_ui,)


@app.cell
def _(busname_ui, mo, order_ui):
    mo.hstack([busname_ui, order_ui], justify="start")
    return


@app.cell
def _(costs, genname_ui):
    data = costs[costs.genname==genname_ui.value].set_index("genname")[["Pmin","MW1","Cost1","MW2","Cost2","MW3","Cost3","Pmax"]].round(2)
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, np):
    price = np.array(
        [[0, data[f"Cost1"].values[0]]]
        + [
            [data[f"MW{n+1}"].values[0], data[f"Cost{n+1}"].values[0]]
            for n in range(3)
            if n == 0 or data[f"Cost{n+1}"].values[0] > 0
        ]
    ).T
    cost = np.vstack([price[0],price[0]*price[1]])
    X = cost[0]
    Y = cost[1]
    return X, Y, cost, price


@app.cell
def _(X, Y, np):
    x = np.arange(X[0],X[-1],1)
    y = np.interp(x,X,Y,X[-1]/100)
    fit = [np.polyfit(x,y,n+1) for n in range(8)]
    e = [np.sqrt(np.linalg.norm(np.polyval(p,x)-y,2)) for p in fit]
    return e, fit, x, y


@app.cell
def _(busname_ui, cost, e, fit, genname_ui, np, order_ui, plt, x, y):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.grid()
    plt.xlabel("Fit order")
    plt.ylabel("RMSE (MW)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost Fit Errors")
    plt.plot(np.round(e,2))

    plt.subplot(1,2,2)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($/MWh)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost")
    plt.scatter(cost[0],cost[1],marker='x')
    plt.plot(x,y)
    plt.plot(x,np.polyval(fit[order_ui.value-1],x))

    plt.gca()
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cvxpy as cp
    return cp, mo, np, pd, plt


if __name__ == "__main__":
    app.run()
