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
    data = costs[costs.genname==genname_ui.value].set_index("genname")[["Pmin","MW1","Cost1","MW2","Cost2","MW3","Cost3","Pmax","No_Load_Cost"]].round(2)
    return (data,)


@app.cell
def _(data, genname_ui, mo):
    _rows = [f"<td>{data[x].values[0]}</td>" for x in data.columns]
    _hdrs = [f"<th>{x}</th>" for x in data.columns]
    mo.md(f"<table><caption>{genname_ui.value}</caption><tr>{''.join(_hdrs)}</tr><tr>{''.join(_rows)}</tr></table>")
    return


@app.cell
def _(data, np):
    price = np.array(
        [
            [data[f"MW{n+1}"].values[0], data[f"Cost{n+1}"].values[0]]
            for n in range(3)
            if n == 0 or data[f"Cost{n+1}"].values[0] > 0
        ]
    ).T
    return (price,)


@app.cell
def _(data, np, price):
    _x = []
    _y = []
    for n in range(len(price[0])):
        _q0,_q1,_p = [price[0][n-1] if n > 0 else 0,price[0][n],price[1][n]]
        _x.append(np.arange(_q0,_q1,1).round(0))
        _y.append(_x[-1]*_p)
    y = np.cumsum(np.hstack(_y)) + data.No_Load_Cost.values[0]
    x = np.hstack(_x)
    fit = [np.polyfit(x, y, n) for n in range(9)]
    e = {len(p)-1:np.sqrt(np.linalg.norm(np.polyval(p, x) - y, 2)) for p in fit}
    return e, fit, n, x, y


@app.cell
def _(fit, mo, order_ui, re):
    _p = fit[order_ui.value]
    _t = ' '.join([f"{x:+.3g}~p^{{{len(_p)-n-1}}}" for n,x in enumerate(_p)])
    _t = re.sub("e([+-][0-9]+)",r"\\times10^{\1}",_t).replace("{+0","{").replace("{-0","-{").replace("p^{0}","").replace("p^{1}","p")
    mo.md(f"Fit order {order_ui.value}: $C(p) = {_t}$")

    return


@app.cell
def _(data, fit, np, order_ui):
    pmin,pmax = data.Pmin.values[0],data.Pmax.values[0]
    p = np.polynomial.Polynomial(fit[order_ui.value][-1::-1],symbol='p')
    p1 = p.deriv()
    p2 = p1.deriv()
    prr = [x for x in p.roots() if isinstance(x,float) and pmin<=x<=pmax]
    p1rr = [x for x in p1.roots() if isinstance(x,float) and pmin<=x<=pmax]
    p2rr = [x for x in p2.roots() if isinstance(x,float) and pmin<=x<=pmax]
    # prr,p1rr,p2rr
    return p, p1, p1rr, p2, p2rr, pmax, pmin, prr


@app.cell
def _(
    busname_ui,
    e,
    fit,
    genname_ui,
    np,
    order_ui,
    p1rr,
    p2rr,
    plt,
    prr,
    x,
    y,
):
    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    plt.grid()
    plt.xlabel("Fit order")
    plt.ylabel("RMSE")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost Fit Errors")
    plt.plot(e.keys(),e.values())
    plt.plot(order_ui.value,e[order_ui.value],'o')

    plt.subplot(1,2,2)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($M/h)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost")
    plt.plot(x,y/1e6,label="Data")
    plt.plot(x,np.polyval(fit[order_ui.value],x)/1e6,label=f"Fit order {order_ui.value}")
    if prr:
        plt.plot(prr,np.zeros(len(prr)),'ok',label='Fit zero')
    if p1rr:
        plt.plot(p1rr,np.zeros(len(p1rr)),'^k',label='Fit minimum')
    if p2rr:
        plt.plot(p1rr,np.zeros(len(p2rr)),'xk',label='Fit non-convexity')
    plt.legend()
    plt.gca()
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cvxpy as cp
    import re
    return cp, mo, np, pd, plt, re


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
