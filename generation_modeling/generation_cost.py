import marimo

__generated_with = "0.16.5"
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
def _(mo):
    constraint_ui = mo.ui.checkbox(label="Constrained fit")
    return (constraint_ui,)


@app.cell
def _(mo):
    withnlc_ui = mo.ui.checkbox(label="Include standby cost")
    return (withnlc_ui,)


@app.cell
def _(busname_ui, constraint_ui, mo, order_ui, withnlc_ui):
    mo.hstack([busname_ui, order_ui,constraint_ui,withnlc_ui])
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
def _(constraint_ui, cp, data, np, price, withnlc_ui):
    _x = []
    _y = []
    for n in range(len(price[0])):
        _q0, _q1, _p = [price[0][n - 1] if n > 0 else 0, price[0][n], price[1][n]]
        _x.append(np.arange(_q0, _q1, 1).round(0))
        _y.append(np.ones(len(_x[-1])) * _p)
    y = np.cumsum(np.hstack(_y)) + (
        data.No_Load_Cost.values[0] if withnlc_ui.value else 0
    )
    x = np.hstack(_x)
    if constraint_ui.value:
        fit = [np.polyfit(x, y, 0)]
        A = np.ones((len(y),1))
        for _n in range(1, 9):
            A = np.hstack([A,np.array([A[:,_n-1]*x]).T])
            p = cp.Variable(_n+1)
            prob = cp.Problem(
                cp.Minimize(cp.sum_squares(A@p - y)), 
                [cp.diff(A@p,k=2) >= 0]
            )
            try:
                prob.solve(solver="Clarabel")
            except Exception as err:
                print("order:",_n,"--> exception",err)
                pass
            if p.value is None:
                print("order:",_n,"-->",prob.status)
                print("order:",_n,"-->",prob.status)
            fit.append(p.value[-1::-1] if p.value is not None else [])
    else:
        fit = [np.polyfit(x, y, n) for n in range(9)]
    e = {n: round(float(np.sqrt(np.linalg.norm(np.polyval(p, x) - y, 2))),1) for n,p in enumerate(fit) if len(p) > 0 }

    return e, fit, x, y


@app.cell
def _(fit, mo, order_ui, re):
    _p = fit[order_ui.value]
    _t = ' '.join([f"{x:+.3g}~p^{{{len(_p)-n-1}}}" for n,x in enumerate(_p)])
    _t = re.sub("e([+-][0-9]+)",r"\\times10^{\1}",_t).replace("{+0","{").replace("{-0","-{").replace("p^{0}","").replace("p^{1}","p")
    mo.md(f"Fit order {order_ui.value}: $C(p) = {_t}$")
    return


@app.cell
def _(data, fit, np, order_ui):
    pmin, pmax = data.Pmin.values[0], data.Pmax.values[0]
    if len(fit[order_ui.value]) > 0:
        p0 = np.polynomial.Polynomial(fit[order_ui.value][-1::-1], symbol="p")
        p1 = p0.deriv()
        p2 = p1.deriv()
    else:
        p0 = p1 = p2 = []
    p0rr = [x for x in p0.roots() if isinstance(x, float) and pmin <= x <= pmax] if len(p0)>0 else []
    p1rr = [x for x in p1.roots() if isinstance(x, float) and pmin <= x <= pmax] if len(p1)>0 else []
    p2rr = [x for x in p2.roots() if isinstance(x, float) and pmin <= x <= pmax] if len(p2)>0 else []
    return p0rr, p1rr, p2rr


@app.cell
def _(
    busname_ui,
    e,
    fit,
    genname_ui,
    np,
    order_ui,
    p0rr,
    p1rr,
    p2rr,
    plt,
    price,
    x,
    y,
):
    plt.figure(figsize=(20,6))

    plt.subplot(1,3,1)
    plt.step([0]+price[0].tolist(),[price[1].tolist()[0]] + price[1].tolist())
    plt.grid()
    plt.xlim([0,price[0][-1]+10])
    plt.ylim([0,price[1][-1]+10])
    plt.xlabel("Power (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Prices")

    plt.subplot(1,3,2)
    plt.grid()
    plt.xlabel("Fit order")
    plt.ylabel("RMSE")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost Fit Errors")
    plt.plot(e.keys(),e.values())
    if order_ui.value in e:
        plt.plot(order_ui.value,e[order_ui.value],'o')

    plt.subplot(1,3,3)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($/h)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost")
    plt.plot(x,y,label="Data")
    if order_ui.value in e:
        plt.plot(x,np.polyval(fit[order_ui.value],x),label=f"Fit order {order_ui.value}")
        if p0rr:
            plt.plot(p0rr,np.polyval(fit[order_ui.value],p0rr),'ok',label='Fit zero')
        if p1rr:
            plt.plot(p1rr,np.polyval(fit[order_ui.value],p1rr),'^k',label='Fit extreme')
        if p2rr:
            plt.plot(p2rr,np.polyval(fit[order_ui.value],p2rr),'xk',label='Fit inflexion')
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


if __name__ == "__main__":
    app.run()
