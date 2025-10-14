import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""This notebook is used to review the generation cost data in the original NREL WECC240 model. The cost data is provided as piece-wise linear""")
    return


@app.cell
def _(
    cost_plot_ui,
    cost_ui,
    data_ui,
    fit_ui,
    gendata,
    mo,
    price_order_ui,
    price_plot_ui,
    price_ui,
):
    mo.ui.tabs({
        "Cost data" : mo.vstack([cost_ui,data_ui,price_ui,fit_ui,cost_plot_ui]),
        "Gen data" : gendata,
        "Price data" : mo.vstack([price_order_ui,price_plot_ui]),
    },lazy=True)
    return


@app.cell
def _(pd):
    # Load generation data
    costs = pd.read_csv("generation_data.csv")
    gencost = pd.read_csv("gencost.csv")
    gen = pd.read_csv("gen.csv")
    gendata = pd.concat([gen,gencost],axis=1)
    return costs, gendata


@app.cell
def _(mo):
    price_order_ui = mo.ui.radio(options=["Standby cost ($k/h)","Energy price ($/MWh)","Scarcity rent ($/kW².h)"],inline=True,value="Energy price ($/MWh)")
    return (price_order_ui,)


@app.cell
def _(costs, gendata, np, plt, price_order_ui):
    # Create price data plots
    plt.figure(figsize=(16,10))
    for _type in costs.Gen_Type.unique():
        _n = costs[costs.Gen_Type==_type].index
        _data = gendata.iloc[_n]
        _costs = list(zip(_data.NCOST,_data.COST0,_data.COST1,_data.COST2,_data.PMAX))
        match price_order_ui.value:
            case "Standby cost ($k/h)":
                qp = np.array([(x[4],x[x[0]]/1000) for x in _costs]).T
            case "Energy price ($/MWh)":
                qp = np.array([(x[4],0 if x[0] == 1 else x[x[0]-1]) for x in _costs]).T
            case "Scarcity rent ($/kW².h)":
                qp = np.array([(x[4],0 if x[0] <= 2 else x[x[0]-2]*1e6) for x in _costs]).T
        plt.scatter(qp[0],qp[1],label=_type)
    plt.grid()
    plt.xlabel("Capacity (MW)")
    plt.ylabel(price_order_ui.value)
    plt.legend()
    price_plot_ui = plt.gca()
    return (price_plot_ui,)


@app.cell
def _(costs, mo):
    # Create bus name dropdown
    _options = [str(x["busname"]) for n,x in costs.iterrows()]
    busname_ui = mo.ui.dropdown(
        options=_options, value=_options[0], label="Bus name:"
    )
    return (busname_ui,)


@app.cell
def _(busname_ui, costs, mo):
    # Create generator type dropdown
    _options = costs[costs.busname.astype(str) == busname_ui.value]
    _options = dict(zip(_options.Gen_Type, _options.genname))
    genname_ui = mo.ui.dropdown(
        options=_options,
        value=list(_options)[0],
        label="Generator type:",
    )
    return (genname_ui,)


@app.cell
def _(mo):
    # Create model fit order slider
    order_ui = mo.ui.slider(start=0, stop=10, value=2, label="Fit order:",show_value=True,debounce=True)
    return (order_ui,)


@app.cell
def _(mo):
    # Create constrained fit switch
    constraint_ui = mo.ui.switch(label="Constrained fit")
    return (constraint_ui,)


@app.cell
def _(mo):
    # Create standby cost w
    withnlc_ui = mo.ui.switch(label="Include standby cost")
    return (withnlc_ui,)


@app.cell
def _(costs, genname_ui):
    # Get selected data
    data = costs[costs.genname==genname_ui.value].set_index("genname")[["Pmin","MW1","Cost1","MW2","Cost2","MW3","Cost3","Pmax","No_Load_Cost"]].round(2)
    return (data,)


@app.cell
def _(constraint_ui, cp, data, np, price, withnlc_ui):
    # Construct cost curve and fit polynomial
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
        fit = np.round([np.polyfit(x, y, 0)],6)
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
        fit = [np.round(np.polyfit(x, y, n),6) for n in range(9)]
    e = {n: round(float(np.sqrt(np.linalg.norm(np.polyval(p, x) - y, 2))),1) for n,p in enumerate(fit) if len(p) > 0 }
    return e, fit, x, y


@app.cell
def _(data, fit, np, order_ui):
    # Identify polynonial critical points, if any
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
def _(busname_ui, genname_ui, mo, withnlc_ui):
    # Show UI controls
    cost_ui = mo.hstack([busname_ui, genname_ui, withnlc_ui])
    return (cost_ui,)


@app.cell
def _(busname_ui, data, genname_ui, mo):
    # Show selected data
    _rows = [f"<td>{data[x].values[0]}</td>" for x in data.columns]
    _hdrs = [f"<th>{x}</th>" for x in data.columns]
    data_ui = mo.md(f"<table><caption>Bus {busname_ui.value} {genname_ui.selected_key}</caption><tr>{''.join(_hdrs)}</tr><tr>{''.join(_rows)}</tr></table>")
    return (data_ui,)


@app.cell
def _(data, mo, np):
    # Construct price curve
    price = np.array(
        [
            [data[f"MW{n+1}"].values[0], data[f"Cost{n+1}"].values[0]]
            for n in range(3)
            if n == 0 or data[f"Cost{n+1}"].values[0] > 0
        ]
    ).T
    if price[0][-1] < data.Pmax.values[0]:
        price_ui = mo.md(f"**<font color=red>WARNING**: non-convex prices from {price[0][-1]:.1f} to {data['Pmax'].values[0]:.1f} MW relaxed from $0.00/MWh to ${price[1][-1]:.2f}/MWh</font>")
        price[0][-1] = data.Pmax.values[0]
    else:
        price_ui = mo.md("")
    return price, price_ui


@app.cell
def _(fit, mo, order_ui, re):
    # Show cost curve polynomial
    _p = fit[order_ui.value]
    _t = ' '.join([f"{x:+.3g}~p^{{{len(_p)-n-1}}}" for n,x in enumerate(_p) if x!=0])
    _t = re.sub("e([+-][0-9]+)",r"\\times10^{\1}",_t).replace("{+0","{").replace("{-0","-{").replace("p^{0}","").replace("p^{1}","p")
    fit_ui = mo.md(f"Fit order {order_ui.value}: $C(p) = {_t if _t else "0.00"}$")
    return (fit_ui,)


@app.cell
def _(
    busname_ui,
    e,
    fit,
    genname_ui,
    mo,
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
    # Show plots and warnings
    plt.figure(figsize=(20,6))

    plt.subplot(1,3,1)
    _q = price[0].tolist()
    _p = price[1].tolist()
    plt.step([0]+_q,[_p[0]]+_p)
    plt.grid()
    plt.xlim([0,_q[-1]])
    plt.ylim([0,_p[-1]*1.1])
    plt.xlabel("Power (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Prices")

    plt.subplot(1,3,2)
    plt.grid()
    plt.xlabel("Fit order")
    plt.ylabel("RMSE")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost Fit Errors")
    plt.bar(list(e.keys())[0:3],list(e.values())[0:3])
    if order_ui.value in e:
        plt.plot(order_ui.value,e[order_ui.value],'o')

    plt.subplot(1,3,3)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($/h)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost")
    plt.plot(x,y,label="Data")
    _output = []
    if order_ui.value in e:
        plt.plot(x,np.polyval(fit[order_ui.value],x),label=f"Fit order {order_ui.value}")
        if p0rr:
            plt.plot(p0rr,np.polyval(fit[order_ui.value],p0rr),'ok',label='Fit zero')
            _output.append(mo.md(f"**<font color=red>WARNING**: negative costs found"))
        if p1rr:
            plt.plot(p1rr,np.polyval(fit[order_ui.value],p1rr),'^k',label='Fit extreme')
            _output.append(mo.md(f"**<font color=red>WARNING**: declining costs found"))
        if p2rr:
            plt.plot(p2rr,np.polyval(fit[order_ui.value],p2rr),'xk',label='Fit inflexion')
            _output.append(mo.md(f"**<font color=red>WARNING**: non-convex costs found"))
    plt.legend()

    _output.insert(0,plt.gca())
    cost_plot_ui = mo.vstack(_output)
    return (cost_plot_ui,)


@app.cell
def _(cost_plot_ui):
    cost_plot_ui
    return


@app.cell
def _():
    # Load modules and setup
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cvxpy as cp
    import re
    return cp, mo, np, pd, plt, re


if __name__ == "__main__":
    app.run()
