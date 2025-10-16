import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""This notebook is used to review the generation cost data in the original NREL WECC240 model. The cost data is provided as piece-wise linear""")
    return


@app.cell
def _():
    #
    # UI Components
    #
    return


@app.cell
def _(
    cost_plot_ui,
    data_ui,
    fit_ui,
    gendata,
    mo,
    plant_ui,
    price_order_ui,
    price_plot_ui,
    warning_ui,
):
    # Show consolidated UI
    mo.ui.tabs(
        {
            "Cost data": mo.vstack(
                [plant_ui, data_ui, fit_ui, warning_ui,cost_plot_ui],heights='equal'
            ),
            "Gen data": gendata,
            "Price data": mo.vstack(
                [
                    mo.hstack(
                        [mo.md("Show terms of $C(p)=ap^2+bp+c$:"), price_order_ui],
                        justify="start",
                    ),
                    price_plot_ui,
                ]
            ),
        },
        lazy=True,
    )
    return


@app.cell
def _(mo):
    # Create price data radio button selector
    price_order_ui = mo.ui.radio(options=["a ($k/kW².h)","b ($/MWh)","c ($k/h)"],inline=True,value="b ($/MWh)")
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
            case "c ($k/h)":
                _qp = np.array([(x[4],x[x[0]]/1000) for x in _costs]).T
            case "b ($/MWh)":
                _qp = np.array([(x[4],0 if x[0] == 1 else x[x[0]-1]) for x in _costs]).T
            case "a ($k/kW².h)":
                _qp = np.array([(x[4],0 if x[0] <= 2 else x[x[0]-2]*1e3) for x in _costs]).T
        plt.scatter(_qp[0],_qp[1],label=_type)
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
def _(mo, warning):
    # Create warning UI
    warning_ui = mo.md(f"**<font color=red>WARNING**: {warning}</font>") if warning else None
    return (warning_ui,)


@app.cell
def _(costs, genname_ui):
    # Get selected data
    data = costs[costs.genname==genname_ui.value].set_index("genname")[["Pmin","MW1","Cost1","MW2","Cost2","MW3","Cost3","MW4","Cost4","Pmax","No_Load_Cost"]].round(2).iloc[0]
    return (data,)


@app.cell
def _(busname_ui, genname_ui, mo):
    # Show UI controls
    plant_ui = mo.hstack([busname_ui, genname_ui],justify='start')
    return (plant_ui,)


@app.cell
def _(busname_ui, data, genname_ui, mo):
    # Create data table UI
    _rows = [f"<td>{data[x]}</td>" for x in data.index]
    _hdrs = [f"<th>{x}</th>" for x in data.index]
    data_ui = mo.md(f"<table><caption>Bus {busname_ui.value} {genname_ui.selected_key}</caption><tr>{''.join(_hdrs)}</tr><tr>{''.join(_rows)}</tr></table>")
    return (data_ui,)


@app.cell
def _(fit, mo, order_ui, re):
    # Show cost curve polynomial
    # _p = fit[order_ui.value]
    _t = ' '.join([f"{x:+.3g}~p^{{{len(fit)-n-1}}}" for n,x in enumerate(fit) if x!=0])
    _t = re.sub("e([+-][0-9]+)",r"\\times10^{\1}",_t).replace("{+0","{").replace("{-0","-{").replace("p^{0}","").replace("p^{1}","p")
    fit_ui = mo.md(f"Fit order {order_ui.value}: $C(p) = {_t if _t else "0.00"}$")
    return (fit_ui,)


@app.cell
def _(
    busname_ui,
    fit,
    genname_ui,
    mo,
    np,
    order_ui,
    p0rr,
    p1rr,
    p2rr,
    plt,
    prices,
    x,
    y,
):
    # Show plots and warnings
    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    _q,_p = [list(x) for x in zip(*prices)]
    plt.step([0]+_q,[_p[0]]+_p)
    plt.grid()
    plt.xlim([0,_q[-1]])
    plt.ylim([0,_p[-1]*1.1])
    plt.xlabel("Power (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Prices")

    plt.subplot(1,2,2)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($/h)")
    plt.title(f"Bus {busname_ui.value} {genname_ui.selected_key} Generation Cost")
    plt.plot(x,y,label="Data")
    _output = []
    plt.plot(x,np.polyval(fit,x),label=f"Fit order {order_ui.value}")
    if p0rr:
        plt.plot(p0rr,np.polyval(fit,p0rr),'ok',label='Fit zero')
        _output.append(mo.md(f"**<font color=red>WARNING**: negative costs found"))
    if p1rr:
        plt.plot(p1rr,np.polyval(fit,p1rr),'^k',label='Fit extreme')
        _output.append(mo.md(f"**<font color=red>WARNING**: declining costs found"))
    if p2rr:
        plt.plot(p2rr,np.polyval(fit,p2rr),'xk',label='Fit inflexion')
        _output.append(mo.md(f"**<font color=red>WARNING**: non-convex costs found"))
    plt.legend()

    _output.insert(0,plt.gca())
    cost_plot_ui = mo.vstack(_output)
    return (cost_plot_ui,)


@app.cell
def _():
    #
    # Analysis
    #
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
def _(costs, gendata, genname_ui):
    # get fit
    _n = costs[costs.genname==genname_ui.value].number.values[0]
    _data = gendata.iloc[_n-1]
    fit = [_data[f"COST{n}"] for n in range(int(_data.NCOST))]
    return (fit,)


@app.cell
def _(costdata, data):
    # read price data and generate cost data
    prices = [
        [float(data[f"MW{n+1}"]), float(data[f"Cost{n+1}"])]
        for n in range(4)
        if n == 0
        or (data[f"Cost{n+1}"] > 0 and data[f"Cost{n}"] < data[f"Cost{n+1}"])
    ]
    x, y, warning = costdata(prices, data.Pmax, data.No_Load_Cost)
    return prices, warning, x, y


@app.cell
def _(data, fit, np):
    # Identify polynomial critical points, if any
    if len(fit) > 0:
        p0 = np.polynomial.Polynomial(fit[-1::-1], symbol="p")
        p1 = p0.deriv()
        p2 = p1.deriv()
    else:
        p0 = p1 = p2 = []
    p0rr = [x for x in p0.roots() if isinstance(x, float) and data.Pmin <= x <= data.Pmax] if len(p0)>0 else []
    p1rr = [x for x in p1.roots() if isinstance(x, float) and data.Pmin <= x <= data.Pmax] if len(p1)>0 else []
    p2rr = [x for x in p2.roots() if isinstance(x, float) and data.Pmin <= x <= data.Pmax] if len(p2)>0 else []
    return p0rr, p1rr, p2rr


@app.cell
def _():
    #
    # Notebook setup
    #
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
    from gendata import costdata
    return costdata, mo, np, pd, plt, re


if __name__ == "__main__":
    app.run()
