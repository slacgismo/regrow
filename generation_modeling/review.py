import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""This notebook is used to review the generation cost function fit output by the `gendata.py` script. The cost function fits are based on the price curves in the original NREL WECC240 model. The input data is provided as monotonic increasing prices for various power levels. If a non-monotonically increasing price is input, the decreasing price is relaxed so as to avoid non-convex cost curves.""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "More information (click here)":mo.md("""The `Inputs` tab displays the input generation data used to develop the cost function for each generator type at each WECC bus.

    The `Results` tab displays the fit terms for all the generation plant types at every bus of the WECC 240 model. The terms $a$, $b$, and $c$ refer to the second-order, first-order, and constant terms of the cost function fit, respectively.
    
    The `Review` tab is used to review the individual cost curve fit for each generator type at each bus in the WECC 240 model. The generation data and the cost function fit are shown. If a relaxation is performance it is noted. The left-hand plot shows the original price curve and the right-hand plot shows the cost data and the cost function fit.
    """)})
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
            "Inputs": gendata,
            "Results": mo.vstack(
                [
                    mo.hstack(
                        [mo.md("Show terms of $C(p)=ap^2+bp+c$:"), price_order_ui],
                        justify="start",
                    ),
                    price_plot_ui,
                ]
            ),
            "Review": mo.vstack(
                [plant_ui, data_ui, fit_ui, warning_ui,cost_plot_ui],heights='equal'
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
        options=_options, value=_options[0], label="Bus id:"
    )
    return (busname_ui,)


@app.cell
def _(busname_ui, costs, mo):
    # Create generator type dropdown
    _options = costs[costs.busname.astype(str) == busname_ui.value]
    _options = dict(zip(_options.Gen_Type, _options.genname))
    genname_ui = mo.ui.radio(
        options=_options,
        value=list(_options)[0],
        label="Generator type:",
        inline=True
    )
    genname_options = {y:x for x,y in _options.items()}
    return genname_options, genname_ui


@app.cell
def _(pd):
    gisdata = pd.read_csv("wecc240_gis.csv",index_col=[0])
    counties = pd.read_csv("counties.csv",index_col=["geocode"])
    return counties, gisdata


@app.cell
def _(busname_ui, counties, gisdata, mo, utils):
    _businfo = gisdata.loc[int(busname_ui.value)]
    _geohash = utils.geohash(_businfo["Lat"],_businfo["Long"])
    _nearest = utils.nearest(_geohash,counties.index)
    county = counties.loc[_nearest]
    info_ui = mo.md(f"Substation: **{_businfo['Bus  Name']}** (**{county.county} {county.usps}**)")
    return county, info_ui


@app.cell
def _(mo):
    # Create model fit order slider
    order_ui = mo.ui.slider(start=0, stop=10, value=2, label="Fit order:",show_value=True,debounce=True)
    return (order_ui,)


@app.cell
def _(mo, warning):
    # Create warning UI
    warning_ui = mo.md(f"**<font color=red>WARNING**: {warning}</font>" if warning else "")
    return (warning_ui,)


@app.cell
def _(costs, genname_ui):
    # Get selected data
    data = costs[costs.genname==genname_ui.value].set_index("genname")[["Pmin","MW1","Cost1","MW2","Cost2","MW3","Cost3","MW4","Cost4","Pmax","No_Load_Cost"]].round(2).iloc[0]
    return (data,)


@app.cell
def _(busname_ui, genname_ui, info_ui, mo):
    # Show UI controls
    plant_ui = mo.hstack([busname_ui, info_ui, genname_ui])
    return (plant_ui,)


@app.cell
def _(busname_ui, data, genname_options, genname_ui, mo):
    # Create data table UI
    _rows = [f"<td>{data[x]}</td>" for x in data.index]
    _hdrs = [f"<th>{x}</th>" for x in data.index]
    data_ui = mo.md(f"<table><caption>Bus {busname_ui.value} {genname_options[genname_ui.value]}</caption><tr>{''.join(_hdrs)}</tr><tr>{''.join(_rows)}</tr></table>")
    return (data_ui,)


@app.cell
def _(fit, mo, order_ui):
    # Show cost curve polynomial
    # _p = fit[order_ui.value]
    match len(fit):
        case 0:
            _poly = "0.00"
        case 1:
            _poly = f"{fit[0]:,.2f}"
        case 2:
            _poly = f"{fit[0]:.2f}~p {fit[1]:+,.2f}"
        case 3:
            _poly = f"{fit[0]:.6f}~p^2 {fit[1]:+.2f}~p {fit[2]:+,.2f}"
        case 4:
            _poly = "(error)"
    fit_ui = mo.md(f"Fit order {order_ui.value}: $C(p) = {_poly}$")
    return (fit_ui,)


@app.cell
def _(
    county,
    fit,
    genname_options,
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
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    _q, _p = [list(x) for x in zip(*prices)]
    plt.step([0] + _q, [_p[0]] + _p)
    plt.grid()
    plt.xlim([0, _q[-1]])
    plt.ylim([0, max(_p[-1] * 1.1,1)])
    plt.xlabel("Power (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(
        f"{county.county} {county.usps} {genname_options[genname_ui.value]} Generation Prices"
    )

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.xlabel("Power (MW)")
    plt.ylabel("Cost ($/h)")
    plt.title(f"{county.county} {county.usps} {genname_options[genname_ui.value]} Generation Cost")
    plt.plot(x, y, ":b", linewidth=3,label="Data")
    _output = []
    plt.plot(
        x, np.polyval(fit, x), "-k", label=f"Fit order {order_ui.value}"
    )
    if p0rr:
        plt.plot(p0rr, np.polyval(fit, p0rr), "ok", label="Fit zero")
        _output.append(mo.md(f"**<font color=red>WARNING**: negative costs found"))
    if p1rr:
        plt.plot(p1rr, np.polyval(fit, p1rr), "^k", label="Fit extreme")
        _output.append(
            mo.md(f"**<font color=red>WARNING**: declining costs found")
        )
    if p2rr:
        plt.plot(p2rr, np.polyval(fit, p2rr), "xk", label="Fit inflexion")
        _output.append(
            mo.md(f"**<font color=red>WARNING**: non-convex costs found")
        )
    plt.xlim([0, x[-1]])
    plt.ylim([0, max(y[-1] * 1.1,1)])
    plt.legend()

    _output.insert(0, plt.gca())
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
    x, y, warning = costdata(prices, data.Pmax, no_load_cost=data.No_Load_Cost)
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
    import utils
    return costdata, mo, np, pd, plt, utils


if __name__ == "__main__":
    app.run()
