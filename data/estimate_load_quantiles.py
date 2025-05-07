import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        This notebook creates the load quantile estimates for 2019-2022 based on observed loads and weather in 2018.

        The total load is the sum of the baseload, heating, and cooling loads.  

        The baseload is weakly a function of weather, and strongly a function of hour of day, day of week, and day of year. Given the weakness of the weather component, and the fact that weather itself is strongly a function of hour of day and day of year, we will ignore weather in the baseload model and compute a strictly smooth multi-period quantile estimator for baseload.

        In contrast, the heating and cooling loads are strongly a function of weather and weakly a function of hour of day, day of week, and day of year. Thus, for the heating and cooling loads we will compute a smooth multi-period quantile estimate of the temperature response of heating and cooling loads, where the temperature response is the deviation from the median response described as a truncated linear function

        $\qquad P = \lfloor {\alpha T + \beta} \rfloor_0$

        where 

        * $P$ is the heating or cooling power
        * $T$ is the outdoor air temperature
        * $\alpha$ is the heating or cooling temperature response
        * $\beta$ is the power at T=0 (possibly negative if truncated)

        This function is found by fitting of the function

        $F(a,b,\tau;T)\qquad (a+b)|T-\tau|  (a-b)(T-\tau)$

        to the observed powers $P$. This is solved as follows:

        $\begin{array}{rll}
            \underset{a,b,\tau} \min & ||P - (a+b)|T-\tau| + (a-b)(T-\tau)||_2^2 & \textrm{least squares fit to power}
        \\
            \textrm{subject to} & constraints
        \\
        \end{array}$

        where the $constraints$ are

        * $a=0$ when cooling
        * $b=0$ when heating
        """
    )
    return


@app.cell
def _(pd):
    T = pd.read_csv(
        "geodata/counties/temperature.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True,
    )
    geocodes = T.columns
    dT = T.diff()
    dT.columns = [f"{x}_dT" for x in dT.columns]

    H = pd.read_csv(
        "geodata/counties/heating.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True,
    )
    dH = H.diff()
    dH.columns = [f"{x}_dH" for x in dH.columns]

    C = pd.read_csv(
        "geodata/counties/cooling.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True,
    )
    dC = C.diff()
    dC.columns = [f"{x}_dC" for x in dC.columns]
    data = pd.concat([dT,dH,dC],axis=1).dropna().round(3)
    return C, H, T, dC, dH, dT, data, geocodes


@app.cell
def _(C, H, T, geocodes, mo, np, pd):
    def solve(X, T, positive=True):
        result = {"rmse": np.inf}
        # a = cp.Variable(1)
        # for tau in np.arange(10 if positive else 5,20 if positive else 15,1):
        #     delta = T-tau
        #     if positive: # cooling mode
        #         model = a*cp.abs(delta) - a*(delta)
        #     else: # heating mode
        #         model = -a*cp.abs(delta) + a*(delta)
        #     prob = cp.Problem(cp.Minimize(cp.sum_squares(X-model)))
        #     this = prob.solve()
        #     if this < result["rmse"]:
        #         result = {"rmse":this,"Tref":round(tau,1),"a":a.value[0]}
        delta = np.array([(t,x) for x,t in zip(X,T) if x>0]).transpose()
        # print(delta)
        if len(delta) > 0:
            res = np.polyfit(delta[0], delta[1],1,full=True)
            p = res[0]
            return {"rmse": res[1][0]**0.5/len(delta[0]), "Tref": -p[1]*p[0], "a": p[0]}
        else:
            return {"rmse": float('nan'), "Tref":15, "a": 0.0}


    fit = {}

    with mo.status.progress_bar(
        collection=geocodes, title="Fitting thermal models", remove_on_exit=True
    ) as bar:
        for _geocode in geocodes:
            _data = pd.concat([x[_geocode] for x in [T, H, C]], axis=1).dropna()
            _data.columns = ["T", "H", "C"]
            _T, _H, _C = [_data[x].values for x in "THC"]

            fit[_geocode] = {
                "heating": solve(_H, _T, False),
                "cooling": solve(_C, _T, True),
            }
            fit[_geocode]
            bar.update()
            # break
    return bar, fit, solve


@app.cell
def _(counties, fit, geocodes, pd):
    results = counties[["usps","county"]].join(pd.DataFrame(
        {
            "geocode":geocodes,
            "heating_sens": [fit[x]["heating"]["a"] for x in geocodes],
            "heating_temp": [fit[x]["heating"]["Tref"] for x in geocodes],
            "heating_rmse": [fit[x]["heating"]["rmse"] for x in geocodes],
            "cooling_sens": [fit[x]["cooling"]["a"] for x in geocodes],
            "cooling_temp": [fit[x]["cooling"]["Tref"] for x in geocodes],
            "cooling_rmse": [fit[x]["cooling"]["rmse"] for x in geocodes],
            },
    ).set_index("geocode"))
    results
    return (results,)


@app.cell
def _(results):
    results.to_csv("thermal_response.csv")
    return


@app.cell
def _(geocodes, pd):
    counties = pd.read_csv("counties.csv")
    counties.drop(counties[~counties["geocode"].isin(geocodes)].index,inplace=True)
    counties.set_index("geocode",inplace=True)
    return (counties,)


@app.cell
def _(counties, mo):
    _options = {f"{y.county} {y.usps} ({x})":x for x,y in counties.iterrows()}
    geocode_ui = mo.ui.dropdown(label="Geocode:",options=_options,value=list(_options)[0])
    geocode_ui
    return (geocode_ui,)


@app.cell
def _(C, H, T, geocode_ui, plt, results):
    _geocode = geocode_ui.value
    print(_geocode)
    print(results.loc[_geocode])
    _data = H.join(C,lsuffix="_H",rsuffix="_C")
    _data = _data.join(T).dropna()
    Th = results.loc[_geocode]["heating_temp"]
    Tc = results.loc[_geocode]["cooling_temp"]
    Hs = results.loc[_geocode]["heating_sens"]
    Cs = results.loc[_geocode]["cooling_sens"]
    # print(Th,Tc,Hs,Cs)
    Hx = [T[_geocode].min(),Th]
    Hy = [(T[_geocode].min()-Th)*Hs,0]
    Cx = [Tc,T[_geocode].max()]
    Cy = [0,(T[_geocode].max()-Tc)*Cs]

    plt.figure(figsize=(20,10))
    plt.plot(_data[_geocode],_data[_geocode+"_H"],".r")
    plt.plot(_data[_geocode],_data[_geocode+"_C"],".b")
    plt.plot(Hx,Hy,"-k")
    plt.plot(Cx,Cy,"-k")
    # plt.ylim([0,max(H.max().max(),C.max().max())])
    plt.grid()
    plt.gca()
    return Cs, Cx, Cy, Hs, Hx, Hy, Tc, Th


@app.cell
def _(mo):
    week_ui = mo.ui.slider(label="Week:",start=0,stop=52,value=0)
    week_ui
    return (week_ui,)


@app.cell
def _():
    # _dT = data[geocode_ui.value + "_dT"].copy()
    # _dT.loc[_dT==0] = float('nan')
    # _dH = data[geocode_ui.value + "_dH"]
    # _dC = data[geocode_ui.value + "_dC"]
    # dHdT = (_dH / _dT).dropna()
    # dCdT = (_dC / _dT).dropna()

    # _spq = qe.SmoothPeriodicQuantiles(
    #     num_harmonics=3,
    #     periods=[365.24 * 24, 7 * 24, 24],
    #     weight=0.1,
    #     quantiles=[0.02,0.5, 0.98],
    # )
    # _spq.fit(dHdT)
    # qH = _spq.fit_quantiles[:,:]
    # _spq.fit(dCdT)
    # qC = _spq.fit_quantiles[:,:]
    return


@app.cell
def _():
    # _start = max(week_ui.value * 24*7,0)
    # _stop = min(_start + 24*7,len(dHdT))
    # _index = np.s_[_start:_stop]
    # _lim = max(qH.max(),-qH.min(),qC.max(),-qC.min())

    # plt.figure(figsize=(15,8))
    # plt.plot(dHdT[_index],".r")
    # plt.plot(dCdT[_index],".b")
    # plt.plot(dHdT[_index].index,qH[_index,0],":r")
    # plt.plot(dHdT[_index].index,qH[_index,1],"-r")
    # plt.plot(dHdT[_index].index,qH[_index,2],":r")
    # plt.plot(dCdT[_index].index,qC[_index,0],":b")
    # plt.plot(dCdT[_index].index,qC[_index,1],"-b")
    # plt.plot(dCdT[_index].index,qC[_index,2],":b")
    # plt.xticks(dHdT[_index].index[::24])
    # plt.grid()
    # plt.ylim([-_lim,_lim])
    # plt.ylabel("Sensitivity [MW/$\circ$C]")
    # plt.title(geocode_ui.value)
    # plt.gca()
    return


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import spcqe as qe
    import cvxpy as cp
    return cp, mo, np, os, pd, plt, qe, sys


if __name__ == "__main__":
    app.run()
