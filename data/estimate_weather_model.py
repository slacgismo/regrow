import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""This notebook reads the county-level geodata for baseloads, heating, and cooling loads from the files in `geodata/counties` and creates a weather model to estimate total load based on weather for any year.""")
    return


@app.cell(hide_code=True)
def _(pd):
    temperature = pd.read_csv(
        "geodata/counties/temperature.csv",
        index_col=["timestamp"],
        parse_dates=True,
        low_memory=True,
    )
    geocodes = temperature.columns
    loads = {
        "baseload": pd.read_csv("geodata/counties/baseload_data.csv",index_col=["timestamp"],parse_dates=["timestamp"]),
        "heating": pd.read_csv("geodata/counties/heating.csv",index_col=["timestamp"],parse_dates=["timestamp"]),
        "cooling": pd.read_csv("geodata/counties/cooling.csv",index_col=["timestamp"],parse_dates=["timestamp"]),
    }
    return geocodes, loads, temperature


@app.cell
def _(tempe):
    tempe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The building weather response model is developed as follows:

        $\begin{array}{rll}
            \underset{\alpha,\beta,\gamma,u,c}\min & \alpha Q + \beta H + \gamma C & \textrm{minimize total energy use}
        \\
            \textrm{subject to} & u (T_S-T) + \alpha Q + \beta H - \gamma C + c \Delta T = 0 & \textrm{building energy balance}
        \\
            & u \ge 0 &  \textrm{heat transfer coefficient must be positive}
        \\
            & 0 \le \alpha \le 1 & \textrm{fractional heatgain between 0 and 1}
        \\
            & 0 \le \beta \le \gamma & \textrm{heating efficiency constraint}
        \\
            & 0 \le \gamma \le 1 & \textrm{cooling efficiency constraint}
        \\
            & c \ge 0 &  \textrm{heat capacity must be positive}
        \end{array}$

        where

        * $u \in \mathbb{R}$ is the lumped estimate heat transfer coefficient of the building population
        * $T_S \in \mathbb{R}$ is the lumped indoor air temperature
        * $T \in \mathbb{R}^N$ is the $N$ time-varying outdoor air temperatures
        * $\alpha \in \mathbb{R}$ is the fraction of baseload that goes to indoor air heat gain
        * $Q \in \mathbb{R}^N$ is the baseload powers
        * $\beta \in \mathbb{R}$ is the lumped heating efficiency
        * $H \in \mathbb{R}^N$ is the heating energy demand
        * $\gamma \in \mathbb{R}$ is the lumped cooling efficiency
        * $C \in \mathbb{R}^N$ is the cooling energy demand
        * $c \in \mathbb{R}$ is the lumped heat capacity of the buildings
        * $\Delta T \in \mathbb{R}$ is the change in building indoor temperature resulting from curtailment, if any
        """
    )
    return


@app.cell
def _(county_ui, cp, data, mo, np, pd, temperature):
    alpha = cp.Variable(1)
    beta = cp.Variable(1)
    gamma = cp.Variable(1)
    u = cp.Variable(1)
    c = cp.Variable(1)
    _T = temperature[county_ui.value].resample("1d").mean()
    _data = data.join(pd.DataFrame(_T.values,_T.index,columns=["temperature"])).dropna()
    Q = np.array(_data["baseload"].values)
    H = np.array(_data["heating"].values)
    C = np.array(_data["cooling"].values)
    T = np.array(_data["temperature"].values)
    Ts = 20.0
    objective = cp.Minimize(cp.sum(alpha*Q+beta*H+gamma*C))
    constraints = [
        u*(Ts-T)+alpha*Q+beta*H-gamma*C == 0,
        # u*(Ts-T)+alpha*Q+beta*H-gamma*C == 0,
        u >= 0,
        alpha >= 0.0, alpha <= 1,
        # beta >= 0, beta <= gamma, 
        # gamma >= 0, gamma <= 1,
        c >= 0,
    ]
    problem = cp.Problem(objective,constraints)
    cost = problem.solve()
    mo.md(f"$cost={cost:.3f}, \\alpha={alpha.value[0]:.3f}, \\beta={beta.value[0]:.3f}, \\gamma={gamma.value[0]:.3f}, u={u.value[0]:.3f}, c={c.value[0]:.3f}$" if problem.status != "infeasible" else "Infeasible")
    return (
        C,
        H,
        Q,
        T,
        Ts,
        alpha,
        beta,
        c,
        constraints,
        cost,
        gamma,
        objective,
        problem,
        u,
    )


@app.cell(hide_code=True)
def _(mo, pd):
    counties = pd.read_csv("counties.csv", index_col=["geocode"])
    state_ui = mo.ui.dropdown(label="State:",options=counties.usps.unique(),value=counties.usps.unique()[0])
    return counties, state_ui


@app.cell(hide_code=True)
def _(counties, mo, state_ui):
    _counties = counties[counties.usps == state_ui.value]
    _options = dict(zip(_counties.county, _counties.index))
    county_ui = mo.ui.dropdown(
        label="County:", options=_options, value=_counties.county.iloc[0]
    )
    return (county_ui,)


@app.cell(hide_code=True)
def _(mo):
    fraction_ui = mo.ui.checkbox(label="Fractional load")
    return (fraction_ui,)


@app.cell(hide_code=True)
def _(county_ui, fraction_ui, mo, state_ui):
    mo.hstack([state_ui,county_ui,fraction_ui],justify='start')
    return


@app.cell(hide_code=True)
def _(county_ui, fraction_ui, geocodes, loads, mo, pd, state_ui):
    mo.stop(
        not county_ui.value in geocodes,
        f"Data for {county_ui.selected_key} {state_ui.value} ({county_ui.value}) not in data",
    )
    data = [
        pd.DataFrame(
            data=loads[x][county_ui.value].values,
            index=loads[x][county_ui.value].index,
            columns=[x],
        )
        for x in ["baseload", "cooling", "heating"]
    ]
    data = pd.DataFrame(pd.concat(data, axis=1).resample(rule="1d").mean())
    if fraction_ui.value:
        _total = data.sum(axis=1,skipna=True)
        for _column in _data.columns:
            data[_column] /= _total/100

    data.plot.area(
        figsize=(15, 8),
        grid=True,
        color=["g", "b", "r"],
        xlabel="Date",
        ylabel="Daily average load [% total]" if fraction_ui.value else "Daily average load [MW]",
        title=f"{state_ui.value} {county_ui.selected_key} ({county_ui.value})",
        ylim=[0,100] if fraction_ui.value else None,
    )
    return (data,)


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
