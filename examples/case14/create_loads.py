import marimo

__generated_with = "0.3.3"
app = marimo.App(width="full")


@app.cell
def __(mo, pd):
    #
    # Load SCADA data
    #
    _head = {}
    _meta = {}
    load = []
    for _year in [2019,2020,2021]:
        _head[_year] = pd.read_csv(f"{_year}.csv",nrows=3,usecols=[0])
        _meta[_year] = pd.read_csv(f"{_year}.csv",nrows=5,skiprows=3,header=1,index_col=0)
        _names = ["DATETIME"]
        _names.extend(list(_meta[_year].columns))
        load.append(pd.read_csv(f"{_year}.csv",
                                index_col = 0,
                                skiprows = 10,
                                names = _names,
                                parse_dates = [0],
                               ))
    load = pd.concat(load)
    for _column in load.columns:
        if _column not in ["WALNT_GR","VIEW","N_CAMAS","119TH_ST","ARCHERCC","BARBERTN","BATLGRND","CAPLES","CARBUNDM.1",
                          "ELMGREN","FOURTH_P","GRAND","HEYE_MYR","HOCKNSON","AST","RUNYAN","VAN_SHIP.1","CASCD_PK"]:
            load.drop(_column,axis=1,inplace=True)
    load = load.sum(axis=1)
    load = pd.DataFrame(load/load.mean()*10,columns=["MW"])
    load[load<=(load.mean()-3.5*load.std())] = float('nan')
    load.dropna(inplace=True)
    load.index.name = 'datetime'

    mo.vstack([
        mo.md("# Original SCADA data"),
        load.plot(figsize=(15,5),grid=True,xlabel='Date/Time',ylabel='Power [MW]')
    ])

    return load,


@app.cell
def __(load):
    #
    # Generate load player data
    #
    player = load.copy()
    player.columns = ['P']
    player.P = player.P.round(3)
    player.to_csv("pp_bus_2.csv",index=True,header=True)
    return player,


@app.cell
def __(pd):
    #
    # Load weather
    #
    weather = pd.read_csv("weather.csv",
                          usecols=["datetime","solar_horizontal[W/sf]","temperature[degF]","heat_index[degF]"],
                          index_col="datetime",
                          parse_dates=True,
                         )
    weather.columns = ["solar_horizontal","temperature","heat_index"]
    return weather,


@app.cell
def __(load, weather):
    #
    # Join load and weather data
    #
    data = load.join(weather).dropna()
    return data,


@app.cell
def __(data, mo):
    #
    # Plot raw data
    #
    _fig1 = data.plot(kind='scatter',x='heat_index',y='MW',figsize=(15,5),grid=True,xlabel='Heat index [$^o$F]',ylabel='Power [MW]')
    mo.vstack([
        mo.md("# Training data"),
        _fig1,
    ])
    return


@app.cell
def __(mo):
    order_ui = mo.ui.slider(start=1,stop=50,value=2,label="Model order")
    holdout_ui = mo.ui.slider(start=1,stop=25,value=5,label="Validation holdout")
    mo.hstack([order_ui,holdout_ui],justify='start')
    return holdout_ui, order_ui


@app.cell
def __(data, holdout_ui, mo, np, order_ui):
    #
    # Create dynamic load model
    #
    holdout = int(len(data)*(100-holdout_ui.value)/100.0) # cut-off for holdout data
    K = order_ui.value # model order (>0)
    _train = data.iloc[0:holdout-1]
    _X = np.matrix(_train['heat_index']).transpose()
    _Y = np.matrix(_train['MW']).transpose()
    _L = len(_Y)
    _M = np.hstack([np.hstack([_Y[n+1:_L-K+n] for n in range(K)]),
                   np.hstack([_X[n+1:_L-K+n] for n in range(K+1)]),
                  ])
    _Mt = _M.transpose()
    model = np.linalg.solve(_Mt*_M,_Mt*_Y[K+1:])
    _a = model.transpose().round(4).tolist()[0][0:K]
    _b = model.transpose().round(4).tolist()[0][K:]
    _A = "".join([f"{x:+.4g}z^{{-{n+1}}}" for n,x in enumerate(_a)])
    _B = f"{_b[0]:+.4g}" + "".join([f"{x:+.4g}z^{{-{n+1}}}" for n,x in enumerate(_b[1:])])
    mo.vstack([
        mo.md(f"Transfer function of order {K} in $z$-domain: $\\frac{{P(z)}}{{T(z)}} = \\frac{{{_B}}}{{1{_A}}}$")
    ])
    return K, holdout, model


@app.cell
def __(K, data, holdout, mo, model, np, plt):
    #
    # Model validation
    #
    _test = data.iloc[holdout:]
    _X = np.matrix(_test['heat_index']).transpose()
    _Y = np.matrix(_test['MW']).transpose()
    _L = len(_Y)
    _M = np.hstack([np.hstack([_Y[n+1:_L-K+n] for n in range(K)]),
                   np.hstack([_X[n+1:_L-K+n] for n in range(K+1)])])

    plt.figure(figsize=(15,5))
    plt.plot(_test.index[K+1:],_Y[K+1:],label='Measured power')
    plt.plot(_test.index[K+1:],_M@model,label='Modeled power')
    plt.plot(_test.index[K+1:],_M@model-_Y[K+1:],label='Model error')
    plt.grid()
    plt.xlabel('Date/Time')
    plt.ylabel('Power [MW]')
    plt.legend()
    _fig1 = plt.gca()

    plt.figure(figsize=(15,10))
    plt.plot(_X[K+1:],_Y[K+1:],label='Measured power')
    plt.plot(_X[K+1:],_M@model,label='Modeled power')
    plt.grid()
    plt.xlabel('Heat index [$^o$F]')
    plt.ylabel('Power [MW]')
    plt.legend()
    _fig2 = plt.gca()

    _err = _M@model - _Y[K+1:]
    _rmse = np.sqrt((_err.transpose()@_err/len(_err))[0,0])

    mo.vstack([
        mo.md(f"""# Validation on {(1-holdout/len(data))*100:.0f}% hold-out data
            RMSE = {_rmse:.2f} MW ({_rmse/_Y.mean()*100:.1f}%)
        """),
        _fig1,
        _fig2,
    ])
    return


@app.cell
def __():
    #
    # Requirements and settings
    #
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    pd.options.display.max_rows = 10
    return mo, np, pd, plt


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
