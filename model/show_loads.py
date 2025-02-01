import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    loads = pd.read_csv("loads.csv",
        parse_dates=[0]
        ).sort_values(["datetime","geocode"])
    loads
    return loads, pd


@app.cell
def _(loads):
    import numpy as np
    import scipy.interpolate as interp
    import matplotlib.pyplot as plt
    _points = np.stack([loads.longitude.tolist(), loads.latitude.tolist()],-1)
    _values = np.array(loads["voltage[deg]"].tolist())
    _x0,_x1 = int(loads.longitude.min()),int(loads.longitude.max()+1)
    _y0,_y1 = int(loads.latitude.min()),int(loads.latitude.max()+1)
    _x,_y = np.mgrid[_x0:_x1:0.02,_y0:_y1:0.02]
    _z = interp.griddata(_points, _values, (_x, _y), method='cubic') 
    plt.imshow(_z,extent=[_x0,_x1,_y0,_y1],origin='lower')
    plt.grid()
    plt.show()
    return interp, np, plt


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
