import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. rapid change in demand
    2. regulatory incentives in adequacy
    3. investiments in capacity
    4. resource adquacy
    5. regulatory incentives for efficiency
    6. electricity affordability
    7. investments in efficiency
    8. reliability and resilience
    9. grid outage risk
    10. regulatory incentives for reliability and resilience
    """)
    return


@app.cell
def _(np):
    x = np.array([.5] * 10)
    return (x,)


@app.cell
def _(np):
    A = np.array([
        [ 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # flows into (1) "rapid change..."
        [ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0], # flows into (2)
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # flows into (3)
        [-1, 0, 1, 0, 0, 0, 1, 0, 0, 0], # flows into (4)
        [ 0, 0, 0, 0, 0,-1, 0, 0, 0, 0], # flows into (5)
        [ 0, 0,-1, 0, 0, 0, 1,-1, 0, 0], # flows into (6)
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # flows into (7)
        [ 0, 0, 0, 0,-1, 0, 0, 0, 0, 1], # flows into (8)
        [ 0, 0, 0,-1, 0, 0,-1,-1, 0, 0], # flows into (9)
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # flows into (10)
    ], dtype=float)
    A[A > 0] = .5 * A[A > 0]
    A[A < 0] = .5* A[A < 0]
    return (A,)


@app.cell
def _(A):
    A
    return


@app.cell
def _(A, np):
    eig_result = np.linalg.eig(A)
    svd_result = np.linalg.svd(A)
    return eig_result, svd_result


@app.cell
def _(eig_result, np, plt):
    plt.plot(np.abs(eig_result.eigenvalues))
    return


@app.cell
def _(A, np):
    np.linalg.trace(A)
    return


@app.cell
def _(eig_result, np, plt):
    plt.plot(np.real(eig_result.eigenvalues))
    return


@app.cell
def _(A, np, pd, x):
    trace = pd.DataFrame(columns=[f'{_i+1}' for _i in range(10)])
    _x_current = np.copy(x)
    _x_next = np.copy(x)
    for _ix in range(25):
        trace.loc[_ix] = _x_current
        _x_next = A @ _x_current
        _x_current = _x_next
    return (trace,)


@app.cell
def _(plt, trace):
    trace.plot(marker='.', markersize=4, linewidth=.75)
    plt.xlabel('number of simulation steps')
    plt.ylabel('metric value')
    return


@app.cell
def _(plt, svd_result):
    plt.stem(svd_result.S)
    plt.gcf()
    return


@app.cell
def _(svd_result):
    svd_result.Vh
    return


if __name__ == "__main__":
    app.run()
