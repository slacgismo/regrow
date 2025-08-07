import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Chapter 3 - Optimal Power Flow

    We use the test cases from PyPOWER (see https://github.com/rwl/pypower) to provision the test model.
    """
    )
    return


@app.cell
def _(Model, mo, os, re):
    # baseline model UI elements
    _list = sorted(
        [x for x in os.listdir(".") if re.match("case.+.py",x)], # get only "case*.py"
        key=lambda x: int(re.sub("[^0-9]*([0-9]+)[^0-9]*", r"\1", x, 1)), # sort by numerical order not lexical
    )
    _options = {os.path.splitext(x)[0]: x for x in _list}
    model_ui = mo.ui.dropdown(
        options=_options, value=os.path.splitext(_list[0])[0]
    )
    verbose_ui = mo.ui.checkbox(
        label="Verbose output (code view only)", value=False
    )
    line_ui = mo.ui.dropdown(
        label="Line property:", options=Model.basecolumns["branch"].split()
    )
    node_ui = mo.ui.dropdown(
        label="Node property:", options=Model.basecolumns["bus"].split()
    )
    return line_ui, model_ui, node_ui, verbose_ui


@app.cell
def _(Model, line_ui, mo, model_ui, node_ui, os, pd, verbose_ui):
    # extract model data into named tuples
    model = Model(os.path.join(".",model_ui.value))
    with mo.capture_stdout() as stdout:
        with mo.capture_stderr() as stderr:
            model.solve_opf(VERBOSE=3,OUT_ALL=1 if verbose_ui.value else 0)
    bus, gen, branch, gencost, dcline, dclinecost = [
        model[x]
        for x in ["bus", "gen", "branch", "gencost", "dcline", "dclinecost"]
    ]
    N, M = len(bus.bus_i), len(branch.fbus)
    mo.accordion(
        {
            f"{model_ui}: PyPOWER OPF has {"succeeded" if model.result["success"] else "failed"}. The output of the solver is \"`{model.result["raw"]["output"]["message"]}`\". (Click here to view results.)": mo.vstack(
                [
                    mo.hstack([verbose_ui],justify='end'),
                    mo.ui.tabs(
                        {
                            "Overview": pd.DataFrame(
                                {
                                    "Bus": {
                                        "Rows": len(bus.bus_i),
                                        "Columns": len(bus),
                                    },
                                    "Branches": {
                                        "Rows": len(branch.fbus),
                                        "Columns": len(branch),
                                    },
                                    "Generation": {
                                        "Rows": len(gen.bus),
                                        "Columns": len(gen),
                                    },
                                    "Generation Costs": {
                                        "Rows": len(gencost.n),
                                        "Columns": len(gencost),
                                    },
                                    "DC Lines": {
                                        "Rows": len(dcline.fbus),
                                        "Columns": len(dcline),
                                    },
                                    "DC Line Costs": {
                                        "Rows": len(dclinecost.n),
                                        "Columns": len(dclinecost),
                                    },
                                }
                            ).T,
                            "Graph": mo.vstack([
                                mo.hstack([line_ui,node_ui],justify='start'),
                                mo.mermaid(model.graph(line=line_ui.value,node=node_ui.value))
                            ]),
                            "Busses": pd.DataFrame(data=bus._asdict()).round(3),
                            "Branches": pd.DataFrame(data=branch._asdict()).round(
                                3
                            ),
                            "Generation": pd.DataFrame(data=gen._asdict()).round(
                                3
                            ),
                            "Generation Costs": pd.DataFrame(
                                data=gencost._asdict()
                            ).round(3),
                            "DC Lines": pd.DataFrame(data=dcline._asdict()).round(
                                3
                            ),
                            "DC Line Costs": pd.DataFrame(
                                data=dclinecost._asdict()
                            ).round(3),
                            "Raw data": model.result,
                            "Output": mo.md(f"~~~\n{stdout.getvalue()}\n~~~"),
                            "Errors": mo.md(f"~~~\n{stderr.getvalue()}\n~~~"),
                        },
                        lazy=True,
                    ),
                ]
            )
        }
    )
    return N, branch, bus, model


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Coding Nomenclature

    The text uses the notation $x_{ij}$ to denote the value at the index $i,j$ in the matrix $x \in \mathbb{R}^{N \times N}$ and the notation $x_i$ to denote the value at the index $i$ in the vector $x 
    \in \mathbb{R}^N$.  To disambiguate cases where the variable $x$ denotes two different things depending on the dimensions of the variable, the code in this notebook will always use capitalized variables for 2-dimensional arrays and lowercase variable names to 1-dimensional arrays.  Thus $x_i$ will always be coded as `x[i]`, while $x_{i,j}$ will be coded as `X[i,j]`. Scalars will be coded as `x_<tag>` to indicate that they are not arrays. In cases where variables are denoted $x^Y_{i}$ or $x^Y_{i,j}$, these will be coded as `xY[i]` and `XY[i,j]`, respectively. 

    Additional variable encodings include the following:

    * $\underline x, \bar x, \hat x, \v x, \tilde x, \dot x, \ddot x \cdots \to$ `xmin`, `xmax`, `xhat`, `xvee`, `xtilde`, `xdot`,`xddot`, ...
    * $\alpha,\beta,\gamma,\cdots \to$ `alpha`, `beta`, `gamma`, ...
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3.1 - Basic Formulation

    The voltage $v_i$ at the node $i$ must satisfy the limits $\underline{v}_i \le |v_i| \le \bar{v}_i$.
    """
    )
    return


@app.cell
def _(bus, mo, model, model_ui, np, pd):
    # bus voltages and voltage limit checks
    v = bus.Vm * (np.cos(bus.Va * np.pi / 180) + np.sin(bus.Va * np.pi / 180) * 1j)
    vmin, vmax = bus.Vmin, bus.Vmax
    v_ok = (vmin <= abs(v)).all() and (abs(v) <= vmax)
    mo.accordion({
        f"{model_ui}: The bus voltages are {'ok' if v_ok.all() else 'not ok'} (click to view).": mo.ui.tabs({
            "Graph": mo.mermaid(model.graph(node=v_ok.tolist())),
            "Table": pd.DataFrame({
                "Node":range(1,len(v_ok)+1),
                "vmin": vmin.round(3),
                "|v|": abs(v).round(3),
                "vmax": vmax.round(3),
                "v_ok":v_ok,
            }),
        })
    })
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The real power flow $p_{ij}$ and reactive power flow $q_{ij}$ on the power lines from the nodes $i$ to the nodes $j$ must remain below the lines' capacities such that $p_{ij}^2+q_{ij}^2 
    \le \bar s_{ij}^2$. Note that in the `pypower` cases, the line limit is nominally `rateA`, with `rateB` and `rateC` reserved for emergency line ratings only.
    """
    )
    return


@app.cell
def _(N, branch, mo, model, model_ui, np, pd):
    # power injections by lines into nodes
    Smax, P, Q = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    for _i, _j, _pf, _qf, _pt, _qt, _smax in list(
        zip(
            *[
                getattr(branch, x)
                for x in ["fbus", "tbus", "Pf", "Qf", "Pt", "Qt", "rateA"]
            ]
        )
    ):  #
        _i, _j = int(_i) - 1, int(_j) - 1
        P[_i, _j], P[_j, _i] = _pt, _pf
        Q[_i, _j], Q[_j, _i] = _qt, _qf
        Smax[_i, _j] = Smax[_j, _i] = _smax
    S = P + Q * 1j

    # line flow check
    smax = branch.rateA
    sf = branch.Pf + branch.Qf * 1j
    st = branch.Pt + branch.Qt * 1j
    sloss = abs(abs(sf) - abs(st))
    s_ok = abs(sf) <= smax

    # show result
    mo.accordion(
        {
            f"{model_ui}: the line flows are {'ok' if s_ok.all() else 'not ok'} (click to view).": mo.ui.tabs(
                {
                    "Graph": mo.mermaid(model.graph(line=s_ok.tolist())),
                    "Table": pd.DataFrame(
                        {
                            "Line": range(1, len(s_ok) + 1),
                            "|sf|": abs(sf).round(1),
                            "|st|": abs(st).round(1),
                            "sloss": [f"{x:.1f} ({(100*x/abs(y)):.1f}%)" for x,y in zip(sloss,sf)],
                            "smax": smax.round(1),
                            "s_ok": s_ok,
                        }
                    ),
                }
            )
        }
    )
    return P, Q


@app.cell
def _(mo):
    mo.md(r"""The real and reactive powers into or out of the node $i$ are the sum of the flows through the transmission lines connected to the node $i$. The real and reactive power at each node $i$ is the difference of the real generation and demand $p_i=pg_i-pd_i$ and the reactive generation and demand $q_i=qg_i-qd_i$.""")
    return


@app.cell
def _(N, P, Q, mo, model, model_ui, pd):
    p,q = P.sum(axis=1),Q.sum(axis=1)
    mo.accordion({
        f"{model_ui}: The nodal nodal power injections are ok (click to view).": mo.ui.tabs({
            "Graph": mo.mermaid(model.graph(node=abs(p+q*1j).round(1).tolist())),
            "Table": pd.DataFrame({"node":range(1,N+1),"p":p,"q":q}).round(1),
        })
    })
    return


@app.cell
def _(mo):
    mo.md(r"""## 3.2 - Linear Approximations""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 3.2.1 - Linearized Power Flow""")
    return


@app.cell
def _():
    import os, sys, json, datetime, importlib, re
    import marimo as mo
    import pandas as pd
    import numpy as np
    import cvxpy as cp
    import pypower.runopf as pr
    import pypower.ppoption as po
    from collections import namedtuple
    from model import Model

    np.set_printoptions(linewidth=999,precision=4,suppress=False,threshold=1000)
    return Model, mo, np, os, pd, re


if __name__ == "__main__":
    app.run()
