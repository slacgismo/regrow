import marimo

__generated_with = "0.11.14-dev6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        """
        # FD Optimal Powerflow (OPF) Using CVX

        This notebook demonstrates how to solve a Fast-decoupled (FD) AC optimal powerflow problem (OPF) using CVX. The solution is designed to always be feasible despite tight constraints on the bus voltages. This is achieved by providing a load shedding variable at a cost $\Lambda$, albeit with the expectation that $\Lambda >> \Psi$, the cost of generation. In the limit of an otherwise infeasible system the load shedding would be 100% of all load and still be feasible.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        The problem is posed for a network with $N$ nodes and $K$ branches as follows:

        $\\begin{array}{rll} \\underset{x,y,g,h,c,d}{\\min} & \\Psi~|g+hj| + \\Lambda d
        \\\\ \\textrm{subject to} & L~x - g + c + \\Re D - d = 0 & \\textrm{(Kirchoff's current/voltage laws for real power)} 
        \\\\ & M~y - h - c + \\Im D - d \\frac{\\Im D}{\\Re D} = 0 & \\textrm{(Kirchoff's current/voltage laws for reactive power)} 
        \\\\ & x_0 = 0 & \\textrm{(reference bus voltage angle)} 
        \\\\ & y_0 = 1 & \\textrm{(reference bus voltage magnitude)} 
        \\\\ & \\Re \\v S \\le g \\le \\Re \\hat S & \\textrm{(generation real power constraints)} 
        \\\\ & \\Im \\v S \\le h \\le \\Im \\hat S & \\textrm{(generation reactive power constraints)} 
        \\\\ & \\v L \\le I ~ x \\le \\hat L & \\textrm{(line real power flow constraints)} 
        \\\\ & |y-1| \\le 0.05 & \\textrm{(voltage standard limits) }
        \\\\ & |c| \le C & \\textrm{(reactor/capacitor bank size)}
        \\\\ & 0 \le d \le \\Re D & \\textrm{(load shedding)}
        \\end{array}$

        where

        * $\Psi \in \mathbb{R}^N$ is the generation real energy price,
        * $L \\in \mathbb{R}^{N \\times N}$ is the weighted graph Laplacian for admittance,
        * $x \in \mathbb{R}^N$ is the nodal voltage angle solution,
        * $g \in \mathbb{R}^N$ is the generation real power dispatch solution,
        * $c \in \mathbb{R}^N$ is the capacitor bank setting solution,
        * $\Lambda \in \mathbb{R}^N$ is the price of load shedding,
        * $d \in \mathbb{R}^N$ is the load shedding dispatch solution,
        * $D \in \mathbb{R}^N$ is the nodal net power demand,
        * $M \in \mathbb{R}^{N \\times N}$ is the weighted graph Laplacian for susceptance,
        * $y \in \mathbb{R}^N$ is the nodal voltage magnitude solution,
        * $h \in \mathbb{R}^N$ is the nodal generation reactive power dispatch solution,
        * $\hat S \in \mathbb{C}^N$ is the maximum nodal generation real and reactive power capacity,
        * $\check S \in \mathbb{C}^N$ is the minimum nodal generation real and reactive power capacity,
        * $\hat L \in \mathbb{R}^K$ is the forward line flow real power constraint,
        * $\check L \in \mathbb{R}^K$ is the reverse line flow real power constraint,
        * $I \in \mathbb{R}^{K \\times N}$ is the weighted line-node incidence matrix, and
        * $C \in \mathbb{R}^N$ is the capacitor bank size.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Example

        In this example you can control the generation capacity and price, the load, and available reactors and capacitors are each bus. You can also control line resistance and line flow limits.  The price of demand response is fixed at 1.0 and reactive power demand of load is set a 1/10 the real power demand.

        Initially the problem is infeasible because there are no capacitors and reactors to support voltage and there is insufficient generation to support the load. You can make the problem feasible by adding these to different busses. Try adding capacitors to make the voltage solution feasible. Then try adding generation to reduce the amount of load shedding to zero.

        If you have trouble finding a feasible configuration of generators and reactor/capacitor banks, try enabling "Solve sizing problem" and following some of the recommendations.
        """
    )
    return


@app.cell
def _(mo):
    #
    # Network configuration
    #
    S0 = mo.ui.slider(0, 1, 0.01, 1.0, label="$C_0$", debounce=True)
    S1 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_1$", debounce=True)
    S2 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_2$", debounce=True)
    S3 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_3$", debounce=True)

    P0 = mo.ui.slider(0, 1, 0.01, 0.5, label="$P_0$", debounce=True)
    P1 = mo.ui.slider(0, 1, 0.01, 0.0, label="$P_1$", debounce=True)
    P2 = mo.ui.slider(0, 1, 0.01, 0.0, label="$P_2$", debounce=True)
    P3 = mo.ui.slider(0, 1, 0.01, 0.0, label="$P_3$", debounce=True)

    D0 = mo.ui.slider(0, 1, 0.01, 0.0, label="$D_0$", debounce=True)
    D1 = mo.ui.slider(0, 1, 0.01, 0.1, label="$D_1$", debounce=True)
    D2 = mo.ui.slider(0, 1, 0.01, 0.2, label="$D_2$", debounce=True)
    D3 = mo.ui.slider(0, 1, 0.01, 0.3, label="$D_3$", debounce=True)

    SC0 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_0$", debounce=True)
    SC1 = mo.ui.slider(0, 1, 0.01, 0.1, label="$C_1$", debounce=True)
    SC2 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_2$", debounce=True)
    SC3 = mo.ui.slider(0, 1, 0.01, 0.0, label="$C_3$", debounce=True)

    R0 = mo.ui.slider(0.01, 1, 0.01, 0.15, label="$R_0$", debounce=True)
    R1 = mo.ui.slider(0.01, 1, 0.01, 0.16, label="$R_1$", debounce=True)
    R2 = mo.ui.slider(0.01, 1, 0.01, 0.13, label="$R_2$", debounce=True)

    C00 = mo.ui.slider(0, 1, 0.01, 1.0, label="$\\hat L_0$", debounce=True)
    C01 = mo.ui.slider(0, 1, 0.01, 1.0, label="$\\hat L_1$", debounce=True)
    C02 = mo.ui.slider(0, 1, 0.01, 1.0, label="$\\hat L_2$", debounce=True)

    C10 = mo.ui.slider(-1, 0, 0.01, -1.0, label="$\\v L_0$", debounce=True)
    C11 = mo.ui.slider(-1, 0, 0.01, -1.0, label="$\\v L_1$", debounce=True)
    C12 = mo.ui.slider(-1, 0, 0.01, -1.0, label="$\\v L_2$", debounce=True)

    mo.vstack(
        [
            mo.hstack([mo.md("Generation capacity:"), S0,S1,S2,S3]),
            mo.hstack([mo.md("Generation price:"), P0, P1, P2, P3]),
            mo.hstack([mo.md("Load demand:"), D0, D1, D2, D3]),
            mo.hstack([mo.md("Reactors/capacitors:"), SC0, SC1, SC2, SC3]),
            mo.md("---"),
            mo.hstack([mo.md("Line resistance:"), R0, R1, R2]),
            mo.hstack([mo.md("Forward flow limit:"), C00, C01, C02]),
            mo.hstack([mo.md("Reverse flow limit:"), C10, C11, C12]),
        ]
    )
    return (
        C00,
        C01,
        C02,
        C10,
        C11,
        C12,
        D0,
        D1,
        D2,
        D3,
        P0,
        P1,
        P2,
        P3,
        R0,
        R1,
        R2,
        S0,
        S1,
        S2,
        S3,
        SC0,
        SC1,
        SC2,
        SC3,
    )


@app.cell
def _(B, C, D, I, M, N, P, XY, c, d, g, h, math, mo, np, plt, x, y):
    _X, _Y = [x[0] for x in XY], [x[1] for x in XY]
    if x.value is None:
        _V = np.zeros(N)
        _M = np.zeros(N)
        _F = np.zeros(M, dtype=complex)
        _G = np.zeros(N)
        _H = np.zeros(N)
        _C = np.zeros(N)
    else:
        _V = x.value
        _M = y.value
        _F = I @ x.value
        _G = g.value
        _H = h.value
        _C = c.value
    for _n, _xy in enumerate(XY):
        plt.text(
            *_xy,
            str(_n),
            color="w",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.text(
            *_xy,
            f"    $V_{_n}={_M[_n]:.2f}\\angle{_V[_n]:.2f}$",
            horizontalalignment="left",
            verticalalignment="bottom",
        )
        plt.text(
            *_xy,
            f"    $G_{_n}={_G[_n]:.2f}{_H[_n]:+.2f}j$",
            horizontalalignment="left",
            verticalalignment="top",
        )
        plt.text(
            *_xy,
            f"$P_{_n}={P[_n]:.2f}, C_{_n}={C[_n]:.2f}$    ",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        plt.text(
            *_xy,
            f"$L_{_n}={D[_n]:.2f}{'' if d.value[_n]<=0.001 else '\\mathbf{{{:+.2f}}}'.format(-round(d.value[_n],2))}$    ",
            horizontalalignment="right",
            verticalalignment="top",
        )
    for _m, _b in enumerate(B):
        _x, _y = [_X[_b[0]], _X[_b[1]]], [_Y[_b[0]], _Y[_b[1]]]
        _a = math.atan2(np.diff(_y)[0], np.diff(_x)[0]) % (2 * np.pi)
        _flip = (np.pi / 2) < _a <= (3 * np.pi / 2)
        # print(_b, _a / (np.pi / 2))
        plt.plot(_x, _y, "-k")
        plt.text(
            np.mean(_x),
            np.mean(_y),
            (
                f"$\\leftarrow ~ F_{_m}={_F[_m].real:.2f}{_F[_m].imag:+.2f}j$"
                if _flip
                else f"$F_{_m}={_F[_m].real:.2f}{_F[_m].imag:+.2f}j ~ \\rightarrow$"
            ),
            rotation=(_a + np.pi if _flip else _a) * 180.0 / np.pi,
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation_mode="anchor",
        )
    plt.plot(_X, _Y, "ok" if x.value is None else "ob", markersize=20)
    plt.axis("off")
    plt.gca().set_aspect("equal")
    mo.hstack([plt.gca()]).center()
    return


@app.cell
def _(np):
    #
    # Network configuration
    #
    B = np.array([[0, 1], [1, 2], [1, 3]])  # network
    XY = np.array([[1, 2], [1, 1], [0, 0], [2, 0]])  # node locations
    return B, XY


@app.cell
def _(B, S):
    #
    # Network properties
    #
    N = len(S)  # number of nodes
    M = len(B)  # number of lines
    return M, N


@app.cell
def _(
    C00,
    C01,
    C02,
    C10,
    C11,
    C12,
    D0,
    D1,
    D2,
    D3,
    P0,
    P1,
    P2,
    P3,
    R0,
    R1,
    R2,
    S0,
    S1,
    S2,
    S3,
    SC0,
    SC1,
    SC2,
    SC3,
    np,
):
    #
    # Network parameters
    #
    j = complex(0, 1)
    pr = 0.1  # reactive/real power ratio at load
    S = np.array([x.value for x in [S0, S1, S2, S3]])  # supply capacity
    Sh = S * (1 + 2 * j * pr)  # double pr for maximum reactive power
    Sl = -Sh  # double pr for minimum reactive power
    D = np.array([x.value for x in [D0, D1, D2, D3]]) * (1 + j * pr)  # net demand
    R = np.array([x.value for x in [R0, R1, R2]]) * (
        1 - j * pr / 2
    )  # line impedance with higher reactance
    C0 = np.array([x.value for x in [C00, C01, C02]])  # forward flow line limits
    C1 = np.array([x.value for x in [C10, C11, C12]])  # reverse flow line limits
    P = np.array([x.value for x in [P0, P1, P2, P3]])  # generation price
    C = np.array([x.value for x in [SC0,SC1,SC2,SC3]]) # cap bank sizes
    return C, C0, C1, D, P, R, S, Sh, Sl, j, pr


@app.cell
def _(B, C0, C1, D, E, M, N, R, XY, mo, np):
    #
    # Network validation
    #
    mo.stop(len(D) != N, "incorrect number of demand values")
    mo.stop(len(R) != M, "incorrect number of resistance values")
    mo.stop(len(XY) != N, "incorrect number of node locations")
    mo.stop(B.min() < 0 or B.max() >= N, "invalid node in network")
    mo.stop(len(C0) != M or min(C0) < 0, "invalid forward flow line limit")
    mo.stop(len(C1) != M or max(C1) > 0, "invalid reverse flow line limit")
    mo.stop(
        sum([1 if x < 1e-9 else 0 for x in np.abs(E)]) > 1,
        "network is not fully connected",
    )
    return


@app.cell
def _(B, N, R, np):
    #
    # Graph Laplacian
    #
    G = np.zeros((N, N), dtype=complex)
    for _n, _l in enumerate(B):
        G[_l[0], _l[1]] = G[_l[1], _l[0]] = 1 / R[_n]
    L = np.diag(sum(G)) - G
    return G, L


@app.cell
def _(L, np):
    #
    # Spectral graph analysis
    #
    _e, _u = np.linalg.eig(L.real)  # eigenvalues and eigenvectors
    _i = _e.argsort()
    E, U = _e[_i], _u.T[_i]
    return E, U


@app.cell
def _(B, M, N, R, np):
    #
    # Link-node incidence matrix
    #
    I = np.zeros((M, N))  # link-node incidence matrix
    for _n, _l in enumerate(B):
        I[_n][_l[0]] = R[_n].real
        I[_n][_l[1]] = -R[_n].real
    return (I,)


@app.cell
def _(C, C0, C1, D, I, L, N, P, R, S, Sh, Sl, cp, mo, np, pd, size_ui, time):
    #
    # Solve powerflow with constraints
    #
    x = cp.Variable(N)  # nodal voltage angles
    y = cp.Variable(N)  # nodal voltage magnitudes
    g = cp.Variable(N)  # generation real power dispatch
    h = cp.Variable(N)  # generation reactive power dispatch
    c = cp.Variable(N)  # capacitor bank settings
    d = cp.Variable(N)  # demand curtailment
    _cost = ( 0.01*cp.sum_squares(cp.abs(g+h*1j)) + P @ cp.abs(g + h * 1j) ) if not size_ui.value else cp.sum(cp.abs(L.real@x+L.imag@y)) #cp.sum(cp.abs((L.real+L.imag*1j)@(x+y*1j)))
    _shed = np.ones(N) @ d
    objective = cp.Minimize(
        _cost + _shed
    )  # minimum cost (generation + demand response)
    constraints = [
        L.real @ x - g + c + D.real - d == 0,  # KCL/KVL real power laws
        L.imag @ y - h - c + D.imag - d@D.imag/D.real== 0,  # KCL/KVL reactive power laws
        x[0] == 0,  # swing bus voltage angle always 0
        y[0] == 1,  # swing bus voltage magnitude is always 1
        g >= 0,  # generation minimum real power
        g <= (S.real if not size_ui.value else 1.0),  # generation maximum real power
        h >= (Sl.imag if not size_ui.value else -0.2),  # generation minimum reactive power
        h <= (Sh.imag if not size_ui.value else 0.2),  # generation maximum reactive power
        I @ x <= C0,  # forward line flow limits
        I @ x >= C1,  # reverse line flow limits
        cp.abs(y-1) <= 0.05, # limit voltage magnitude to 5% deviation
        cp.abs(c) <= (C if not size_ui.value else 1.0), # reactor/capacitor bank settings
        d >= 0, d <= D.real,  # demand curtailment
    ]
    problem = cp.Problem(objective, constraints)

    _tic = time.time()
    problem.solve()
    _toc = time.time()

    if x.value is None:
        result = mo.md(f"The problem is {problem.status}")

    else:
        node = pd.DataFrame(
            dict(
                voltage=[f"{_y.value:.2f}<{_x.value:.2f}" for _x, _y in zip(x, y)],
                dispatch=[
                    f"{abs(_g.value):.2f}{_h.value:+.2f}j" for _g, _h in zip(g, h)
                ],
                capbank=[f"{abs(_c.value):.2f}/{_C:.2f}" for _c, _C in zip(c, C)],
                loadshed=[f"{abs(_d.value):.2f}/{abs(_l.real):.2f}" for _d, _l in zip(d, D)],
                cost=g.value * P,
            )
        )
        node.index.name = "node"

        _P = I @ x.value / R
        line = pd.DataFrame(
            dict(flow=[f"{p:.2f}{q:+.2f}j" for p, q in zip(_P.real, _P.imag)])
        )
        line.index.name = "line"

        result = mo.vstack(
            [
                mo.md(f"Problem solved in {_toc - _tic:.3f} seconds."),
                mo.hstack(
                    [
                        mo.ui.table(node.round(3).reset_index()),
                        mo.ui.table(line.round(3).reset_index()),
                    ],
                    justify="start",
                ),
            ]
        )
    return (
        c,
        constraints,
        d,
        g,
        h,
        line,
        node,
        objective,
        problem,
        result,
        x,
        y,
    )


@app.cell
def _(mo):
    size_ui = mo.ui.checkbox(label="Solve sizing problem")
    return (size_ui,)


@app.cell
def _(S0, S1, S2, S3, SC0, SC1, SC2, SC3, mo, node, size_ui, x):
    _gen = []
    _cap = []
    _ss = [S0, S1, S2, S3]
    _cc = [SC0, SC1, SC2, SC3]
    if x.value is None:
        sizing = mo.md("Insufficient resources to solve problem")
    else:
        for _n, _value in node.iterrows():
            if abs(complex(_value.dispatch)) > abs(complex(_ss[_n].value)):
                _gen.append(
                    f"\n* Increase generator {_n} to {abs(complex(_value.dispatch)):.2f}"
                )
            if float(_value.capbank.split("/")[0]) > float(_cc[_n].value):
                _cap.append(
                    f"\n* Increase reactor/capacitor bank {_n} to {float(_value.capbank.split('/')[0]):+.2f}"
                )
        sizing = (
            mo.md(f"""## Sizing Recommendations

    ### Generators

    {"".join(_gen)}

    ### Capacitors

    {"".join(_cap)}
    """)
            if size_ui.value
            else mo.md("")
        )
    return (sizing,)


@app.cell
def _(P, d, g, mo, result, size_ui, sizing):
    mo.ui.tabs({
         "Solution":
            mo.vstack([size_ui,
                       mo.md(
                            f"Total cost = ${g.value@P:.2f}/h with {'no' if round(sum(d.value),3) == 0 else '{:.2f}'.format(sum(d.value))} load shed"
                            if not g.value is None
                            else "No solution"
                        ),
                        result if not size_ui.value else sizing,
                      ]),
        "Problem":
            mo.md(
                """
    CVX Solution
    ~~~
    x = cp.Variable(N)  # nodal voltage angles
    y = cp.Variable(N)  # nodal voltage magnitudes
    g = cp.Variable(N)  # generation real power dispatch
    h = cp.Variable(N)  # generation reactive power dispatch
    c = cp.Variable(N)  # capacitor bank settings
    d = cp.Variable(N)  # demand curtailment
    objective = cp.Minimize(P @ cp.abs(g+h*1j) + np.ones(N) @ d)  # minimum cost
    constraints = [
        L.real @ x - g + c + D.real - d == 0,  # KCL/KVL real power laws
        L.imag @ y - h - c + D.imag - d@D.imag/D.real== 0,  # KCL/KVL reactive power laws
        x[0] == 0,  # swing bus voltage angle always 0
        y[0] == 1,  # swing bus voltage magnitude is always 1
        g >= 0,  # generation minimum real power
        g <= S.real,  # generation maximum real power
        h >= Sl.imag,  # generation minimum reactive power
        h <= Sh.imag,  # generation maximum reactive power
        I @ x <= C0,  # forward line flow limits
        I @ x >= C1,  # reverse line flow limits
        cp.abs(y-1) <= 0.05, # limit voltage magnitude to 5% deviation
        cp.abs(c) <= C, # reactor/capacitor bank settings
        d >= 0, d <= D.real,  # demand curtailment
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    ~~~
    """
            ),
        }
    )
    return


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import time
    return cp, math, mo, np, pd, plt, time


if __name__ == "__main__":
    app.run()
