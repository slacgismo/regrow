import marimo

__generated_with = "0.6.26"
app = marimo.App(width="medium")


@app.cell
def __():
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import pandas as pd
    return cp, mo, np, pd, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Stochastic control

        - Want to control a dynamical system
          \[
          x_{t+1} = f(x_t, u_t, w_t), \quad t=0,1,\ldots,T-1
          \]
        - \( x_t \in \mathcal{X} \) is the state, \( u_t \in \mathcal{U} \) is the input or action, \( w_t \in \mathcal{W} \) is the disturbance
        - \( x_0, w_0, \ldots, w_{T-1} \) are independent random variables
        - State feedback **policy** \( u_t = \pi_t(x_t) \), \( t=0,1,\ldots,T-1 \)
        - Want to find policy that minimizes expected cost
        - Examples: investing, consumption, *grid management*, ...

        ---

        ## Model predictive control (MPC)

        - Can find such a policy using heuristics, dynamic programming, ...
        - **Model predictive control** provides principled policy framework
          - **Forecast**: predict stochastic future values
          - **Plan**: solve certainty equivalent problem assuming forecasts are correct
          - **Execute**: take first action in plan
          - **Repeat**
        - Works extremely well in practice (*e.g.*, to land rockets)

        ---

        ## MPC for grid management

        - Can specify as convex optimization problem

        $$
        \begin{array}{ll}
        \text{minimize} & - \mathbf{1}^Tq_T + \sum_{t=1}^T \|c_t\|_1 \Delta + \sum_{j=1}^M R_i f_i^2 \Delta + \tau \sum_{t=1}^T \|g_t\|_2^2\\
        \text{subject to} & q_0 = s, \quad q_{t+1} = q_t + c_t \Delta, \\
        & Af_t - c_t + r^{\text{curt}}_t + g_t = l_t, \\
        & q_t \leq Q, \quad r^{\text{curt}}_t \leq r_t, \quad g_t \leq g^{\text{lim}}_t,\\
        & |c_{i,t}| \leq C_i, \quad |f_{j, t}| \leq F_j, \\
        \end{array}
        $$

        with variables $c_t \in \mathbf{R}_+^{N}$, $q \in \mathbf{R}_+^{N}$, $r^{\text{curt}}_t \in \mathbf{R}_+^{N}$, $g_t \in \mathbf{R}_+^{N}$, $f_t \in \mathbf{R}^{M}$. 

        - Tractable optimization problem that can be solved efficiently and robustly
        - Can be implemented in only a few lines of code
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    tau_slider = mo.ui.slider(0, 100, 0.5, label=r"$\tau$", show_value=True)
    tau_slider
    return tau_slider,


@app.cell
def __(cp, np, sp, tau_slider):
    # In Python:

    def mpc(
        s: np.array,
        Q: np.array,
        A: sp.spmatrix,
        F: np.array,
        C: np.array,
        r: np.array,
        g_lim: np.array,
        l: np.array,
        R: np.array,
        kappa: np.array,
        verbose: bool = False
    ) -> (np.array, cp.Problem):
        """
        Args:
            s: battery SOC,                 np.array of shape N
            A: incidence matrix,            sp.sparray of shape N x M
            Q: battery capacity,            np.array of shape N
            F: line capacity,               np.array of shape M
            C: charge/discharge limit,      np.array of shape N
            r: renewable generation,        np.array of shape N x T
            g_lim: fossil generation limit  np.array of shape N x T
            l: load,                        np.array of shape N x T
            R: line resistance,             np.array of shape M
            kappa: charge/discharge cost,   np.array of shape N

        Returns:
            c: charge/discharge rate, np.array of shape N x T
            positive for charge, negative for discharge
            g: fossil generation, np.array of shape N x T
            problem: the CVXPY problem instance
        """

        N, T = r.shape
        M = A.shape[1]
        Delta_T = 1

        c = cp.Variable((N, T))
        q = cp.Variable((N, T + 1), nonneg=True)
        r_curt = cp.Variable((N, T), nonneg=True)
        g = cp.Variable((N, T), nonneg=True)
        f = cp.Variable((M, T))

        constraints = [
            r_curt <= r,
            q[:, 0] == s,
            q[:, 1:] == q[:, :-1] + c * Delta_T,
            A @ f - c + r_curt + g == l,
            q <= Q[:, None],
            g <= g_lim,
            cp.abs(f) <= F[:, None],
            cp.abs(c) <= C[:, None],
        ]

        objective = -cp.sum(q[:, -1]) + cp.norm1(c) * Delta_T + cp.sum(R @ cp.square(f)) * Delta_T + tau_slider.value * cp.sum_squares(g)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=verbose)

        return c.value, g.value, q.value, r_curt.value, problem
    return mpc,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## IEEE 14-Bus Grid Model

        The IEEE 14-bus system is a widely used test case in power system studies.

        It consists of 14 buses, 5 generators, and 11 loads.
        The network includes:

        - Buses: Nodes where power is either injected (generation) or drawn (load).
        - Lines: Transmission lines connecting the buses.
        - Generators: Units producing electrical power.
        - Loads: Consumption points within the system.

        This model is used to evaluate power flow solutions, stability analysis, and economic dispatch problems.
        It serves as the basis for this notebook.
        """
    )
    return


@app.cell
def __(pd):
    weather = pd.read_csv("data/weather/solar.csv", index_col=0)
    solar = weather.iloc[:72, :14].values.T
    solar[:, 60:] = 0
    return solar, weather


@app.cell
def __(batter_soc_slider, np, solar):
    # Load problem data

    A = np.array(
    [[-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [1.,0.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,1.,0.,1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,1.,0.,0.,1.,0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.,-1.,-1.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,-1.,-1.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,-1.,-1.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,-1.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,-1.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,-1.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,1.]]
    )

    N, M = A.shape
    T = 72

    # TODO: replace r with PVWatts data for 14 bus locations.

    def hello_world():
        Q = np.ones(N) * 10000
        F = np.ones(M) * 10000
        C = np.ones(N) * 10000
        g = np.ones((N, T)) * 50
        l = np.ones((N, T)) * 100
        s = np.ones(N) * batter_soc_slider.value
        r = solar * 10
        R = np.zeros(M)
        kappa = np.zeros(N)
        return s, Q, A, F, C ,r, g, l, R, kappa
    return A, M, N, T, hello_world


@app.cell
def __(mo):
    batter_soc_slider = mo.ui.slider(0,1000,1,value=263,label="initial battery SOC", show_value=True)
    batter_soc_slider
    return batter_soc_slider,


@app.cell
def __(plt, solar):
    plt.figure()
    plt.plot(solar[0])
    return


@app.cell
def __(hello_world, mpc, plt):
    # Execute single plan step
    _c, _g, _q, _r, _problem = mpc(*hello_world())
    print(_problem.status)
    fig = plt.figure()
    axis0 = plt.subplot(4,1,1)
    axis0.plot(_c[0])
    axis0.set_title("charge/discharge")
    axis1 = plt.subplot(4,1,2)
    axis1.plot(_g[0])
    print(sum(_g[0]))
    axis1.set_title("fossil generation")
    axis2 = plt.subplot(4,1,3)
    axis2.plot(_q[0])
    axis2.set_title("battery SOC")
    axis3 = plt.subplot(4,1,4)
    axis3.plot(_r[0])
    axis3.set_title("renewable generation")
    plt.tight_layout()
    fig

    return axis0, axis1, axis2, axis3, fig


@app.cell
def __(A, M, N, T, np, solar):
    def changing_renewable_generation():
        Q = np.ones(N) * 10000
        F = np.ones(M) * 10000
        C = np.ones(N) * 10000
        g = np.ones((N, T)) * 50
        l = np.ones((N, T)) * 100
        s = np.zeros(N)
        r = solar * 10
        r[:, : 24] = r[:, : 24]*1.5
        r[:, 24:48] = r[:, 24:48]*0.5
        R = np.zeros(M)
        kappa = np.zeros(N)
        return s, Q, A, F, C ,r, g, l, R, kappa

    def noisy():
        Q = np.ones(N) * 10000
        F = np.ones(M) * 10000
        C = np.ones(N) * 10000
        g = np.ones((N, T)) * 50 + (np.random.random((N,T)) - 0.3) * 10
        l = np.ones((N, T)) * 100 + (np.random.random((N,T)) - 0.7) * 10
        s = np.zeros(N)
        r = solar * 10
        R = np.zeros(M)
        kappa = np.zeros(N)
        return s, Q, A, F, C ,r, g, l, R, kappa
    return changing_renewable_generation, noisy


@app.cell
def __(changing_renewable_generation, mpc, plt):
    # Execute single plan step
    _c, _problem = mpc(*changing_renewable_generation())
    print(_problem.status)
    plt.figure()
    plt.plot(_c[0])
    return


@app.cell
def __(mpc, noisy, plt):
    # Execute single plan step
    _c, _problem = mpc(*noisy())
    print(_problem.status)
    plt.figure()
    plt.plot(_c[0])
    return


@app.cell
def __(mo):
    mo.md("## Prescient vs. forecasting")
    return


@app.cell
def __(mo):
    mo.md("## Robust MPC using quantile regression")
    return


@app.cell(hide_code=True)
def __(mo):
    demand_slider = mo.ui.slider(0,100,1,value=70,label="demand quantile", show_value=True)
    generation_slider = mo.ui.slider(0,100,1,value=30,label="generation quantile", show_value=True)
    mo.vstack([demand_slider, generation_slider])
    return demand_slider, generation_slider


@app.cell
def __(A, M, N, T, np, solar):
    def quantiles(d, g):
        Q = np.ones(N) * 10000
        F = np.ones(M) * 10000
        C = np.ones(N) * 10000
        g = np.ones((N, T)) * 50 + (np.random.random((N,T)) - 0.3) * 10
        l = np.ones((N, T)) * 100 + (np.random.random((N,T)) - 0.7) * 10
        s = np.zeros(N)
        r = solar * 10
        R = np.zeros(M)
        kappa = np.zeros(N)
        return s, Q, A, F, C ,r, g, l, R, kappa
    return quantiles,


@app.cell
def __(demand_slider, generation_slider, mpc, plt, quantiles):
    # Execute single plan step
    _c, _problem = mpc(*(quantiles(demand_slider.value, generation_slider.value)))
    print(_problem.status)
    plt.figure()
    plt.plot(_c[0])
    return


if __name__ == "__main__":
    app.run()
