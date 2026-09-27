import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import itertools
    # from collections import defaultdict

    MARKERS = '*vp.o3'*2
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""Prescient / MPC Parameters: frac $=.25$ and $\mu =0.1$""")
    return


@app.cell(hide_code=True)
def _():
    import os
    import cvxpy as cp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    from tqdm.notebook import tqdm
    from ipywidgets import interact, Dropdown, IntSlider, DatePicker
    def horizon(x, M, H):
        number_of_samples = len(x) - H - M
        future = np.zeros((number_of_samples, H))
        c = 0
        for i in range(M, len(x) - H):
            future_slice = x[i:i + H]
            future[c] = future_slice
            c += 1
        return future

    def process_single_node_data_2018(G, verbose=False):
        df = pd.read_csv('single_node_data.csv', index_col=0, parse_dates=True)
        df[df.columns] = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
        df = df[~df.index.duplicated(keep='first')]
        df['load[MW]'] = df['load[MW]'].mask(df['load[MW]'] < 100).interpolate(limit_direction='both').ffill().bfill()
        df.index += pd.Timedelta(hours=8)
        df = df.loc["2018-01-01 00:00":"2018-12-31 23:00"]

        l = df['load[MW]'].to_numpy() / 1000
        s = df['pv[MW]'].to_numpy() / 1000
        w = df['wind[MW]'].to_numpy() / 1000
        R = w + s
        netload = l - (R + G)

        infeasible_indices = np.where(netload > 0)[0]
        shortfall = np.maximum(l - (R + G), 0)
        if verbose:
            print("See the scenario WITHOUT BATTERIES: ") 
            print(f"percentage of shortfall times = {len(infeasible_indices) / len(l) * 100:.2f} %")
            print(f"average load = {np.mean(l):.2f} GW")
            print(f"average renewable generation = {np.mean(R):.2f} GW")
            print(f"average shortfall = {np.mean(shortfall):.2f} GW")
            print(f"maximum fossil generation = {np.max(G):.2f} GW")
        return l, R, shortfall

    G = 1.0
    M = 0
    l, R, shortfall = process_single_node_data_2018(G, True)
    hor_l_1 = horizon(l, M, 1)
    hor_R_1 = horizon(R, M, 1)
    hor_l_24 = horizon(l, M, 24)
    hor_R_24 = horizon(R, M, 24)
    hor_l_72 = horizon(l, M, 72)
    hor_R_72 = horizon(R, M, 72)

    def make_constants (M, G, sparse, etas):
        constants = {
            'hor_l': None,
            'hor_R': None,
            'G': G,
            'C_max': 0.33,
            'cons_alpha': 1.25,
            'cons_beta': 0.5,
            'cons_lambda': 20.0,
            'cons_gamma': 5.0,
            'cons_sparse': sparse,
            'cons_mu': None,
            'cons_frac': None,
            'cons_Q': None,
            'cons_eta_storage':etas,
            'cons_eta_charge':etas,
            'cons_eta_discharge':etas,
            'H': None,
            'M': M,
            'num_samples': None,
            'number_of_Qs': 6,
            'Qs': np.array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            'number_of_mus': 1,
            'mus': np.array([0.1]),
            'number_of_fracs': 1,
            'fracs': np.array([0.25]),
            }
        return constants
    return cp, np, os, plt


@app.cell(hide_code=True)
def _(cp, np):
    def one_shot(l, R, G):
        T = R.shape[0]
        param_Q = cp.Parameter(nonneg=True, name='Q')
        param_alpha = cp.Parameter(nonneg=True, name='alpha')
        param_beta = cp.Parameter(nonneg=True, name='beta')
        param_gamma = cp.Parameter(nonneg=True, name='gamma')
        param_lambda = cp.Parameter(nonneg=True, name='lambda')
        param_sparse = cp.Parameter(nonneg=True, name='sparse')

        param_eta_storage = cp.Parameter(nonneg=True, name='eta_storage')
        param_eta_charge = cp.Parameter(nonneg=True, name='eta_charge')
        param_eta_discharge = cp.Parameter(nonneg=True, name='eta_discharge')

        B = cp.Variable(nonneg=True, name='B')
        b_charge = cp.Variable(T, nonneg=True, name = 'b_charge')
        b_discharge = cp.Variable(T, nonneg=True, name = 'b_discharge')
        q = cp.Variable(T+1, nonneg=True, name='q')
        r = cp.Variable(T, nonneg=True, name='r')
        u = cp.Variable(T, nonneg=True, name='u')
        s = cp.Variable(T, nonneg=True, name='s')
        c = cp.Variable(T, nonneg=True, name='c')
        constraints = [
            c == R - r,
            r <= R,
            q <= param_Q,
            u <= G,
            b_charge <= B,
            b_discharge <= B,
            q[1:]==param_eta_storage*q[:-1]+ -b_discharge/param_eta_discharge + b_charge*param_eta_charge,
            s <= l,
            b_discharge - b_charge + r + u == l - s,
            B == 0.33*param_Q,
            q[0] == 1.0*param_Q,
        ]
        objective = 1/T*(param_gamma*cp.sum(c)+param_lambda*cp.sum(s)+
                         param_alpha*cp.sum(u)+param_beta*cp.sum_squares(u)+                  param_sparse*cp.sum(b_discharge+b_charge))

        problem = cp.Problem(cp.Minimize(objective), constraints)
        return problem

    def retrieve(problem):
        q = problem.var_dict['q'].value
        b_charge = problem.var_dict['b_charge'].value
        b_discharge = problem.var_dict['b_discharge'].value
        u = problem.var_dict['u'].value
        s = problem.var_dict['s'].value
        c = problem.var_dict['c'].value
        r = problem.var_dict['r'].value
        B = problem.var_dict['B'].value
        return q, b_charge, b_discharge, u, s, c, r, B

    def prescient_sweep_Q(l, R, constants, label=None):
        Qs = constants['Qs']
        G = constants['G']
        storage = np.zeros((Qs.shape[0], l.shape[0]+1)) #q
        charge = np.zeros((Qs.shape[0], l.shape[0])) #b_charge
        discharge = np.zeros((Qs.shape[0], l.shape[0])) #b_discharge
        fossil = np.zeros((Qs.shape[0], l.shape[0])) #u
        slack = np.zeros((Qs.shape[0], l.shape[0])) #s
        curtailment = np.zeros((Qs.shape[0], l.shape[0])) #c
        renewable = np.zeros((Qs.shape[0], l.shape[0])) #r
        max_discharge = np.zeros((Qs.shape[0],)) #B
        objs = np.zeros((Qs.shape[0],)) 
        problem = one_shot(l, R, G)
        for i,Q in enumerate(Qs):
            print(f"Q = {Q:.2f}")
            problem.param_dict['Q'].value = Q
            problem.param_dict['alpha'].value = constants['cons_alpha']
            problem.param_dict['beta'].value = constants['cons_beta']
            problem.param_dict['gamma'].value = constants['cons_gamma']
            problem.param_dict['lambda'].value = constants['cons_lambda']
            problem.param_dict['sparse'].value = constants['cons_sparse']
            problem.param_dict['eta_storage'].value = constants['cons_eta_storage']
            problem.param_dict['eta_charge'].value = constants['cons_eta_charge']
            problem.param_dict['eta_discharge'].value = constants['cons_eta_discharge']
            problem.solve(solver=cp.CLARABEL, verbose=False)
            if problem.status == cp.INFEASIBLE:
                raise ValueError(f"Problem infeasible for Q = {Q:.2f}")
            elif problem.status == cp.UNBOUNDED:
                raise ValueError(f"Problem unbounded for Q = {Q:.2f}")
            print(f'objective: {problem.value:.4f}')
            objs[i] = problem.value
            q, b_charge, b_discharge, u, s, c, r, B = retrieve(problem)
            storage[i] = q
            charge[i] = b_charge
            discharge[i] = b_discharge
            fossil[i] = u
            slack[i] = s
            curtailment[i] = c
            renewable[i] = r
            max_discharge[i] = B
        if label != None:
            names = "store charge discharge fossil slack curtail renew maxDischarge obj".split(' ')
            for i,var in enumerate([storage[:,1:], charge, discharge, fossil,
                                    slack, curtailment, renewable, max_discharge, objs]):
                np.save(f'test_data/{label}_{names[i]}.npy', var)
        return storage[:,1:], charge, discharge, fossil, slack, curtailment, renewable, max_discharge

    return


@app.cell
def _():
    all_eta = [1, .98, .95, .92]
    all_sparse = [3e-3, 1e-2, 3e-2]
    # for etas, sparse in itertools.product(all_eta, all_sparse):
    #     constants = make_constants(M,G, etas = etas, sparse = sparse)
    #     label = f'sparse{int(sparse*1000)}eta{int(etas*100)}'
    #     (q_pre, b_charge_pre, b_discharge_pre,
    #      u_pre, s_pre, c_pre, r_pre, B_pre) = prescient_sweep_Q(l, R, constants, label)
    return all_eta, all_sparse


@app.cell
def _(os):
    print([i for i in os.listdir('test_data') if 'sparse' in i][:3])
    return


@app.cell
def _(all_eta, all_sparse, mo):
    drop_etas = mo.ui.dropdown(all_eta, value= all_eta[0], label = 'charging efficiency eta')
    drop_sparse = mo.ui.dropdown(all_sparse, value = all_sparse[0], label = 'sparsity weight')
    caps = [0,2,4,8,16,32]
    drop_qs = mo.ui.dropdown(options={str(caps[i]):i for i in range(6)}, value= str(2),label="which battery capacity")
    num_start = mo.ui.number(0,8760, 1, value = 0, label='start time')
    num_window = mo.ui.number(24,24*30, 1, value = 24, label='time window')

    return caps, drop_etas, drop_qs, drop_sparse, num_start, num_window


@app.cell
def _(drop_etas, drop_qs, drop_sparse, mo, num_start, num_window):
    mo.vstack([drop_etas, drop_sparse, num_start, num_window, drop_qs])
    return


@app.cell
def _(caps, drop_etas, drop_sparse, np):

    name_root = (f'test_data/sparse{int(drop_sparse.value*1000)}'+ f'eta{int(drop_etas.value*100)}_')
    dischrg = np.load(name_root+'discharge.npy')
    chrg = np.load(name_root+'charge.npy')
    og = np.load('test_data/Giray_discharge.npy')
    is_chrg = ~np.isclose(chrg,0, atol = 1e-5)
    is_dischrg  = ~np.isclose(dischrg,0, atol = 1e-5)
    both = np.where(np.logical_and(is_chrg, is_dischrg))
    # print(f'simultaneous charge and discharge times (6 simulated years, one per battery size): {len(both[0])}')
    arr_both = np.array([both[0], both[1]])
    both_qs = {}
    for nq in range(6):
       both_qs[caps[nq]]= np.sum(arr_both[0,:]==nq)
    # print(pd.DataFrame(both_qs, index = ['simultaneous charge by battery size']))



    return arr_both, chrg, dischrg, og


@app.cell
def _(
    chrg,
    dischrg,
    drop_etas,
    drop_qs,
    drop_sparse,
    np,
    num_start,
    num_window,
    og,
    plt,
):
    a,b,qi = num_start.value, num_window.value, drop_qs.value
    t = np.arange(a,a+b)
    plt.plot(t, dischrg[qi,a:a+b], label = 'discharge', marker = '.')
    plt.plot(t, chrg[qi,a:a+b], label = 'charge', marker = '.')
    plt.plot(t, dischrg[qi,a:a+b]-chrg[qi,a:a+b], label = 'net discharge',  dashes = [2,2])

    plt.plot(t, og[qi,a:a+b], label = 'single Giray',  dashes = [1,2])
    plt.legend()
    plt.ylabel('MWh')
    plt.xlabel('hours')
    plt.title(f'eta {drop_etas.value} sparse weight {drop_sparse.value}')
    return


@app.cell
def _(mo):
    array_slide = mo.ui.slider(0,500,value = 0,label = 'we explore the mistaken indeces', full_width = True)
    array_slide
    return (array_slide,)


@app.cell
def _(arr_both, array_slide):
    print(arr_both[:,array_slide.value:array_slide.value+12])
    print('the upper row is the array index relevant to the battery size (0,2,4,8,26,32). \nthe lower row refers to the hour index of simultaneous chargin and discharging')
    return


if __name__ == "__main__":
    app.run()
