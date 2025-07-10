import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    eta_storage, eta_charge, eta_discharge = 1,1,1
    L1_charge_weights = True
    LABEL = 'SPARSE01eta100'
    MARKERS = '*vp.o3'*2
    return (
        L1_charge_weights,
        LABEL,
        MARKERS,
        eta_charge,
        eta_discharge,
        eta_storage,
        mo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Prescient / MPC Parameters: frac $=.25$ and $\mu =0.1$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Load data, set constants""")
    return


@app.cell
def _(eta_charge, eta_discharge, eta_storage):
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

    constants = {
        'hor_l': None,
        'hor_R': None,
        'G': G,
        'C_max': 0.33,
        'cons_alpha': 1.25,
        'cons_beta': 0.5,
        'cons_lambda': 20.0,
        'cons_gamma': 5.0,
        'cons_sparse': 0.01,
        'cons_mu': None,
        'cons_frac': None,
        'cons_Q': None,
        'cons_eta_storage':eta_storage,
        'cons_eta_charge':eta_charge,
        'cons_eta_discharge':eta_discharge,
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
    return R, constants, cp, l, np, os, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Prescient""")
    return


@app.cell
def _(L1_charge_weights, cp, np):

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
        if L1_charge_weights:
            objective = 1/T*(param_gamma*cp.sum(c)+param_lambda*cp.sum(s)+
                         param_alpha*cp.sum(u)+param_beta*cp.sum_squares(u)+                  param_sparse*(cp.norm1(b_discharge)+cp.norm1(b_charge)))
        else:
            objective = 1/T*(param_gamma*cp.sum(c)+param_lambda*cp.sum(s)+
                         param_alpha*cp.sum(u)+param_beta*cp.sum_squares(u)) 

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

    return (prescient_sweep_Q,)


@app.cell
def _(mo):
    button = mo.ui.run_button()
    return (button,)


@app.cell(hide_code=True)
def _(button):
    button
    return


@app.cell
def _(LABEL, R, button, constants, l, mo, prescient_sweep_Q):
    mo.stop(not button.value)


    (q_pre, b_charge_pre, b_discharge_pre,
     u_pre, s_pre, c_pre, r_pre, B_pre) = prescient_sweep_Q(l, R, constants, LABEL)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    The prescient runs with various efficiencies are run and saved in the 'test_data' folder

    * `perf` is battery with separate charge and discharge variables and efficiency = 1 \n
    * `L1perf` is as perf, but L1 penalty to charging and discharging \n
    * `L1eta{num}` will be a run as L1 perf but with the etas of storage, charge and discharge set to num percent
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Analysis""")
    return


@app.cell(hide_code=True)
def _(MARKERS, constants, np, os, plt):
    objs = {i[:-4]:np.load('test_data/'+i) for i in os.listdir('test_data') if 'obj' in i}
    plt.figure(figsize=(5,3))
    for i, key in enumerate(objs.keys()):
        plt.scatter(constants['Qs'], objs[key],
                    label = key, marker= MARKERS[i], alpha = .5)
    plt.legend()
    plt.title('Objectives across runs')
    plt.ylabel('Objective')
    plt.xlabel('Battery Size')
    plt.show()
    return (objs,)


@app.cell(hide_code=True)
def _(mo, objs):
    dropdown = mo.ui.dropdown(options=list(objs.keys()), value= 'Giray_obj',label="check error against original")
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown, mo):
    mo.vstack([dropdown])

    return


@app.cell(hide_code=True)
def _(constants, dropdown, objs, plt):
    plt.scatter(constants['Qs'], 
        100*(objs['Giray_obj'] - objs[dropdown.value])/(objs['Giray_obj']))
    plt.title("Objective error against Giray's prescient")
    plt.ylabel('percent error')
    plt.xlabel('Battery Size')
    plt.show()
    return


@app.cell
def _(mo, os):
    curve_types = [k.split('_')[1][:-4] for k in os.listdir('test_data') if 'eta100' in k][:9]

    run_types = [k.split('_')[0] for k in os.listdir('test_data') if 'discharge' in k]


    dropdown_curve = mo.ui.dropdown(options=curve_types, value= 'discharge',label="which variable curve")

    dropdown_runs = mo.ui.dropdown(options=run_types, value= 'L1eta95',label="which run")

    caps = [0,2,4,8,16,32]
    dropdown_Qs = mo.ui.dropdown(options={str(caps[i]):i for i in range(6)}, value= str(2),label="which battery capacity")

    slide = mo.ui.slider(24, 8760, label = 'start hour', full_width = True)
    slide_len = mo.ui.slider(24,24*30, label = 'length of window', full_width = True)

    return dropdown_Qs, dropdown_curve, dropdown_runs, slide, slide_len


@app.cell
def _(dropdown_Qs, dropdown_curve, dropdown_runs, mo, slide, slide_len):
    mo.vstack([dropdown_curve, dropdown_runs, dropdown_Qs, slide, slide_len ])
    return


@app.cell
def _(dropdown_Qs, dropdown_curve, dropdown_runs, np, plt, slide, slide_len):
    arr = np.load('test_data/'+
                  dropdown_runs.value+'_'+dropdown_curve.value+'.npy')

    og = np.load('test_data/giray_'+dropdown_curve.value+'.npy')
    charge = np.load('test_data/'+dropdown_runs.value+'_charge.npy')

    net_discharge = arr - charge

    plt.plot(arr[dropdown_Qs.value,slide.value:slide.value+slide_len.value], label = dropdown_curve.value +'_'+dropdown_runs.value, marker = '.')
    plt.plot(og[dropdown_Qs.value,slide.value:slide.value+slide_len.value], label = 'giray_'+dropdown_curve.value )
    plt.plot(charge[dropdown_Qs.value,slide.value:slide.value+slide_len.value], label = ' charge' +'_'+dropdown_runs.value, marker = '.')
    plt.plot(net_discharge[dropdown_Qs.value,slide.value:slide.value+slide_len.value],
             label = ' our net discharge' +'_'+dropdown_runs.value, dashes = [2,4])


    plt.legend()
    # error_percent = np.linalg.norm(arr- og,2)/np.norm(og)

    return


@app.cell
def _(os):
    os.listdir('test_data')
    ### USE NUMPY ALMOST EQUALS ZERO AND XOR/ OR LOIGIC. NP LOGICAL AND/OR etc
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
