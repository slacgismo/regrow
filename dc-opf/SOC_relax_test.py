import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _(cp, np):
    def make_problem(incidence_mat, generator_dict, load_dict, flow_dict, augmented=None, rho=1):
        def indices_not_in_list(array_length, given_indices):
            all_indices = set(range(array_length))
            given_indices_set = set(given_indices)
            not_in_list_indices = all_indices - given_indices_set
            return list(not_in_list_indices)
        _B = incidence_mat
        _A = np.abs(_B) @ np.abs(_B.T) - np.diag(np.diag(np.abs(_B) @ np.abs(_B.T)))
        _yij = 1/(np.array([_v['r'] for _k, _v in flow_dict.items()]) + 1j * np.array([_v['x'] for _k, _v in flow_dict.items()]))
        num_buses = _B.shape[0]
        num_edges = _B.shape[1]
        _Y = np.zeros((num_buses, num_buses), dtype=complex)
        for _idx, _edge in enumerate(_B.T):
            _i, _j = np.where(_edge==-1)[0], np.where(_edge==1)[0]
            _Y[_i, _j] = -_yij[_idx]
            _Y[_j, _i] = -_yij[_idx]
        np.fill_diagonal(_Y, np.sum(-_Y, axis=1))
        _YH = _Y.conjugate().T
        gen_ixs = np.array(list(generator_dict.keys())) - 1
        nogen_ixs = indices_not_in_list(num_buses, gen_ixs)
        # p_flows = cp.Variable((num_edges), name='line flows')
        p_gen = cp.Variable((num_buses), nonneg=True, name='p_gen')
        p_load = cp.Variable((num_buses), nonneg=True, name='p_load')
        q_bus = cp.Variable((num_buses), name='q_bus')
        V = cp.Variable((num_buses, num_buses), complex=True, name='V')
        P = cp.Variable((num_buses, num_buses), name='P')
        Q = cp.Variable((num_buses, num_buses), name='P')
        l_min = cp.Parameter(
            shape=num_buses,
            value=[_v['l_min'] for _k, _v in load_dict.items()],
            name='l_min'
        )
        l_upper = cp.Parameter(
            shape=num_buses,
            value=[_v['l_upper'] for _k, _v in load_dict.items()],
            name='l_upper'
        )
        # l_cost = cp.Parameter(
        #     shape=num_buses,
        #     value=[_v['cost'] for _k, _v in load_dict.items()],
        #     name='l_cost'
        # )
        l_cost = [_v['cost'] for _k, _v in load_dict.items()]
        g_min = cp.Parameter(
            shape=len(gen_ixs),
            value=np.array([_v['p_min'] for _k, _v in generator_dict.items()]),
            name='g_min'
        )
        g_max = cp.Parameter(
            shape=len(gen_ixs),
            value=np.array([_v['p_max'] for _k, _v in generator_dict.items()]),
            name='g_max'
        )
        c0 = cp.Parameter(
            shape=len(gen_ixs),
            value=np.array([_v['c0'] for _k, _v in generator_dict.items()]),
            name='c0'
        )
        c1 = cp.Parameter(
            shape=len(gen_ixs),
            value=np.array([_v['c1'] for _k, _v in generator_dict.items()]),
            name='c1'
        )
        # c2 = cp.Parameter(
        #     shape=len(gen_ixs),
        #     value=np.array([_v['c2'] for _k, _v in generator_dict.items()]),
        #     name='c2'
        # )
        c2 = np.array([_v['c2'] for _k, _v in generator_dict.items()])
        r = cp.Parameter(
            shape=num_edges,
            value=np.array([_v['r'] for _k, _v in flow_dict.items()]),
            name='line resistance'
        )
        f_max = cp.Parameter(
            shape=num_edges,
            value=np.array([_v['f_max'] for _k, _v in flow_dict.items()]),
            name='line capacities'
        )
        # generator costs
        cost = cp.sum(c0 + cp.multiply(c1, p_gen[gen_ixs]) + cp.multiply(c2, cp.square(p_gen[gen_ixs])))
        # load curtailment costs
        cost += cp.sum(
            cp.multiply(
                cp.neg(p_load - l_upper),
                l_cost
            )
        )
        # line penalties
        cost += cp.sum_squares(cp.multiply(r, P))
        # admm
        if augmented is not None:
            cost += rho * 0.5 * cp.sum_squares(V - augmented)
        constraints = [
            (p_gen - p_load) + 1j * q_bus == cp.diag(V @ _YH),
            cp.sum(P, axis=1) == p_gen - p_load,
            cp.sum(Q, axis=1) == q_bus,
            cp.multiply(P, 1 - _A) == 0,
            cp.multiply(Q, 1 - _A) == 0,
            p_load >= l_min,
            cp.abs(P) <= f_max,
            p_gen[nogen_ixs] == 0,
            p_gen[gen_ixs] <= g_max,
            p_gen[gen_ixs] >= g_min,
            cp.real(cp.diag(V)) >= 0.9,
            cp.real(cp.diag(V)) <= 1.1,
            cp.imag(cp.diag(V)) == 0,
            cp.real(V) == cp.real(V.T),
            cp.imag(V) == -cp.imag(V.T)
        ]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        return problem
    return (make_problem,)


@app.cell
def _(np):
    def proj_rank1(mat):
        _evals, _evecs = np.linalg.eigh(mat)
        _k = 1
        _M = _evecs[:, -_k:]
        _D = np.diag(_evals[-_k:])
        return _M @ _D @ _M.conjugate().T
    return (proj_rank1,)


@app.cell
def _(
    B,
    flow_dict,
    generator_dict,
    load_dict,
    make_problem,
    np,
    num_buses,
    proj_rank1,
):
    def admm_step(zlast=None, ulast=None, rho=1):
        if zlast is None:
            zlast = np.zeros((num_buses, num_buses))
        if ulast is None:
            ulast = np.zeros((num_buses, num_buses))
        problem = make_problem(B, generator_dict, load_dict, flow_dict, augmented=zlast - ulast, rho=rho)
        problem.solve(solver='CLARABEL')
        xnew = problem.var_dict['V'].value
        znew = proj_rank1(xnew + ulast)
        unew = ulast + xnew - znew
        return xnew, znew, unew, np.linalg.norm(xnew - znew)
    return (admm_step,)


@app.cell
def _(admm_step, mo, plt):
    _zl, _ul = None, None
    _rho = 1
    rs = []
    for _i in range(500):
        if (_i + 1) % 10 == 0:
            _rho *= 2
        xnext, znext, unext, r = admm_step(zlast=_zl, ulast=_ul, rho=_rho)
        _zl, _ul = znext, unext
        rs.append(r)
        fig = plt.plot(rs)
        mo.output.replace(fig)
        plt.close()
    return xnext, znext


@app.cell
def _(am_solving, np, plt, sns, xnext):
    am_solving
    sns.heatmap(np.real(xnext), cmap='seismic', center=0)
    plt.title('real part')
    plt.gcf()
    return


@app.cell
def _(am_solving, np, plt, xnext):
    am_solving
    plt.stem(np.diag(xnext))
    plt.gcf()
    return


@app.cell
def _(np, znext):
    _eval, _evec = np.linalg.eigh(znext)
    _eval
    return


@app.cell
def _():
    return


@app.cell
def _(am_solving, np, plt, problem, sns):
    am_solving
    sns.heatmap(np.real(problem.var_dict['V'].value), cmap='seismic', center=0)
    plt.title('real part')
    plt.gcf()
    return


@app.cell
def _(am_solving, np, plt, problem2, sns):
    am_solving
    sns.heatmap(np.imag(problem2.var_dict['V'].value), cmap='seismic', center=0)
    plt.title('imag part')
    plt.gcf()
    return


@app.cell
def _(am_solving, np, plt, problem):
    am_solving
    plt.stem(np.diag(problem.var_dict['V'].value))
    plt.gcf()
    return


@app.cell
def _(problem, proj_rank1):
    V_proj = proj_rank1(problem.var_dict["V"].value)
    error = problem.var_dict["V"].value - V_proj
    return V_proj, error


@app.cell
def _(
    B,
    V_proj,
    error,
    flow_dict,
    flow_limits,
    gen_limits,
    generator_dict,
    load_dict,
    load_limits,
    make_problem,
):
    problem2 = make_problem(B, generator_dict, load_dict, flow_dict, augmented=V_proj - error)
    # solve with updated parameters
    # am_solving = True
    problem2.param_dict['g_min'].value = [_v[0] for _v in gen_limits.value]
    problem2.param_dict['g_max'].value = [_v[1] for _v in gen_limits.value]
    problem2.param_dict['l_min'].value = [_v[0] for _v in load_limits.value]
    problem2.param_dict['l_upper'].value = [_v[1] for _v in load_limits.value]
    problem2.param_dict['line capacities'].value = flow_limits.value
    obj_val2 = problem2.solve(verbose=True, solver='CLARABEL')
    return (problem2,)


@app.cell
def _(am_solving, np, problem2):
    am_solving
    evals, evecs = np.linalg.eigh(problem2.var_dict['V'].value)
    return (evals,)


@app.cell
def _(evals):
    evals
    return


@app.cell(hide_code=True)
def _(mo, os, re):
    # model loading widget 
    _list = sorted(
        [x for x in os.listdir(".") if re.match("case.+.py",x)], # get only "case*.py"
        key=lambda x: int(re.sub("[^0-9]*([0-9]+)[^0-9]*", r"\1", x, 1)), # sort by numerical order not lexical
    )
    _options = {os.path.splitext(x)[0]: x for x in _list}
    model_ui = mo.ui.dropdown(
        options=_options, value=os.path.splitext(_list[0])[0], label='select model:'
    )
    return (model_ui,)


@app.cell(hide_code=True)
def _(model_ui, os):
    # import model from file
    from importlib.machinery import SourceFileLoader
    _n = model_ui.value.split('.')[0]
    model_loader = SourceFileLoader(_n, os.path.join(".",model_ui.value)).load_module()
    load_model = getattr(model_loader, _n)
    model_data = load_model()
    return (model_data,)


@app.cell(hide_code=True)
def _(model_data):
    # define network
    num_buses = model_data['bus'].shape[0]
    lines = [(int(_b[0]), int(_b[1])) for _b in model_data['branch']]
    return lines, num_buses


@app.cell(hide_code=True)
def _(both, gen_only, lines, load_only):
    # build graph 
    graph = ["flowchart LR"]
    for _ix, _l in enumerate(lines):
        _f = _l[0]
        _t = _l[1]
        graph.append(f"  {_f:.0f}(({_f})) == {_ix+1} === {_t:.0f}(({_t}))")
    graph = "\n".join(graph)
    graph += '''\n
        classDef gen stroke-dasharray: 5 5;\n
        classDef load fill:#f9f;\n
        classDef genload stroke-dasharray: 5 5,fill:#f9f;'''
    if len(load_only) > 0:
        _str = ",".join([str(int(_i)) for _i in load_only])
        graph += "\n    class "+_str+" gen;"
    if len(gen_only) > 0:
        _str = ",".join([str(int(_i)) for _i in gen_only])
        graph += "\n    class "+_str+" load;"
    if len(both) > 0:
        _str = ",".join([str(int(_i)) for _i in both])
        graph += "\n    class "+_str+" genload;"
    return (graph,)


@app.cell(hide_code=True)
def _(lines, np, num_buses, sns):
    # define incidence matrix
    B = np.zeros((num_buses, len(lines)))
    for _ix, _l in enumerate(lines):
        B[_l[0]-1, _ix] = -1
        B[_l[1]-1, _ix] = 1
    fig_heatmap = sns.heatmap(B, cmap='bwr', center=0)
    return B, fig_heatmap


@app.cell
def _(model_data, model_ui, np):
    # construct dictionaries for problem formulation
    def merge_nested_dicts(dict1, dict2):
        result = {}

        for key in dict1.keys() | dict2.keys():
            inner_dict1 = dict1.get(key, {})
            inner_dict2 = dict2.get(key, {})

            # Merge the inner dictionaries
            merged_inner_dict = {**inner_dict1, **inner_dict2}

            result[key] = merged_inner_dict

        return result
    # minimum generation value occaisonally negative
    generator_dict1 = {int(_g[0]): {'p_min': np.clip(_g[9], 0, np.inf), 'p_max': _g[8]*2} for _g in model_data['gen']}
    generator_dict2 = {int(model_data['gen'][_ix, 0]): {'c0': _g[-1], 'c1': _g[-2], 'c2': _g[-3]} for _ix, _g in enumerate(model_data['gencost'])}
    generator_dict = merge_nested_dicts(generator_dict1, generator_dict2)
    # power demand is occaisonally negative
    # load_dict = {int(_l[0]): {'l_min': 0, 'l_upper': np.abs(_l[2]), 'cost': 250} for _l in model_data['bus']}
    load_dict = {int(_l[0]): {'l_min': np.abs(_l[2]*0.5), 'l_upper': np.abs(_l[2]), 'cost': 1e4} for _l in model_data['bus']}
    _fmax = 20 * np.max([np.max(_b[5:7]) for _b in model_data['branch']])

    flow_dict = {}
    for _ix, _b in enumerate(model_data['branch']):
        _fx = np.max(_b[5:7])
        flow_dict[_ix] = {'r': _b[2], 'x': _b[3], 'b': _b[4], 'g': _b[2] / (_b[2]**2 + _b[3]**2)}
        if model_ui.value == 'case14.py':
            flow_dict[_ix]['f_max'] = 175
        elif _fx > 0:
            flow_dict[_ix]['f_max'] = _fx
        else:
            flow_dict[_ix]['f_max'] = _fmax
    line_resistance = model_data['branch'][:, 2]
    return flow_dict, generator_dict, load_dict


@app.cell(hide_code=True)
def _(model_data, np):
    # label nodes
    generator_nodes = model_data['gen'][:, 0]
    load_nodes = np.where(model_data['bus'][:, 2] != 0)[0] + 1
    set_gen_nodes = set(generator_nodes)
    set_load_nodes = set(load_nodes)
    gen_only = np.asarray(list(set_gen_nodes.difference(set_load_nodes)))
    load_only = np.asarray(list(set_load_nodes.difference(set_gen_nodes)))
    both = np.asarray(list(set_gen_nodes.intersection(set_load_nodes)))
    return both, gen_only, load_only


@app.cell
def _(model_ui):
    model_ui
    return


@app.cell(hide_code=True)
def _(fig_heatmap, generator_dict, graph, load_dict, mo):
    mo.ui.tabs({
        'graph': mo.mermaid(graph), 
        'incidence matrix': fig_heatmap, 
        'generators': mo.accordion({str(_k): _v for _k, _v in generator_dict.items()}),
        'loads': mo.accordion({str(_k): _v for _k, _v in load_dict.items()})
    })
    return


@app.cell
def _(flow_limits, gen_limits, load_limits, mo):
    mo.hstack([load_limits, gen_limits, flow_limits])
    return


@app.cell
def _(B, flow_dict, generator_dict, load_dict, make_problem):
    problem = make_problem(B, generator_dict, load_dict, flow_dict)
    return (problem,)


@app.cell(hide_code=True)
def _(flow_limits, gen_limits, load_limits, mo, np, problem):
    # solve with updated parameters
    am_solving = True
    problem.param_dict['g_min'].value = [_v[0] for _v in gen_limits.value]
    problem.param_dict['g_max'].value = [_v[1] for _v in gen_limits.value]
    problem.param_dict['l_min'].value = [_v[0] for _v in load_limits.value]
    problem.param_dict['l_upper'].value = [_v[1] for _v in load_limits.value]
    problem.param_dict['line capacities'].value = flow_limits.value
    obj_val = problem.solve(verbose=True, solver='CLARABEL')
    constrained_lines = np.where(~np.isclose(problem.constraints[2].dual_value, 0, atol=1e-2))[0]
    mo.md(f'objective value: {obj_val:.2f}')
    return am_solving, constrained_lines


@app.cell
def _(bus_power_fig, mo, solution, solution2):
    mo.ui.tabs({
        'bus power': bus_power_fig,
        'line flows': mo.mermaid(solution),
        'line flows ordered': mo.mermaid(solution2)
    })
    return


@app.cell(hide_code=True)
def _(generator_dict, mo, np):
    # generator ui
    _max = np.ceil(1.5*np.max([_v['p_max'] for _k, _v in generator_dict.items()]))
    gen_limits = mo.ui.array([mo.ui.range_slider(start=0, 
                                    stop=_max, 
                                    label=f'gen {_ix}', 
                                    value=[_gen['p_min'], _gen['p_max']],
                                    debounce=True,
                                    show_value=True) 
                 for _ix, _gen in generator_dict.items()], label='generator limits')
    return (gen_limits,)


@app.cell(hide_code=True)
def _(load_dict, mo, np):
    # load ui
    _max = np.ceil(1.5*np.max([_v['l_upper'] for _k, _v in load_dict.items()]))
    load_limits = mo.ui.array([mo.ui.range_slider(start=0, 
                                    stop=_max, 
                                    label=f'load {_ix}', 
                                    value=[_load['l_min'], _load['l_upper']],
                                    debounce=True,
                                    show_value=True,) 
                 for _ix, _load in load_dict.items()], label='load limits')
    return (load_limits,)


@app.cell(hide_code=True)
def _(flow_dict, mo, np):
    # flow ui
    _max = np.ceil(1.5*np.max([_v['f_max'] for _k, _v in flow_dict.items()]))
    flow_limits = mo.ui.array([mo.ui.slider(start=0, 
                                    stop=_max, 
                                    label=f'line {_ix+1}', 
                                    value=_line['f_max'],
                                    debounce=True,
                                    show_value=True,) 
                 for _ix, _line in flow_dict.items()], label='flow limits')
    return (flow_limits,)


@app.cell(hide_code=True)
def _(
    am_solving,
    gen_limits,
    generator_dict,
    load_limits,
    np,
    num_buses,
    plt,
    problem,
):
    # plot bus generators and loads
    am_solving
    _fig, _ax = plt.subplots(nrows=3, sharex=True, figsize=(9, 5.5))
    _ax[0].stem(np.arange(num_buses)+1, problem.var_dict['p_gen'].value, label='gen')
    _ax[0].plot((np.arange(num_buses)+1)[np.array(list(generator_dict.keys()))-1], [_v[1] for _v in gen_limits.value], 
                ls='none', marker=7, color='orange', label='max')
    _ax[0].plot((np.arange(num_buses)+1)[np.array(list(generator_dict.keys()))-1], [_v[0] for _v in gen_limits.value], 
                ls='none', marker=6, color='red', label='min')
    _ax[0].legend()
    _ax[0].set_title('generator production')
    _ax[0].set_ylabel('power [MW]')
    _ax[1].stem(np.arange(num_buses)+1, problem.var_dict['p_load'].value, label='served')
    _ax[1].plot(np.arange(num_buses)+1, [_v[1] for _v in load_limits.value], 
                ls='none', marker=7, color='orange', label='desired')
    _ax[1].plot(np.arange(num_buses)+1, [_v[0] for _v in load_limits.value], 
                ls='none', marker=6, color='red', label='min')
    _ax[1].legend()
    _ax[1].set_title('load served')
    _ax[1].set_ylabel('power [MW]')
    _ax[2].scatter(np.arange(num_buses)+1, -1 * problem.constraints[0].dual_value)
    _ax[2].set_title('node prices')
    _ax[2].set_ylabel('dual value (price)')
    _ax[2].set_xlabel('bus number')
    plt.tight_layout()
    bus_power_fig = _fig
    return (bus_power_fig,)


@app.cell(hide_code=True)
def _(
    am_solving,
    both,
    constrained_lines,
    gen_only,
    lines,
    load_only,
    problem,
):
    # power flow graph view 1
    am_solving
    solution2 = ["flowchart LR"]
    for _ix, _l in enumerate(lines):
        _f = _l[0]
        _t = _l[1]
        _lf = problem.var_dict['line flows'][_ix].value
        if _lf >= 0:
            solution2.append(f"  {_f:.0f}(({_f})) == {_lf:.2f} ==> {_t:.0f}(({_t}))")
        else:
            # solution2.append(f"  {_f:.0f}(({_f})) ~~~ {_t:.0f}(({_t}))")
            solution2.append(f"  {_t:.0f}(({_t})) == {-_lf:.2f} ==> {_f:.0f}(({_f}))")
    solution2 = "\n".join(solution2)
    solution2 += '''\n
        classDef gen stroke-dasharray: 5 5;\n
        classDef load fill:#f9f;\n
        classDef genload stroke-dasharray: 5 5,fill:#f9f;'''
    if len(load_only) > 0:
        _str = ",".join([str(int(_i)) for _i in load_only])
        solution2 += "\n    class "+_str+" gen;"
    if len(gen_only) > 0:
        _str = ",".join([str(int(_i)) for _i in gen_only])
        solution2 += "\n    class "+_str+" load;"
    if len(both) > 0:
        _str = ",".join([str(int(_i)) for _i in both])
        solution2 += "\n    class "+_str+" genload;"
    if len(constrained_lines) > 0:
        _str = ",".join([str(int(_i)) for _i in constrained_lines])
        solution2 += "\n linkStyle "+_str+" color:red;"
    return (solution2,)


@app.cell
def _(solution):
    solution
    return


@app.cell(hide_code=True)
def _(
    am_solving,
    both,
    constrained_lines,
    gen_only,
    lines,
    load_only,
    problem,
):
    # power flow graph view 2
    am_solving
    solution = ["flowchart LR"]
    for _ix, _l in enumerate(lines):
        _f = _l[0]
        _t = _l[1]
        _lf = problem.var_dict['line flows'][_ix].value
        solution.append(f"  {_f:.0f}(({_f})) == {_lf:.2f} ==> {_t:.0f}(({_t}))")
        # if _lf >= 0:
        #     solution.append(f"  {_f:.0f}(({_f})) == {_lf:.2f} ==> {_t:.0f}(({_t}))")
        # else:
        #     solution.append(f"  {_t:.0f}(({_t})) == {-_lf:.2f} ==> {_f:.0f}(({_f}))")
    solution = "\n".join(solution)
    solution += '''
        classDef gen stroke-dasharray: 5 5;
        classDef load fill:#f9f;
        classDef genload stroke-dasharray: 5 5,fill:#f9f;'''
    if len(load_only) > 0:
        _str = ",".join([str(int(_i)) for _i in load_only])
        solution += "\n    class "+_str+" gen;"
    if len(gen_only) > 0:
        _str = ",".join([str(int(_i)) for _i in gen_only])
        solution += "\n    class "+_str+" load;"
    if len(both) > 0:
        _str = ",".join([str(int(_i)) for _i in both])
        solution += "\n    class "+_str+" genload;"
    if len(constrained_lines) > 0:
        _str = ",".join([str(int(_i)) for _i in constrained_lines])
        solution += "\n linkStyle "+_str+" color:red;"
    return (solution,)


@app.cell
def _():
    import marimo as mo
    import os, re
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import seaborn as sns
    return cp, mo, np, os, plt, re, sns


if __name__ == "__main__":
    app.run()
