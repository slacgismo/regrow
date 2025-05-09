import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md('In this version, we take the log of the load data before model fitting. This performs slightly better, suggesting the components are multiplicative rather than additive.')
    return


@app.cell
def _(mo):
    SHEETS = [
        "ISO NE CA",
        "ME",
        "NH",
        "VT",
        "CT",
        "RI",
        "SEMA",
        "WCMA",
        "NEMA"
    ]
    sheet_slct = mo.ui.dropdown(options=SHEETS, value='RI', label='select data')
    return (sheet_slct,)


@app.cell
def _(Path, mo, pd):
    @mo.cache
    def make_data(sheet='RI'):
        years = [2020, 2021, 2022]
        # years = [2020]
        df_list = []
        for _yr in years:
            fp = Path('.') / 'NE_ISO_Data' / f'{_yr}_smd_hourly.xlsx' 
            df = pd.read_excel(fp, sheet_name=sheet)
            df['year'] = _yr
            df.index = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hr_End'].map(lambda x: f"{x-1}:00:00")) + pd.Timedelta(hours=1)
            df_list.append(df)
        return pd.concat(df_list, axis=0)
    return (make_data,)


@app.cell
def _(make_data, sheet_slct):
    df = make_data(sheet=sheet_slct.value)
    return (df,)


@app.cell
def _(mo):
    mo.md("""## Select Data""")
    return


@app.cell
def _(sheet_slct):
    sheet_slct
    return


@app.cell
def _(tabs):
    tabs
    return


@app.cell
def _(df, np):
    y = np.log(df.loc["2020":"2021"]["RT_Demand"])
    x = df.loc["2020":"2021"]["Dry_Bulb"]
    return x, y


@app.cell
def _(mo):
    mo.md("""## Set up linear regressors""")
    return


@app.cell
def _(make_H, make_basis_matrix, make_regularization_matrix, np, x, y):
    # mult-Fourier basis matrix with cross terms
    nharmon = [6, 4, 3]
    F = make_basis_matrix(
        num_harmonics=nharmon,
        length=len(y),
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    # weight matrix for regularized Fourier parameters
    Wf = make_regularization_matrix(
        num_harmonics=nharmon,
        weight=1,
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    # Temperature terms
    nK = 10
    knots = np.linspace(np.min(x), np.max(x), nK)
    H = make_H(x, knots, include_offset=False)
    return F, H, Wf, knots


@app.cell
def _(mo):
    mo.md("""## Fit baseline model with time and temperature features""")
    return


@app.cell
def _(F, H, Wf, cvx, y):
    a = cvx.Variable(F.shape[1]) # coefficients for time features
    c = cvx.Variable(H.shape[1]) # coefficients for temperature features
    error = cvx.sum_squares(y.values - F @ a - H @ c) / F.shape[0]
    regularization = 1e-4 * cvx.sum_squares(Wf @ a) + 1e-4 * cvx.sum_squares(c)
    problem = cvx.Problem(cvx.Minimize(error + regularization))
    problem.solve(verbose=False)
    model = (F @ a + H @ c).value
    return a, c, model


@app.cell
def _(mo, model, np, y):
    rmse = np.sqrt(np.average(np.power(np.exp(y.values) - np.exp(model), 2)))
    mo.md(f"RMS error of model fit: {rmse:.2f}, or {rmse * 100 / np.nanmean(np.exp(y)):.2f}% of average")
    return


@app.cell
def _(a, c, mo, model, np, r2_score, y):
    r2 = r2_score(np.exp(y.values), np.exp(model))
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) + len(a[1:].value) + len(c.value) - 1)
    mo.vstack([f"R2: {r2:.3f}", f"R2-adj: {r2_adj:.3f}"])
    return


@app.cell
def _(a, c, plt):
    _fig, _ax = plt.subplots(nrows=2, sharey=False)
    _ax[0].stem(a[1:].value)
    _ax[1].stem(c.value)
    plt.gcf()
    return


@app.cell
def _(F, H, a, c, mo, model, np, plt, y):
    plt.plot(np.exp(F @ a.value) - np.average(np.exp(F @ a.value)) + np.average(np.exp(y)), linewidth=1, color='grey', label='time only')
    plt.plot(np.exp(H @ c.value + a[0].value), linewidth=1, color='grey', label='temp only', ls=':')
    plt.plot(np.exp(y.values), label='actual', ls='--')
    plt.plot(np.exp(model), label='predicted')
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(model, plt, y):
    plt.plot(y.values, model, marker='.', linewidth=1, alpha=.4)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.gcf()
    return


@app.cell
def _(model, plt, y):
    baseline_residuals = y.values - model
    plt.hist(baseline_residuals, bins=100)
    plt.title('Distribution of baseline residuals')
    plt.gcf()
    return (baseline_residuals,)


@app.cell
def _(y):
    y.shape
    return


@app.cell
def _(a, c, df, knots, make_H, np, plt, x, y):
    x_sort = np.sort(x.values)
    plt.scatter(df.loc["2020":"2021"]['Dry_Bulb'].values, np.exp(y), marker='.',
                label='data', s=10, alpha=.5, color='orange')
    plt.plot(x_sort, np.exp((make_H(x_sort, knots) @ c).value + a[0].value), label='temperature response')
    plt.title('Inferred temperature dependence')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md("""## Fit AR model to baseline residuals""")
    return


@app.cell
def _(baseline_residuals, cvx, np, running_view):
    B = running_view(baseline_residuals, 36)
    # usable records with AR lags
    use_set = np.all(~np.isnan(B), axis=1)
    theta = cvx.Variable(B.shape[1])
    constant = cvx.Variable()
    problem2 = cvx.Problem(
        cvx.Minimize(cvx.sum_squares(baseline_residuals[use_set] - B[use_set] @ theta - constant)),
        [cvx.norm1(theta) <= 0.95]
    )
    problem2.solve(solver='CLARABEL')
    return B, constant, theta, use_set


@app.cell
def _(cvx, mo, theta):
    mo.md(f"""sum-abs of AR coefficients: {cvx.norm1(theta).value:.2f}""")
    return


@app.cell
def _(ar_model, baseline_residuals, mo, np, use_set, y):
    mo.md(
        f"""
    Baseline MAE: {np.average(np.abs(baseline_residuals)):.2f}, or {np.average(np.abs(baseline_residuals)) * 100 / np.nanmean(y):.2f}% of average\n
    Autoregressive MAE: {np.average(np.abs(baseline_residuals[use_set] - ar_model)):.2f}, or {np.average(np.abs(baseline_residuals[use_set] - ar_model)) * 100 / np.nanmean(y):.2f}% of average
    """
    )
    return


@app.cell
def _(plt, theta):
    plt.stem(theta.value[::-1])
    plt.title('AR coefficients')
    plt.gcf()
    return


@app.cell
def _(np, plt, theta):
    plt.stem(np.abs(theta.value[::-1]))
    plt.yscale('log')
    plt.title('magnitude of AR coefficients')
    plt.gcf()
    return


@app.cell
def _(B, constant, theta, use_set):
    ar_model = (B[use_set] @ theta + constant).value
    return (ar_model,)


@app.cell
def _(ar_model, baseline_residuals, np, plt, use_set):
    plt.plot(baseline_residuals[use_set], ar_model, marker='.', linewidth=1, alpha=.4)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title("AR model of residuals")
    _ix = np.argmax(baseline_residuals[use_set] - ar_model)
    plt.plot([baseline_residuals[use_set][_ix]], [ar_model[_ix]], color='red', markersize=10, marker='.',
            label='largest postive outlier')
    _ix = np.argmin(baseline_residuals[use_set] - ar_model)
    plt.plot([baseline_residuals[use_set][_ix]], [ar_model[_ix]], color='orange', markersize=10, marker='.',
            label='largest negative outlier')
    plt.legend()
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.gcf()
    return


@app.cell
def _(ar_model, baseline_residuals, np, plt, stats, use_set):
    plt.hist(baseline_residuals[use_set] - ar_model, bins=100, density=True)
    _xs = np.linspace(np.min(baseline_residuals[use_set] - ar_model), np.max(baseline_residuals[use_set] - ar_model), 1001)
    lap_loc, lap_scale = stats.laplace.fit(baseline_residuals[use_set] - ar_model)
    plt.plot(_xs, stats.laplace.pdf(_xs, lap_loc, lap_scale), label='laplace fit')
    plt.legend()
    plt.title('Distribution of AR residuals')
    plt.gcf()
    return lap_loc, lap_scale


@app.cell
def _(ar_model, baseline_residuals, plt, sm, stats, use_set):
    _pplot = sm.ProbPlot(baseline_residuals[use_set] - ar_model, stats.laplace, fit=True)
    _fig1 = _pplot.ppplot(line="45")
    _h = plt.title("P-P plot for Laplace")
    _fig1

    return


@app.cell
def _(ar_model, baseline_residuals, plt, sm, stats, use_set):
    _pplot = sm.ProbPlot(baseline_residuals[use_set] - ar_model, stats.norm, fit=True)
    _fig1 = _pplot.ppplot(line="45")
    _h = plt.title("P-P plot for Normal")
    _fig1
    return


@app.cell
def _(ar_model, baseline_residuals, plt, sm, stats, use_set):
    _pplot = sm.ProbPlot(baseline_residuals[use_set] - ar_model, stats.laplace, fit=True)
    _fig1 = _pplot.qqplot(line="45")
    _h = plt.title("Q-Q plot for Laplace")
    _fig1

    return


@app.cell
def _(ar_model, baseline_residuals, plt, sm, stats, use_set):
    _pplot = sm.ProbPlot(baseline_residuals[use_set] - ar_model, stats.norm, fit=True)
    _fig1 = _pplot.qqplot(line="45")
    _h = plt.title("Q-Q plot for Normal")
    _fig1

    return


@app.cell
def _(mo):
    mo.md("""## Generate data for third year""")
    return


@app.cell
def _(make_H, make_basis_matrix, np, stats):
    def roll_out_ar_noise(length, ar_coeff, intercept, loc, scale, random_state=None):
        window = stats.laplace.rvs(loc=loc, scale=scale, size=len(ar_coeff), random_state=random_state)
        nvals = length+len(ar_coeff) * 2
        gen_data = np.empty(nvals, dtype=float)
        for it in range(nvals):
            new_val = ar_coeff @ window + intercept + stats.laplace.rvs(loc=loc, scale=scale, random_state=random_state)
            gen_data[it] = new_val
            new_window = np.roll(window, -1)
            new_window[-1] = new_val
            window = new_window
        return np.exp(gen_data[-length:])

    def predict_baseline(time_idxs, temp_data, time_coeff, temp_coeff, knots):
        F = make_basis_matrix(
            num_harmonics=[6, 4, 3],
            length=max(time_idxs) +1,
            periods=[365.2425 * 24, 7 * 24, 24]
        )
        F = F[time_idxs]
        H = make_H(temp_data, knots, include_offset=False)
        baseline = F @ time_coeff + H @ temp_coeff
        return np.exp(baseline)
    return predict_baseline, roll_out_ar_noise


@app.cell
def _(a, c, df, knots, np, predict_baseline):
    new_idx = np.arange(np.sum(df['year'] == 2022)) + np.sum(df['year'] != 2022) - 1
    new_baseline = predict_baseline(new_idx, df.loc["2022"]["Dry_Bulb"].values, a.value, c.value, knots)
    return (new_baseline,)


@app.cell
def _(
    constant,
    df,
    lap_loc,
    lap_scale,
    loc_sldr,
    np,
    roll_out_ar_noise,
    scale_sldr,
    theta,
):
    new_noise = roll_out_ar_noise(np.sum(df['year'] == 2022), theta.value, constant.value, lap_loc+loc_sldr.value, lap_scale*scale_sldr.value)
    return (new_noise,)


@app.cell
def _(new_baseline, new_noise):
    new_residuals = new_baseline * new_noise - new_baseline
    return (new_residuals,)


@app.cell
def _(df, mo, new_baseline, np):
    test_mae = np.nanmean(np.abs(df.loc["2022"]["RT_Demand"].values - new_baseline))
    mo.md(f"test MAE: {test_mae:.2f}, or {100*test_mae / np.nanmean(df.loc["2022"]["RT_Demand"].values):.2f}% of average")
    return


@app.cell
def _(df, new_baseline, new_noise, np, pd):
    ppower_actual = np.max(df.loc["2022"]["RT_Demand"].values)
    ppower_predict = np.max(new_baseline * new_noise)
    ppower_predict_noar = np.max(new_baseline)
    ppower_time_actual = np.argmax(df.loc["2022"]["RT_Demand"].values)
    ppower_time_predict =  np.argmax(new_baseline * new_noise)
    ppower_time_predict_noar =  np.argmax(new_baseline)
    _index = ["actual", 'predicted', 'predicted no AR model']
    _data = {
        "peak power": [ppower_actual, ppower_predict, ppower_predict_noar],
        "index of peak": [ppower_time_actual, ppower_time_predict, ppower_time_predict_noar]
    }
    pd.DataFrame(data=_data, index=_index)
    return


@app.cell
def _(df, new_baseline, new_noise, plt):
    plt.plot(df.loc["2022"]["RT_Demand"].values, new_baseline, marker='.', linewidth=1, alpha=.4, label='true')
    plt.plot(new_baseline*new_noise, new_baseline, marker='.', linewidth=1, alpha=.4, label='sampled')
    plt.xlabel('realization')
    plt.ylabel('baseline')
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.title("Holdout year (2022)")
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(mo):
    loc_sldr = mo.ui.slider(-10, 10, 0.1, value=0, label='location adjustment')
    scale_sldr = mo.ui.slider(.1, 3, 0.1, value=1, label='scale adjustment')
    return loc_sldr, scale_sldr


@app.cell
def _(df, new_baseline):
    df.loc["2022"]["RT_Demand"].values - new_baseline
    return


@app.cell
def _(df, new_baseline, new_residuals, np, plt):
    plt.hist(df.loc["2022"]["RT_Demand"].values - new_baseline, bins=100, label='true residuals', alpha=0.75, density=True)
    plt.hist(new_residuals, bins=100, label='sampled residuals', alpha=0.75, density=True)
    plt.axvline(np.median(df.loc["2022"]["RT_Demand"].values - new_baseline), ls='--')
    plt.axvline(np.median(new_residuals), ls='--', color='orange')
    plt.axvline(np.quantile(df.loc["2022"]["RT_Demand"].values - new_baseline, 0.95), ls=':')
    plt.axvline(np.quantile(new_residuals, 0.95), ls=':', color='orange')
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(loc_sldr, mo, scale_sldr):
    mo.vstack([loc_sldr, scale_sldr])
    return


@app.cell
def _(df, mo, new_baseline, new_noise, plt):
    plt.plot(new_baseline, label='predicted baseline', linewidth=1, color='red', ls=':')
    plt.plot(df.loc["2022"]["RT_Demand"].values, label='actual', ls='--')
    plt.plot(new_baseline*new_noise, label='sampled')

    plt.legend()
    plt.title("Holdout year (2022)")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(np):
    def running_view(arr, window, axis=-1):
        """
        return a running view of length 'window' over 'axis', nan-padding the start to get the same
        first dimension as the input
        the returned array has an extra last dimension, which spans the window
        """
        mod_arr = np.r_[np.ones(window) * np.nan, arr[:-1]]
        shape = list(mod_arr.shape)
        shape[axis] -= (window-1)
        assert(shape[axis]>0)
        return np.lib.stride_tricks.as_strided(
            mod_arr,
            shape=shape + [window],
            strides=mod_arr.strides + (mod_arr.strides[axis],))
    return (running_view,)


@app.cell
def _(np):
    def d_func(x, k, k_max):
        n1 = np.clip(np.power(x - k, 3), 0, np.inf)
        n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
        d1 = k_max - k
        out = (n1 - n2) / d1
        return out


    def make_H(x, knots, include_offset=False):
        nK = len(knots)
        H = np.ones((len(x), nK), dtype=float)
        H[:, 1] = x
        for _i in range(nK - 2):
            _j = _i + 2
            H[:, _j] = d_func(x, knots[_i], knots[-1]) - d_func(
                x, knots[-2], knots[-1]
            )
        if include_offset:
            return H
        else:
            return H[:, 1:]
    return (make_H,)


@app.cell
def _(df, mo, scatter_fig, ts_fig):
    tabs = mo.ui.tabs({
        'table': df,
        'time series': ts_fig,
        "scatter": scatter_fig
    })
    return (tabs,)


@app.cell
def _(df, plt):
    ts_fig, _ax = plt.subplots(nrows=2, sharex=True)
    df.plot(y='RT_Demand', ax=_ax[0], legend=False)
    df.plot(y='Dry_Bulb', ax=_ax[1], legend=False)
    _ax[0].set_ylabel('realtime demand [MW]')
    _ax[1].set_ylabel('dry bulb temp [deg F]')
    ts_fig = plt.gcf()
    return (ts_fig,)


@app.cell
def _(df, plt):
    for _yr in [2020, 2021, 2022]:
        plt.scatter(df[df['year'] == _yr]['Dry_Bulb'], df[df['year'] == _yr]['RT_Demand'], marker='.',
                    label=_yr, s=10, alpha=.5)
        # df[df['year'] == _yr].plot(x='Dry_Bulb', y='RT_Demand', ls='none', marker='.', legend=False)
    plt.ylabel('realtime demand [MW]')
    plt.xlabel('dry bulb temp [deg F]')
    plt.legend()
    scatter_fig = plt.gcf()
    return (scatter_fig,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import cvxpy as cvx
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.metrics import r2_score
    import statsmodels.api as sm
    from spcqe import make_basis_matrix, make_regularization_matrix
    return (
        Path,
        cvx,
        make_basis_matrix,
        make_regularization_matrix,
        mo,
        np,
        pd,
        plt,
        r2_score,
        sm,
        stats,
    )


if __name__ == "__main__":
    app.run()
