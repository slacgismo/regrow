import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""In this version, we take the log of the load data before model fitting. This performs slightly better, suggesting the components are multiplicative rather than additive.""")
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
        df_out = pd.concat(df_list, axis=0)
        # df_out = df.groupby(df.index).first()
        return df_out
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
def _(df, sns):
    sns.heatmap(df["RT_Demand"].values.reshape((24, -1), order='F'))
    return


@app.cell
def _():
    (1096 - 1) / 3
    return


@app.cell
def _(df):
    df["RT_Demand"].values.reshape((24, -1), order='F').shape
    return


@app.cell
def _(mo):
    mo.md("""## Set up linear regressors""")
    return


@app.cell
def _(
    make_H,
    make_basis_matrix,
    make_offset_H,
    make_regularization_matrix,
    np,
    x,
    y,
):
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
    H0 = make_H(x, knots, include_offset=False)
    Hm1 = make_offset_H(H0, -1)
    Hm2 = make_offset_H(H0, -2)
    Hm3 = make_offset_H(H0, -3)
    Hp1 = make_offset_H(H0, 1)
    Hp2 = make_offset_H(H0, 2)
    Hp3 = make_offset_H(H0, 3)
    Hs = [Hm3, Hm2, Hm1, H0, Hp1, Hp2, Hp3]
    first_use_set = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)
    return F, H0, Hs, Wf, first_use_set, knots


@app.cell
def _(df, np):
    y = np.log(df.loc["2020":"2021"]["RT_Demand"])
    x = df.loc["2020":"2021"]["Dry_Bulb"]
    return x, y


@app.cell
def _(df):
    len(df)
    return


@app.cell
def _(df):
    len(df.loc["2020":"2021"]) + len(df.loc["2022"])
    return


@app.cell
def _(df, np):
    np.arange(len(df.loc["2020":"2021"]))
    return


@app.cell
def _(df, np):
    len(df.loc["2020":"2021"]) + np.arange(len(df.loc["2022"]))
    return


@app.cell
def _(df):
    len(df.loc["2020"])
    return


@app.cell
def _(df):
    len(df.loc["2021"])
    return


@app.cell
def _(df, make_features, np):
    training_config = {
        'time_idxs': np.arange(len(df.loc["2020":"2021"])),
        'temps': df.loc["2020":"2021", "Dry_Bulb"].values,
        'num_harmonics': [6, 4, 3],
        'periods': [365.2425 * 24, 7 * 24, 24],
        'length': len(df.loc["2020":"2021"]),
        'num_knots': 10,
        'temp_window': 3
    
    }
    test_config = {
        'time_idxs': len(df.loc["2020":"2021"]) + np.arange(len(df.loc["2022"])),
        'temps': df.loc["2022", "Dry_Bulb"].values,
        'num_harmonics': [6, 4, 3],
        'periods': [365.2425 * 24, 7 * 24, 24],
        'length': len(df.loc["2022"]),
        'num_knots': 10,
        'temp_window': 3
    
    }
    train_feat = make_features(**training_config)
    test_feat = make_features(**test_config)
    return test_feat, train_feat


@app.cell
def _(df, fit_baseline, train_feat):
    baseline_fit = fit_baseline(df.loc["2020":"2021", "RT_Demand"].values, train_feat)
    return (baseline_fit,)


@app.cell
def _(baseline_fit, predict, test_feat):
    test_predict = predict(test_feat, baseline_fit['time_param'], baseline_fit['temp_param'], take_log=True)
    return (test_predict,)


@app.cell
def _(baseline_fit, run_baseline_fit_analysis):
    run_baseline_fit_analysis(baseline_fit['exp_target'], baseline_fit['exp_predict'])
    return


@app.cell
def _(df, run_test_analysis, test_feat, test_predict):
    run_test_analysis(df.loc["2022", "RT_Demand"][test_feat['use_set']], test_predict)
    return


@app.cell
def _(baseline_fit, plt):
    plt.plot(baseline_fit['exp_target'], baseline_fit['exp_predict'], marker='.', linewidth=.5, alpha=0.4)
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.gcf()
    return


@app.cell
def _(df, plt, test_feat, test_predict):
    plt.plot(df.loc["2022", "RT_Demand"][test_feat['use_set']], test_predict, marker='.', linewidth=.5, alpha=0.4)
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.gcf()
    return


@app.cell
def _(df, mo, plt, test_feat, test_predict):
    _ixs = df.loc["2022", "RT_Demand"].index[test_feat['use_set']]
    plt.plot(_ixs, df.loc["2022", "RT_Demand"][test_feat['use_set']].values, linewidth=1, label='actual')
    plt.plot(_ixs, test_predict, linewidth=1, label='predicted')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    make_H,
    make_basis_matrix,
    make_offset_H,
    make_regularization_matrix,
    np,
):
    def make_features(time_idxs, temps, num_harmonics, periods, length, num_knots, temp_window=3):
        # Fourier time features
        F = make_basis_matrix(
            num_harmonics=num_harmonics,
            length=np.max(time_idxs) + 1,
            periods=periods
        )
        F = F[time_idxs]
        # weight matrix for regularized Fourier parameters
        W = make_regularization_matrix(
            num_harmonics=num_harmonics,
            weight=1,
            periods=periods
        )
        # temperature features
        nK = num_knots
        knots = np.linspace(np.min(temps), np.max(temps), nK)
        H0 = make_H(temps, knots, include_offset=False)
        if temp_window > 0:
            Hs = np.asarray([make_offset_H(H0, _ix) for _ix in np.arange(start=-temp_window, stop=temp_window+1)])
            first_use_set = np.all(np.all(~np.isnan(np.asarray(Hs)), axis=-1), axis=0)
        else:
            Hs = H0
            first_use_set = np.all(~np.isnan(np.asarray(Hs)), axis=-1)
        out = {
            'fourier_mat': F,
            'weight_mat': W,
            'temp_mats': Hs,
            'use_set': first_use_set
        }
        return out
    return (make_features,)


@app.cell
def _(H0, cvx, first_use_set, length, np):
    def fit_baseline(targets, model_features, take_log=True, use_time=True, use_temp=True, use_temp_window=True, 
                     time_reg=1e-4, temp_reg=1e-4, window_reg=1e0):
        F = model_features['fourier_mat']
        Wf = model_features['weight_mat']
        Hs = model_features['temp_mats']
        use_set = model_features['use_set']
        if take_log:
            y = np.log(targets)
        else:
            y = targets
        a = cvx.Variable(F.shape[1]) # coefficients for time features
        b = cvx.Variable((Hs.shape[-1], Hs.shape[0])) # coefficients for temperature features
        if use_temp_window:
            _s = first_use_set
            if use_temp:
                temp = cvx.sum([H[_s] @ b[:, _ix] for _ix, H in enumerate(Hs)])
            else:
                temp = cvx.sum([H[_s] @ b[:, _ix] for _ix, H in enumerate(Hs) if _ix != 3])
        else:
            _s = np.ones(length, dtype=bool)
            if use_temp:
                temp = H0[_s] @ b[:, 3]
            else:
                temp = 0
        if use_time:
            time = F[_s, 1:] @ a[1:]
        else:
            time = 0
        error = cvx.sum_squares(y[_s] - time - temp - a[0]) / np.sum(_s)
        regularization = (time_reg * cvx.sum_squares(Wf @ a) 
                          + temp_reg * cvx.sum_squares(b) 
                          + window_reg * cvx.sum_squares(cvx.diff(b, axis=1)))
        problem = cvx.Problem(cvx.Minimize(error + regularization))
        problem.solve(solver='CLARABEL', verbose=False)
        model_predict = (a[0] + time + temp).value
        output = {
            'predict': model_predict,
            'residuals': y[_s] - model_predict,
            'time_param': a.value,
            'temp_param': b.value,
            'target': y[_s]
        }
        if take_log:
            output['exp_predict'] = np.exp(model_predict)
            output['exp_target'] = np.exp(y[_s])
        return output
    return (fit_baseline,)


@app.cell
def _(np):
    def predict(features, time_param, temp_param, take_log=True):
        F = features['fourier_mat']
        Wf = features['weight_mat']
        Hs = features['temp_mats']
        use_set = features['use_set']
        time = F[use_set] @ time_param
        temp =  np.sum([H[use_set] @ temp_param[:, _ix] for _ix, H in enumerate(Hs)], axis=0)
        model_predict = time + temp
        if take_log:
            model_predict = np.exp(model_predict)
        return model_predict
    return (predict,)


@app.cell
def _(mo, np, r2_score):
    def run_baseline_fit_analysis(actual, predicted, return_method='markdown'):
        residual = actual - predicted
        rmse = np.sqrt(np.average(np.power(residual, 2)))
        rmse_per = rmse * 100 / np.nanmean(actual)
        r2 = r2_score(actual, predicted)
        if return_method == 'markdown':
            rms_string = f"RMS error of model fit: {rmse:.2f}, or {rmse_per:.2f}% of average"
            r2_string = f"R2 is {r2:.3f}"
            return mo.md(rms_string+'\n\n'+r2_string)
        elif return_method == 'dict':
            out = {
                'rmse': rmse,
                'rmse percent': rmse_per,
                'r2': r2
            }
            return out

    def run_test_analysis(actual, predicted, return_method='markdown'):
        residual = actual - predicted
        rmse = np.sqrt(np.average(np.power(residual, 2)))
        rmse_per = rmse * 100 / np.nanmean(actual)
        if return_method == 'markdown':
            rms_string = f"RMS error on test data: {rmse:.2f}, or {rmse_per:.2f}% of average"
            return mo.md(rms_string)
        elif return_method == 'dict':
            out = {
                'rmse': rmse,
                'rmse percent': rmse_per
            }
            return out
    return run_baseline_fit_analysis, run_test_analysis


@app.cell
def _(F, H0, Hs, Wf, cvx, first_use_set, np, y):
    a = cvx.Variable(F.shape[1]) # coefficients for time features
    c = cvx.Variable((H0.shape[1], len(Hs))) # coefficients for temperature features
    _s = first_use_set
    temp = cvx.sum([H[_s] @ c[:, _ix] for _ix, H in enumerate(Hs)])
    error = cvx.sum_squares(y.values[_s] - F[_s] @ a - temp) / np.sum(_s)
    regularization = 1e-4 * cvx.sum_squares(Wf @ a) + 1e-4 * cvx.sum_squares(c) + 1e0 * cvx.sum_squares(cvx.diff(c, axis=1))
    problem = cvx.Problem(cvx.Minimize(error + regularization))
    problem.solve(solver='CLARABEL', verbose=False)
    model = (F[_s] @ a + temp).value
    return a, c, model, temp


@app.cell
def _(first_use_set, mo, model, np, y):
    rmse = np.sqrt(np.average(np.power(np.exp(y.values[first_use_set]) - np.exp(model), 2)))
    mo.md(f"RMS error of model fit: {rmse:.2f}, or {rmse * 100 / np.nanmean(np.exp(y)):.2f}% of average")
    return


@app.cell
def _(a, c, first_use_set, mo, model, np, r2_score, y):
    r2 = r2_score(np.exp(y.values[first_use_set]), np.exp(model))
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) + len(a[1:].value) + len(c.value) - 1)
    mo.vstack([f"R2: {r2:.3f}", f"R2-adj: {r2_adj:.3f}"])
    return


@app.cell
def _(Hs, a, c, plt):
    _fig, _ax = plt.subplots(nrows=len(Hs)+1, sharey=False, figsize=(6, 7))
    _ax[0].stem(a[1:].value)
    for _ix in range(len(Hs)):
        _ax[_ix+1].stem(c[:, _ix].value)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(F, a, df, first_use_set, mo, model, np, plt, temp, y):
    _ix = df.loc["2020":"2021"].index
    plt.plot(_ix, np.exp(F @ a.value) - np.average(np.exp(F @ a.value)) + np.average(np.exp(y)), linewidth=1, color='grey', label='time only')
    plt.plot(_ix[first_use_set], np.exp(temp.value + a[0].value), linewidth=1, color='grey', label='temp only', ls=':')
    plt.plot(_ix, np.exp(y.values), label='actual', ls='--')
    plt.plot(_ix[first_use_set], np.exp(model), label='predicted')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(first_use_set, model, plt, y):
    plt.plot(y.values[first_use_set], model, marker='.', linewidth=1, alpha=.4)
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
def _(first_use_set, model, plt, y):
    baseline_residuals = y.values[first_use_set] - model
    plt.hist(baseline_residuals, bins=100)
    plt.title('Distribution of baseline residuals')
    plt.gcf()
    return


@app.cell
def _(temp):
    temp.value
    return


@app.cell
def _(a, df, first_use_set, np, plt, temp, x, y):
    x_sort = np.sort(x.values)
    plt.scatter(df.loc["2020":"2021"]['Dry_Bulb'].values, np.exp(y), marker='.',
                label='data', s=10, alpha=.5, color='orange')
    # plt.plot(x_sort, np.exp((make_H(x_sort, knots) @ c).value + a[0].value), label='temperature response')
    plt.plot(x.values[first_use_set], np.exp(temp.value + a[0].value), label='temperature response', marker='.', ls='none')
    plt.title('Inferred temperature dependence')
    plt.legend()
    return


@app.cell
def _(mo):
    mo.md("""## Generate data for third year""")
    return


@app.cell
def _(np):
    np.sum([np.arange(5) + _ix for _ix in range(12)], axis=0)
    return


@app.cell
def _(make_H, make_basis_matrix, make_offset_H, np, stats):
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

    def predict_baseline(time_idxs, temp_data, time_coeff, temp_coeff, knots, model='all'):
        F = make_basis_matrix(
            num_harmonics=[6, 4, 3],
            length=max(time_idxs) +1,
            periods=[365.2425 * 24, 7 * 24, 24]
        )
        F = F[time_idxs]
        H0 = make_H(temp_data, knots, include_offset=False)
        Hm1 = make_offset_H(H0, -1)
        Hm2 = make_offset_H(H0, -2)
        Hm3 = make_offset_H(H0, -3)
        Hp1 = make_offset_H(H0, 1)
        Hp2 = make_offset_H(H0, 2)
        Hp3 = make_offset_H(H0, 3)
        Hs = [Hm3, Hm2, Hm1, H0, Hp1, Hp2, Hp3]
        temp = np.sum([H @ temp_coeff[:, _ix] for _ix, H in enumerate(Hs)], axis=0)
        if model == 'all':
            baseline = F @ time_coeff + temp
        elif model == 'time':
            baseline = F @ time_coeff
        elif model == 'temp':
            baseline = temp + time_coeff[0]
        return np.exp(baseline)
    return predict_baseline, roll_out_ar_noise


@app.cell
def _(a, c, df, knots, np, predict_baseline):
    new_idx = np.arange(np.sum(df['year'] == 2022)) + np.sum(df['year'] != 2022) - 1
    new_baseline = predict_baseline(new_idx, df.loc["2022"]["Dry_Bulb"].values, a.value, c.value, knots)
    return new_baseline, new_idx


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
    return


@app.cell
def _(df, mo, new_baseline, np):
    test_mae = np.nanmean(np.abs(df.loc["2022"]["RT_Demand"].values - new_baseline))
    mo.md(f"test MAE: {test_mae:.2f}, or {100*test_mae / np.nanmean(df.loc["2022"]["RT_Demand"].values):.2f}% of average")
    return


@app.cell
def _(df, new_baseline, new_noise, np, pd):
    ppower_actual = np.nanmax(df.loc["2022"]["RT_Demand"].values)
    ppower_predict = np.nanmax(new_baseline * new_noise)
    ppower_predict_noar = np.nanmax(new_baseline)
    ppower_time_actual = np.nanargmax(df.loc["2022"]["RT_Demand"].values)
    ppower_time_predict =  np.nanargmax(new_baseline * new_noise)
    ppower_time_predict_noar =  np.nanargmax(new_baseline)
    _index = ["actual", 'predicted', 'predicted no AR model']
    _data = {
        "peak power": [ppower_actual, ppower_predict, ppower_predict_noar],
        "index of peak": [ppower_time_actual, ppower_time_predict, ppower_time_predict_noar]
    }
    pd.DataFrame(data=_data, index=_index)
    return


@app.cell
def _(df, new_baseline, plt):
    plt.plot(df.loc["2022"]["RT_Demand"].values, new_baseline, marker='.', linewidth=1, alpha=.4, label='true')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.title("Holdout year (2022)")
    # plt.legend()
    plt.gcf()
    return


@app.cell
def _(mo):
    loc_sldr = mo.ui.slider(-10, 10, 0.1, value=0, label='location adjustment')
    scale_sldr = mo.ui.slider(.1, 3, 0.1, value=1, label='scale adjustment')
    return loc_sldr, scale_sldr


@app.cell
def _(df, new_baseline, np, plt):
    plt.hist(df.loc["2022"]["RT_Demand"].values - new_baseline, bins=100, label='true residuals', density=True)
    plt.axvline(np.nanmedian(df.loc["2022"]["RT_Demand"].values - new_baseline), ls='--')
    plt.axvline(np.nanquantile(df.loc["2022"]["RT_Demand"].values - new_baseline, 0.95), ls=':')
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(
    a,
    c,
    df,
    knots,
    mo,
    new_baseline,
    new_idx,
    new_noise,
    np,
    plt,
    predict_baseline,
    y,
):
    _ix = df.loc["2022"].index
    time_only = predict_baseline(
        new_idx, df.loc["2022"]["Dry_Bulb"].values, a.value, c.value, knots, model='time'
    )
    plt.plot(_ix, time_only - np.average(time_only) + np.average(np.exp(y)), linewidth=1, color='grey', label='time only')
    temp_only = predict_baseline(
        new_idx, df.loc["2022"]["Dry_Bulb"].values, a.value, c.value, knots, model='temp'
    )
    plt.plot(_ix, temp_only, linewidth=1, color='grey', label='temp only', ls=':')
    plt.plot(_ix, df.loc["2022"]["RT_Demand"].values, label='actual', ls='--')
    plt.plot(_ix, new_baseline, label='full model')
    plt.plot(_ix, new_baseline*new_noise, label='noisy sample', marker='.', linewidth=.7, markersize=2, alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
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
    return


@app.cell
def _(np):
    def make_offset_H(H, offset):
        newH = np.copy(H)
        newH = np.roll(newH, -offset, axis=0)
        if offset > 0:
            newH[-offset:] = np.nan
        elif offset < 0:
            newH[:-offset] = np.nan
        return newH
    return (make_offset_H,)


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
        sns,
        stats,
    )


if __name__ == "__main__":
    app.run()
