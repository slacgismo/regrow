"""Load modeling library"""

# TODO: parameterization


# mods
# 1. gridlabd uses hour beginning instead of hour-ending (remove +1hr)

# suggestions
# 1. calculate HI from DB & RH and use that for temperature
# 2. add slope to Hs array in both fit and predict
# 3. convert to sklearn fit_predict() implementation

import os
import marimo as mo
import pandas as pd
import numpy as np
import cvxpy as cvx
import datetime as dt
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score
import statsmodels.api as sm
from spcqe import make_basis_matrix, make_regularization_matrix
from load_model import LoadModel

pd.options.display.max_columns=None
pd.options.display.width=None

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
LOCATION_ADJ = 0.0
SCALE_ADJ = 1.0

def make_data(sheet):
    years = range(2020,2023)
    model = LoadModel()
    for _yr in years:
        model.read_xlsx(f"NE_ISO_Data/{_yr}_smd_hourly.xlsx",sheet,
            time_col=["Date","Hr_End"],
            load_col="RT_Demand",
            temperature_col="Dry_Bulb",
            temperature=None,
            load=None,
            ordinal_hours=True,
            index="dt",
            keep_columns=True,
            index_split=["year"],
            inplace='append',
            )
    return model.data

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

def make_offset_H(H, offset):
    newH = np.copy(H)
    newH = np.roll(newH, -offset, axis=0)
    if offset > 0:
        newH[-offset:] = np.nan
    else:
        newH[:-offset] = np.nan
    return newH

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

if __name__ == "__main__":

    sheet = "RI"
    cache = sheet + ".csv.gz"

    if os.path.exists(cache):
        df = pd.read_csv(cache,index_col=0)
    else:
        df = make_data(sheet=SHEETS[1])
        df.to_csv(cache,compression="gzip")

    print(df)

    #
    # Extract data
    #
    y = np.log(df.loc["2020":"2021"]["RT_Demand"])
    x = df.loc["2020":"2021"]["Dry_Bulb"]

    # 
    # Setup linear regressors
    #
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

    #
    # Fit baseline model with time and temperature features
    #
    print("\n")
    print("Baseline model fit")
    print("------------------")
    a = cvx.Variable(F.shape[1]) # coefficients for time features
    c = cvx.Variable((H0.shape[1], len(Hs))) # coefficients for temperature features
    _s = first_use_set
    temp = cvx.sum([H[_s] @ c[:, _ix] for _ix, H in enumerate(Hs)])
    error = cvx.sum_squares(y.values[_s] - F[_s] @ a - temp) / np.sum(_s)
    regularization = 1e-4 * cvx.sum_squares(Wf @ a) + 1e-4 * cvx.sum_squares(c) + 1e0 * cvx.sum_squares(cvx.diff(c, axis=1))
    problem = cvx.Problem(cvx.Minimize(error + regularization))
    problem.solve(solver='CLARABEL', verbose=False)
    model = (F[_s] @ a + temp).value

    #
    # RMSE
    #
    rmse = np.sqrt(np.average(np.power(np.exp(y.values[first_use_set]) - np.exp(model), 2)))
    print(f"RMS error of model fit: {rmse:.2f}, or {rmse * 100 / np.nanmean(np.exp(y)):.2f}% of average")

    #
    # R2
    #
    r2 = r2_score(np.exp(y.values[first_use_set]), np.exp(model))
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) + len(a[1:].value) + len(c.value) - 1)
    print("\n".join([f"R2: {r2:.3f}", f"R2-adj: {r2_adj:.3f}"]))

    #
    # Plot linear model
    #
    x_sort = np.sort(x.values)
    plt.figure(figsize=(15,10))
    plt.scatter(df.loc["2020":"2021"]['Dry_Bulb'].values, np.exp(y), marker='.',
                label='data', s=10, alpha=.5, color='orange')
    plt.plot(x.values[first_use_set], np.exp(temp.value + a[0].value), label='temperature response', marker='.', ls='none')
    plt.title('Inferred temperature dependence')
    plt.legend()
    plt.grid()
    plt.savefig(sheet+"_1.png")

    # 
    # Fit AR model to residuals
    #
    print("\n")
    print("AR residual model fit")
    print("---------------------")
    baseline_residuals = y.values[first_use_set] - model
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
    ar_model = (B[use_set] @ theta + constant).value
    lap_loc, lap_scale = stats.laplace.fit(baseline_residuals[use_set] - ar_model)
    print(f"""sum-abs of AR coefficients: {cvx.norm1(theta).value:.2f}""")
    print(f"""Baseline MAE: {np.average(np.abs(baseline_residuals)):.2f}, or {np.average(np.abs(baseline_residuals)) * 100 / np.nanmean(y):.2f}% of average""")
    print("""Autoregressive MAE: {np.average(np.abs(baseline_residuals[use_set] - ar_model)):.2f}, or {np.average(np.abs(baseline_residuals[use_set] - ar_model)) * 100 / np.nanmean(y):.2f}% of average""")

    #
    # Generate test data
    #
    test_data = df.loc["2022":]
    new_idx = np.arange(np.sum(df['year'] == 2022)) + np.sum(df['year'] != 2022) - 1
    new_baseline = predict_baseline(new_idx, test_data["Dry_Bulb"].values, a.value, c.value, knots)
    new_noise = roll_out_ar_noise(np.sum(df['year'] == 2022), theta.value, constant.value, lap_loc+LOCATION_ADJ, lap_scale*SCALE_ADJ)
    new_residuals = new_baseline * new_noise - new_baseline
    test_mae = np.nanmean(np.abs(test_data["RT_Demand"].values - new_baseline))
    print(f"test MAE: {test_mae:.2f}, or {100*test_mae / np.nanmean(test_data["RT_Demand"].values):.2f}% of average")

    # 
    # Predict AR residuals
    #
    ppower_actual = np.nanmax(test_data["RT_Demand"].values)
    ppower_predict = np.nanmax(new_baseline * new_noise)
    ppower_predict_noar = np.nanmax(new_baseline)
    ppower_time_actual = np.nanargmax(df.loc["2022":]["RT_Demand"].values)
    ppower_time_predict =  np.nanargmax(new_baseline * new_noise)
    ppower_time_predict_noar =  np.nanargmax(new_baseline)
    _index = ["actual", 'predicted', 'predicted no AR model']
    _data = {
        "peak power": [ppower_actual, ppower_predict, ppower_predict_noar],
        "index of peak": [ppower_time_actual, ppower_time_predict, ppower_time_predict_noar]
    }
    print(pd.DataFrame(data=_data, index=_index))

    #
    # Plot final model
    #
    plt.figure(figsize=(15,10))
    plt.plot(test_data["RT_Demand"].values, new_baseline, marker='.', linewidth=1, alpha=.4, label='true')
    plt.plot(new_baseline*new_noise, new_baseline, marker='.', linewidth=1, alpha=.4, label='sampled')
    plt.xlabel('realization')
    plt.ylabel('baseline')
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], color='yellow', ls='--', linewidth=1)
    plt.xlim(_xlim)
    plt.ylim(_ylim)
    plt.title("Holdout year (2022)")
    plt.grid()
    plt.legend()
    plt.savefig(sheet+"_2.png")
