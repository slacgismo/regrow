"""Load modeling library"""
#
# Modifications from original v3 notebook
# 1. removed +1h from input timestamps (sampling of NEISO is trailing)
#
# To do:
# 1. Fix window
#
# Suggestions
# 2. Add slope to Hs array in both fit and predict
# 3. Convert to sklearn fit_predict() implementation
#

import os
import marimo as mo
import pandas as pd
import numpy as np
import cvxpy as cvx
import datetime as dt
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score
import statsmodels.api as sm
from spcqe import make_basis_matrix, make_regularization_matrix
import random

pd.options.display.max_columns=None
pd.options.display.width=None

CACHE=True # generate and use cache
# CACHE=False # don't use cache

def read_NEISO_data(sheet):
    """Collate data from NE ISO Excel sheets

    Arguments
    ---------

    TODO
    """
    years = [2020, 2021, 2022]
    df_list = []
    for _yr in years:
        fp = Path('.') / 'NE_ISO_Data' / f'{_yr}_smd_hourly.xlsx' 
        df = pd.read_excel(fp, sheet_name=sheet)
        df['year'] = _yr
        df.index = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hr_End'].map(lambda x: f"{x-1}:00:00"))
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    return df

def d_func(x, k, k_max):
    """d_func

    Arguments
    ---------

    TODO
    """
    n1 = np.clip(np.power(x - k, 3), 0, np.inf)
    n2 = np.clip(np.power(x - k_max, 3), 0, np.inf)
    d1 = k_max - k
    out = (n1 - n2) / d1
    return out


def make_H(x, knots, include_offset=False):
    """make_H

    Arguments
    ---------

    TODO
    """
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
    """make_offset_H

    Arguments
    ---------

    TODO
    """
    newH = np.copy(H)
    newH = np.roll(newH, -offset, axis=0)
    if offset > 0:
        newH[-offset:] = np.nan
    else:
        newH[:-offset] = np.nan
    return newH

def running_view(arr, window, axis=-1):
    """Return a running view of length 'window' over 'axis'

    Nan-pads the start to get the same first dimension as the input the
    returned array has an extra last dimension, which spans the window
    
    Arguments
    ---------

    TODO
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
    """roll_out_ar_noise

    Arguments
    ---------

    TODO
    """
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

def predict_baseline(time_idxs, temp_data, time_coeff, temp_coeff, knots, model='all',window=3):
    F = make_basis_matrix(
        num_harmonics=[6, 4, 3],
        length=max(time_idxs) +1,
        periods=[365.2425 * 24, 7 * 24, 24]
    )
    """Predict baseline

    Arguments
    ---------

    TODO
    """
    F = F[time_idxs]
    H0 = make_H(temp_data, knots, include_offset=False)
    Hs = [H0]
    for n in range(window):
        Hs = [make_offset_H(H0,-n-1)] + Hs + [make_offset_H(H0,n+1)]
    temp = np.sum([H @ temp_coeff[:, _ix] for _ix, H in enumerate(Hs)], axis=0)
    if model == 'all':
        baseline = F @ time_coeff + temp
    elif model == 'time':
        baseline = F @ time_coeff
    elif model == 'temp':
        baseline = temp + time_coeff[0]
    return np.exp(baseline)

class LinearRegressor:
    """LinearRegressor implementation

    Parameters
    ----------

        F: the basis matrix

        Wf: the regularization matrix

        knots: the list of knots

        H0: the H matrix

        Hs: the H matrix with windowed averages
    """
    def __init__(self,
            x:list[float|np.float64]|np.ndarray,
            y:list[float|np.float64]|np.ndarray,
            nharmon:int,
            periods:list[float|np.float64],
            nK:int,
            window:int=3,
        ):
        """Construct a linear regressor from data

        Arguments
        ----------

            x: x values (array of floats)

            y: y values (array of floats)
            
            nharmon: number of harmonics for each period
            
            periods: values of periods (list of positive floats)
            
            nK: number of knots (positive integer)

            window: the forward/backward window size (non-negative integer)
        """
        self.x = x
        self.y = y
        self.F = make_basis_matrix(
            num_harmonics=nharmon,
            length=len(y),
            periods=periods
        )
        # weight matrix for regularized Fourier parameters
        self.Wf = make_regularization_matrix(
            num_harmonics=nharmon,
            weight=1,
            periods=[365.2425 * 24, 7 * 24, 24]
        )
        # Temperature terms
        self.knots = np.linspace(np.min(x), np.max(x), nK)
        self.H0 = make_H(x, self.knots, include_offset=False)
        self.Hs = [self.H0]
        for n in range(window):
            hn = make_offset_H(self.H0,-n-1)
            hp = make_offset_H(self.H0,n+1)
            self.Hs = [hn] + self.Hs + [hp]
        self.first_use_set = np.all(np.all(~np.isnan(np.asarray(self.Hs)), axis=-1), axis=0)

    def __str__(self):
        return ", ".join([f"{x}={y}" for x,y in self.todict().items()])

    def todict(self):
        return {x:getattr(self,x) for x in dir(self) if not x.startswith("_")}

class BaselineModel:
    """Baseline model implementation

    Parameters:

        model: the baseline model (F*a+temp)
    """
    def __init__(self,a,c,LR):
        _s = LR.first_use_set
        self.LR = LR
        self.a = a
        self.c = c
        temp = cvx.sum([H[_s] @ c[:, _ix] for _ix, H in enumerate(LR.Hs)])
        error = cvx.sum_squares(LR.y.values[_s] - LR.F[_s] @ a - temp) / np.sum(_s)
        regularization = 1e-4 * cvx.sum_squares(LR.Wf @ a) + 1e-4 * cvx.sum_squares(c) + 1e0 * cvx.sum_squares(cvx.diff(c, axis=1))
        problem = cvx.Problem(cvx.Minimize(error + regularization))
        problem.solve(solver='CLARABEL', verbose=False)
        self.model,self.temp = (LR.F[_s] @ a + temp).value,temp

    def __str__(self):
        return ", ".join([f"{x}={y}" for x,y in self.todict().items()])

    def todict(self):
        return {x:getattr(self,x) for x in dir(self) if not x.startswith("_")}

class AutoregressionModel:
    """Autoregressive model implementation

    Parameters:

        baseline_residuals: residuals from the baseline model

        theta:

        constant:

        lap_loc: (experimental)

        lap_scale: (experimental)

        use_set:

        model:

    """
    def __init__(self,y,BM):
        """Construct the AR model

        Arguments
        ----------

            y: y-values

            BM: baseline model
        """
        self.BM = BM
        baseline_residuals = y.values[BM.LR.first_use_set] - BM.model
        B = running_view(baseline_residuals, 36)
        # usable records with AR lags
        use_set = np.all(~np.isnan(B), axis=1)
        theta = cvx.Variable(B.shape[1])
        # TODO: add labels to variables
        constant = cvx.Variable()
        problem2 = cvx.Problem(
            cvx.Minimize(cvx.sum_squares(baseline_residuals[use_set] - B[use_set] @ theta - constant)),
            [cvx.norm1(theta) <= 0.95]
        )
        problem2.solve(solver='CLARABEL')
        model = (B[use_set] @ theta + constant).value
        lap_loc, lap_scale = stats.laplace.fit(baseline_residuals[use_set] - model)

        self.baseline_residuals = baseline_residuals
        self.theta = theta
        self.constant = constant
        self.lap_loc = lap_loc
        self.lap_scale = lap_scale
        self.use_set = use_set
        self.model = model

        # TODO add problem and problem data to attributes

    def __str__(self):
        return ", ".join([f"{x}={y}" for x,y in self.todict().items()])

    def todict(self):
        return {x:getattr(self,x) for x in dir(self) if not x.startswith("_")}

class LoadModel:
    """Generate a load model given load data

    Parameters
    ----------

    LR: the linear regressor (see LinearRegressor)

    BM: the baseline model (see BaselineModel)

    AR: the autoregression model (see AutoregressionModel)

    test_data: the test data used to compute new data

    new_baseline: the new baseline data

    new_noise: the new noise generated

    new_residuals: the new residuals

    results: analysis of the new data
    """
    def __init__(self,
        data, # (t,x,y)
        holdout, # index at which test data begins
        *,
        period_harmonics = {
            365.2425*24: 6,
            7*24: 4,
            24: 3,
        },
        knots=10,
        LOCATION_ADJ=0.0, # experimental
        SCALE_ADJ=1.0, # experimental
        verbose = lambda x: None,
        window=3,
        # t0:int = 0, # data index origin 
        ):
        """Construct a load model from data

        Arguments
        ---------

        data: tuple of (t,x,y) of dates, temperatures, and loads

        holdout: value from t for start of holdout test data

        verbose: callable for verbose progress output
        """
        self.data = pd.DataFrame(index=data[0],data={"x":data[1],"y":data[2]})
        self.holdout = holdout

        #
        # Extract data
        #
        x = self.data.loc[:holdout]["x"] 
        y = np.log(self.data.loc[:holdout]["y"]) # TODO: preprocessing to be done by sklearn API

        # 
        # Setup linear regressors
        #
        LR = LinearRegressor(
            x,y,
            nharmon=list(period_harmonics.values()),
            periods=list(period_harmonics.keys()),
            nK=knots,
            window=window,
            )
        self.LR = LR

        #
        # Fit baseline model with time and temperature features
        #
        verbose("\n")
        verbose("Baseline model fit")
        verbose("------------------")
        a = cvx.Variable(LR.F.shape[1]) # coefficients for time features
        c = cvx.Variable((LR.H0.shape[1], len(LR.Hs))) # coefficients for temperature features
        BM = BaselineModel(a,c,LR)
        self.BM = BM

        #
        # RMSE
        #
        rmse = np.sqrt(np.average(np.power(np.exp(y.values[LR.first_use_set]) - np.exp(BM.model), 2)))
        verbose(f"RMS error of model fit: {rmse:.2f}, or {rmse * 100 / np.nanmean(np.exp(y)):.2f}% of average")

        #
        # R2
        #
        r2 = r2_score(np.exp(y.values[LR.first_use_set]), np.exp(BM.model))
        r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) + len(a[1:].value) + len(c.value) - 1)
        verbose("\n".join([f"R2: {r2:.3f}", f"R2-adj: {r2_adj:.3f}"]))
        self.scores = { # TODO: use sklearn API
            "rmse" :rmse,
            "r2": r2,
            "r2_adj" : r2_adj,
        }

        # 
        # Fit AR model to residuals
        #
        verbose("\n")
        verbose("AR residual model fit")
        verbose("---------------------")
        AR = AutoregressionModel(y,BM)
        self.AR = AR

        verbose(f"""sum-abs of AR coefficients: {cvx.norm1(AR.theta).value:.2f}""")
        verbose(f"""Baseline MAE: {np.average(np.abs(AR.baseline_residuals)):.2f}, or {np.average(np.abs(AR.baseline_residuals)) * 100 / np.nanmean(y):.2f}% of average""")
        verbose(f"""Autoregressive MAE: {np.average(np.abs(AR.baseline_residuals[AR.use_set] - AR.model)):.2f}, or {np.average(np.abs(AR.baseline_residuals[AR.use_set] - AR.model)) * 100 / np.nanmean(y):.2f}% of average""")

        #
        # Generate test data
        #
        test_data = self.data[holdout:]
        new_idx = np.arange(len(self.data[:holdout]),len(self.data)) - 1
        new_baseline = predict_baseline(new_idx, test_data["x"].values, a.value, c.value, LR.knots)
        new_noise = roll_out_ar_noise(new_idx[-1]-new_idx[0]+1, AR.theta.value, AR.constant.value, AR.lap_loc+LOCATION_ADJ, AR.lap_scale*SCALE_ADJ)
        new_residuals = new_baseline * new_noise - new_baseline
        test_mae = np.nanmean(np.abs(test_data["y"].values - new_baseline))
        verbose(f"test MAE: {test_mae:.2f}, or {100*test_mae / np.nanmean(test_data["y"].values):.2f}% of average")
        self.test_data = test_data
        self.new_baseline = new_baseline
        self.new_noise = new_noise
        self.new_residuals = new_residuals

        # 
        # Predict AR residuals
        #
        ppower_actual = np.nanmax(test_data["y"].values)
        ppower_predict = np.nanmax(new_baseline * new_noise)
        ppower_predict_noar = np.nanmax(new_baseline)
        ppower_time_actual = np.nanargmax(test_data["y"].values)
        ppower_time_predict =  np.nanargmax(new_baseline * new_noise)
        ppower_time_predict_noar =  np.nanargmax(new_baseline)
        index = ["actual", 'predicted', 'predicted no AR model']
        data = {
            "peak power": [ppower_actual.round(1), ppower_predict.round(1), ppower_predict_noar.round(1)],
            "index of peak": [ppower_time_actual, ppower_time_predict, ppower_time_predict_noar]
        }
        self.results = pd.DataFrame(data=data, index=index)
        verbose(self.results)


    def plot_LR(self,*,
        figsize=(15,10),
        scatter=dict(
            marker='.',
            label='Data', 
            s=10, 
            alpha=.5, 
            color='orange',
            ),
        plot=dict(
            label='Temperature response', 
            marker='.', 
            ls='none',
            ),
        title="Inferred temperature dependence",
        xlabel=r"Nominal temperature ($^o$C)",
        ylabel="Demand (MW)",
        legend=True,
        grid=True,
        ):
        #
        # Plot linear model
        #
        x_sort = np.sort(self.LR.x.values)
        plt.figure(figsize=figsize)
        plt.scatter(self.LR.x,np.exp(self.LR.y),**scatter)
        plt.plot(self.LR.x.values[self.LR.first_use_set], 
            np.exp(self.BM.temp.value + self.BM.a[0].value),**plot)
        if title:
            plt.title(title)
        if isinstance(legend,list):
            plt.legend(*legend)
        elif legend == True:
            plt.legend()
        if grid:
            plt.grid()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        return plt

    def plot_LM(self,*,
        figsize=(15,10),
        old=dict(marker='.', 
            linewidth=1, 
            alpha=.4, 
            label='true',),
        new=dict(marker='.', 
            linewidth=1, 
            alpha=.4, 
            label='sampled',),
        equal=dict(color='yellow', ls='--', linewidth=1),
        title="Holdout samples",
        xlabel="Realization",
        ylabel="Baseline",
        legend=True,
        grid=True,
        ):
        #
        # Plot final load model
        #
        plt.figure(figsize=figsize)
        plt.plot(self.test_data["y"].values,self.new_baseline,**old)
        plt.plot(self.new_baseline * self.new_noise,self.new_baseline,**new)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        _xlim = plt.xlim()
        _ylim = plt.ylim()
        plt.plot([-1e6, 1e6], [-1e6, 1e6], **equal)
        plt.xlim(_xlim)
        plt.ylim(_ylim)
        if title:
            plt.title(title)
        if grid:
            plt.grid()
        if isinstance(legend,list):
            plt.legend(*legend)
        elif legend == True:
            plt.legend()
        return plt

    # TODO: implement MC trials to get probability of within nhours of peak

if __name__ == "__main__":

    # input data
    # SHEETS = [
    #     "ISO NE CA",
    #     "ME",
    #     "NH",
    #     "VT",
    #     "CT",
    #     "RI",
    #     "SEMA",
    #     "WCMA",
    #     "NEMA"
    # ]
    sheet = "ME"
    cache = sheet + ".csv.gz"

    np.random.seed(42) # what do you get when you multiply nine by six?

    if os.path.exists(cache) and CACHE == True:
        _df = pd.read_csv(cache,index_col=0)
    else:
        _df = read_NEISO_data(sheet)
        if CACHE == True:
            _df.to_csv(cache,compression="gzip",index=True,header=True)

    t = _df.index
    x = _df["Dry_Bulb"].values
    y = _df["RT_Demand"].values

    #
    # Generate load model
    #
    LM = LoadModel((t,x,y),"2022",window=3,verbose=print)
    LM.plot_LR().savefig(sheet+"_LR.png")
    LM.plot_LM().savefig(sheet+"_LM.png")
