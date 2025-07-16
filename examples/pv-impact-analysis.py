import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full", app_title="PV Heatwave Loss Analysis")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy.stats import mode
    import matplotlib.pyplot as plt
    import warnings
    return mo, mode, np, pd, plt, warnings


@app.cell
def _(mo):
    load_data = mo.ui.file_browser(multiple=False)
    load_data
    return (load_data,)


@app.cell
def _(df, mo):
    select_columns = mo.ui.multiselect(options=list(df.columns), full_width=False, label='select columns to analyze').form()
    # run_analysis = mo.ui.run_button(label='run analysis')
    # mo.vstack([select_columns, run_analysis])
    select_columns
    return (select_columns,)


@app.cell
def _(new_data):
    len(new_data)
    return


@app.cell
def _(energy_df, np, pd, select_columns):
    output = {}
    for _c in select_columns.value:
        sys_data = {}
        for _yr in set(energy_df.index.year):
            try:
                new_data = energy_df.loc[str(_yr)+"-08", _c].values
            except:
                continue
            if not np.all(np.isnan(new_data)):
                if len(new_data) == 31:
                    sys_data[_yr] = new_data
                else:
                    _copy = np.zeros(31)
                    _copy[:] = np.nan
                    _idxs = energy_df.loc[str(_yr)+"-08", _c].index.day - 1
                    _copy[_idxs] = new_data
        sys_data = pd.DataFrame(data=sys_data)
        baseline = sys_data.drop([2020], axis=1)
        baseline = np.nanmedian(baseline.values, axis=1)
        _bl = baseline
        _b = np.r_[_bl, np.zeros(len(_bl) - 2)]
        _A = np.concatenate([np.eye(len(_bl)), (10 ** 1.2) * np.diff(np.eye(len(_bl)), axis=0, n=2)], axis=0)
        _x, _, _, _ = np.linalg.lstsq(_A, _b)
        sys_data['baseline_raw'] = baseline
        sys_data['baseline_smoothed'] = _x
        output[_c] = sys_data
    return new_data, output


@app.cell
def _(mo):
    smooth_weight = mo.ui.slider(start=-3, stop=5, step=0.1, value=0.8, label="smoothing weight")
    diff_order = mo.ui.number(start=1, stop=4, value=1, label='smoothing difference order')
    mo.hstack([smooth_weight, diff_order])
    return diff_order, smooth_weight


@app.cell
def _(diff_order, np, output, plt, smooth_weight, sys_dropdown):
    _bl = output[sys_dropdown.value]['baseline_raw']
    b = np.r_[_bl, np.zeros(len(_bl) - diff_order.value)]
    A = np.concatenate([np.eye(len(_bl)), (10 ** smooth_weight.value) * np.diff(np.eye(len(_bl)), axis=0, n=diff_order.value)], axis=0)
    x, _, _, _ = np.linalg.lstsq(A, b)
    plt.plot(_bl, label='raw baseline')
    plt.plot(x, label='smoothed baseline')
    plt.legend()
    plt.gcf()
    return


@app.cell
def _(mo, output):
    sys_dropdown = mo.ui.dropdown(output.keys(), value=list(output.keys())[0])
    return (sys_dropdown,)


@app.cell
def _(fig1, fig2, mo, sys_dropdown):
    mo.vstack([sys_dropdown, mo.hstack([fig1, fig2])])
    return


@app.cell
def _(output, plt, sys_dropdown):
    _k = sys_dropdown.value
    output[_k].plot(y=['baseline_raw', 'baseline_smoothed', 2020])
    df_to_plot = output[_k].drop(['baseline_raw', 'baseline_smoothed', 2020], axis=1)
    df_to_plot.plot(linewidth=1, ls=':', ax=plt.gca())
    plt.axvline(14, color='red', ls='--', label='heatwave')
    plt.axvline(19, color='red', ls='--')
    plt.legend()
    plt.ylabel('daily energy [Wh or kWh]')
    plt.xlabel('day in August')
    plt.title(sys_dropdown.value)
    fig1 = plt.gcf()
    return (fig1,)


@app.cell
def _(output, plt, sys_dropdown):
    _k = sys_dropdown.value
    plt.plot(output[_k][2020] / output[_k]['baseline_raw'], label='raw baseline deviation')
    plt.plot(output[_k][2020] / output[_k]['baseline_smoothed'], label='smoothed baseline deviation')
    plt.axvline(14, color='red', ls='--', label='heatwave')
    plt.axvline(19, color='red', ls='--')
    plt.legend()
    plt.ylabel('daily energy [Wh or kWh]')
    plt.xlabel('day in August')
    plt.title(sys_dropdown.value + ' 2020 fraction of baseline')
    fig2=plt.gcf()
    return (fig2,)


@app.cell
def _(df, mode, np):
    float(mode(np.diff(df.index) / 6e10)[0]) / 60
    return


@app.cell
def _(energy_df, select_columns):
    energy_df.plot(y=select_columns.value)
    return


@app.cell
def _(energy_df):
    energy_df
    return


@app.cell
def _(count_df):
    count_df
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mode, np, pd, warnings):
    sum_df = df.groupby(df.index.date).aggregate('sum')
    count_df = df.groupby(df.index.date).aggregate('count')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy_df = sum_df * (float(mode(np.diff(df.index) / 6e10)[0]) / 60)
    energy_df.index = pd.to_datetime(energy_df.index)
    return count_df, energy_df


@app.cell
def _(load_data, pd):
    df = pd.read_csv(load_data.value[0].id, index_col=0, parse_dates=[0])
    return (df,)


if __name__ == "__main__":
    app.run()
