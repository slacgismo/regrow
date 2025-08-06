import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return np, pd


@app.cell
def _(np, pd):
    def process_single_node_data(G, add_event=False, verbose=False):
        df = pd.read_csv('https://raw.githubusercontent.com/slacgismo/regrow/refs/heads/control/battery_control/single_node_data.csv', index_col=0, parse_dates=True)
        df[df.columns] = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
        df = df[~df.index.duplicated(keep='first')]
        df['load[MW]'] = df['load[MW]'].mask(df['load[MW]'] < 100).interpolate(limit_direction='both').ffill().bfill()
        # df.index += pd.Timedelta(hours=8)
        df = df.loc["2018":"2020"]
        if add_event:
            df.loc["2019-08-15":"2019-08-20", 'load[MW]'] *= 1.5
            df.loc["2019-08-15":"2019-08-20", 'pv[MW]'] *= 0.25
        daily_df = df.groupby(df.index.date).aggregate('sum') / 1000
        daily_df.index = pd.to_datetime(daily_df.index)

        l = df['load[MW]'].to_numpy() / 1000
        s = df['pv[MW]'].to_numpy() / 1000
        w = df['wind[MW]'].to_numpy() / 1000
        R = w + s
        netload = l - (R + G)

        infeasible_indices = np.where(netload > 0)[0]
        shortfall = np.maximum(l - (R + G), 0)
        if verbose:
            print(f"percentage of shortfall times = {len(infeasible_indices) / len(l) * 100:.2f} %")
            print(f"average load = {np.mean(l):.2f} GW")
            print(f"average renewable generation = {np.mean(R):.2f} GW")
            print(f"average shortfall = {np.mean(shortfall):.2f} GW")
            print(f"maximum fossil generation = {np.max(G):.2f} GW")
        return l, R, shortfall, df.index, daily_df
    return


if __name__ == "__main__":
    app.run()
