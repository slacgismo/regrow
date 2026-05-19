import numpy as np
import pandas as pd


def process_single_node_data(
    G,
    data_path='../single_node_data.csv',
    data_start='2018',
    data_end='2020',
    add_event=False,
    event_start='2019-08-15',
    event_end='2019-08-20',
    event_load_factor=1.5,
    event_pv_factor=0.25,
    verbose=False,
):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df[df.columns] = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
    df = df[~df.index.duplicated(keep='first')]
    df['load[MW]'] = df['load[MW]'].mask(df['load[MW]'] < 100).interpolate(limit_direction='both').ffill().bfill()
    df = df.loc[data_start:data_end]
    if add_event:
        df.loc[event_start:event_end, 'load[MW]'] *= event_load_factor
        df.loc[event_start:event_end, 'pv[MW]'] *= event_pv_factor
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
