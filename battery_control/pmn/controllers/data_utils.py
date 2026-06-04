import numpy as np
import pandas as pd


def process_single_node_data(
    G,
    data_path="../single_node_data.csv",
    data_start="2018",
    data_end="2020",
    add_event=False,
    event_start="2019-08-15",
    event_duration_days=2,
    event_load_factor=1.5,
    event_pv_factor=0.25,
    verbose=False,
):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df[df.columns] = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")
    df = df[~df.index.duplicated(keep="first")]
    df["load[MW]"] = df["load[MW]"].mask(df["load[MW]"] < 100).interpolate(limit_direction="both").ffill().bfill()
    df = df.loc[data_start:data_end]
    event_end = str((pd.Timestamp(event_start) + pd.Timedelta(days=event_duration_days - 1)).date())
    event_mask = None
    if add_event:
        event_slice = df.loc[event_start:event_end]
        baseline_shortfall_event = np.sum(
            np.maximum(event_slice["load[MW]"] / 1000 - (event_slice[["pv[MW]", "wind[MW]"]].sum(axis=1) / 1000 + G), 0)
        )
        df.loc[event_start:event_end, "load[MW]"] *= event_load_factor
        df.loc[event_start:event_end, "pv[MW]"] *= event_pv_factor
        event_mask = df.index.isin(df.loc[event_start:event_end].index)
    daily_df = df.groupby(df.index.date).aggregate("sum") / 1000
    daily_df.index = pd.to_datetime(daily_df.index)

    l = df["load[MW]"].to_numpy() / 1000
    s = df["pv[MW]"].to_numpy() / 1000
    w = df["wind[MW]"].to_numpy() / 1000
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
    event_shortfall_stats = None
    if add_event:
        event_shortfall_after = np.sum(shortfall[event_mask])
        added = event_shortfall_after - baseline_shortfall_event
        pct_increase = added / baseline_shortfall_event * 100 if baseline_shortfall_event > 0 else float("nan")
        event_shortfall_stats = {
            "baseline_shortfall": baseline_shortfall_event,
            "event_shortfall": event_shortfall_after,
            "added_shortfall": added,
            "pct_increase": pct_increase,
        }
        if verbose:
            print()
            print(f"event period ({event_start} to {event_end}, {event_mask.sum()} steps):")
            print(f"  baseline shortfall = {baseline_shortfall_event:.3f} GWh")
            print(f"  event shortfall    = {event_shortfall_after:.3f} GWh")
            print(f"  added shortfall    = {added:+.3f} GWh  ({pct_increase:+.1f}%)")
    return l, R, shortfall, df.index, daily_df, event_mask, event_shortfall_stats
