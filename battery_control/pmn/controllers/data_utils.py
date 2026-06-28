import numpy as np
import pandas as pd


def build_lsw_df(
    data_path="../single_node_data.csv",
    data_start="2018",
    data_end="2020",
):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df[df.columns] = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")
    df = df[~df.index.duplicated(keep="first")]
    df["load[MW]"] = df["load[MW]"].mask(df["load[MW]"] < 100).interpolate(limit_direction="both").ffill().bfill()
    df = df.loc[data_start:data_end]
    return pd.DataFrame(
        {"l": df["load[MW]"] / 1000, "s": df["pv[MW]"] / 1000, "w": df["wind[MW]"] / 1000},
        index=df.index,
    )


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
    lsw_df = build_lsw_df(data_path=data_path, data_start=data_start, data_end=data_end)
    delta = (lsw_df.index[1] - lsw_df.index[0]).total_seconds() / 3600
    event_end = str((pd.Timestamp(event_start) + pd.Timedelta(days=event_duration_days - 1)).date())
    event_mask = None
    baseline_shortfall_event = None
    if add_event:
        event_slice = lsw_df.loc[event_start:event_end]
        baseline_shortfall_event = delta * np.sum(np.maximum(event_slice["l"] - (event_slice["s"] + event_slice["w"] + G), 0))
        lsw_df = lsw_df.copy()
        lsw_df.loc[event_start:event_end, "l"] *= event_load_factor
        lsw_df.loc[event_start:event_end, "s"] *= event_pv_factor
        event_mask = lsw_df.index.isin(lsw_df.loc[event_start:event_end].index)
    daily_df = lsw_df.groupby(lsw_df.index.date).aggregate("sum")
    daily_df = daily_df.rename(columns={"l": "load[MW]", "s": "pv[MW]", "w": "wind[MW]"})
    daily_df.index = pd.to_datetime(daily_df.index)

    l = lsw_df["l"].to_numpy()
    s = lsw_df["s"].to_numpy()
    w = lsw_df["w"].to_numpy()
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
    event_end = str((pd.Timestamp(event_start) + pd.Timedelta(days=event_duration_days - 1)).date())
    if add_event:
        event_shortfall_after = delta * np.sum(shortfall[event_mask])
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
    return l, R, shortfall, lsw_df.index, daily_df, event_mask, event_shortfall_stats


def stress_event_generator(lsw_df, event_start, event_duration, event_energy, scaling_ratio, verbose=False):
    """
    add a stress event of specified energy magnitude and duration to the lsw (load, solar, wind) data

    event_energy: target increase in net-load energy over the event period (GWh)
    scaling_ratio: ratio of fractional load increase to fractional renewable decrease
                   applied until renewables reach zero (beyond that only load is scaled further).
    """
    event_end = event_start + event_duration
    assert event_energy > 0, f"event_energy must be positive, got {event_energy}"
    event_energy_magnitude = event_energy
    assert event_start >= lsw_df.index[0], f"event_start {event_start} is before data start {lsw_df.index[0]}"
    assert event_end <= lsw_df.index[-1], f"event_end {event_end} is after data end {lsw_df.index[-1]}"
    ev = lsw_df.loc[event_start:event_end]
    delta = (lsw_df.index[1] - lsw_df.index[0]).total_seconds() / 60**2
    E_L_ev = delta * ev["l"].sum()
    E_R_ev = delta * (ev["s"] + ev["w"]).sum()
    E_phase1_max = scaling_ratio * E_L_ev + E_R_ev

    phase2 = event_energy_magnitude > E_phase1_max
    if not phase2:
        alpha = event_energy_magnitude / (E_L_ev + E_R_ev / scaling_ratio)
        load_scaling = 1.0 + alpha
        renewable_scaling = 1.0 - alpha / scaling_ratio
    else:
        load_scaling = 1.0 + (event_energy_magnitude - E_R_ev) / E_L_ev
        renewable_scaling = 0.0

    lsw_scaled_df = lsw_df.copy()
    lsw_scaled_df.loc[event_start:event_end, "l"] *= load_scaling
    lsw_scaled_df.loc[event_start:event_end, "s"] *= renewable_scaling
    lsw_scaled_df.loc[event_start:event_end, "w"] *= renewable_scaling
    if verbose:
        net_before = delta * (ev["l"] - ev["s"] - ev["w"]).sum()
        ev_after = lsw_scaled_df.loc[event_start:event_end]
        net_after = delta * (ev_after["l"] - ev_after["s"] - ev_after["w"]).sum()

        print(f"load scaling:        {load_scaling:.4f}")
        print(f"renewable scaling:   {renewable_scaling:.4f}")
        print(f"net load energy before: {net_before:.2f} GWh")
        print(f"net load energy after:  {net_after:.2f} GWh")
        print(f"phase 1 max addable energy:  {E_phase1_max:.2f} GWh")
    return lsw_scaled_df, phase2


def sample_stress_event(
    lsw_df,
    scaling_ratio,
    duration_range_days=(1, 14),
    energy_range_gwh=(5.0, 200.0),
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    duration_days = int(rng.integers(duration_range_days[0], duration_range_days[1] + 1))
    event_duration = pd.Timedelta(days=duration_days)
    all_dates = pd.DatetimeIndex(sorted(set(lsw_df.index.normalize())))
    valid_dates = all_dates[all_dates <= lsw_df.index[-1] - event_duration]
    event_start = valid_dates[rng.integers(len(valid_dates))]
    event_energy = float(rng.uniform(energy_range_gwh[0], energy_range_gwh[1]))
    lsw_stressed, phase2 = stress_event_generator(lsw_df, event_start, event_duration, event_energy, scaling_ratio)
    return lsw_stressed, event_start, event_duration, event_energy, phase2
