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
        baseline_shortfall_event = delta * np.sum(
            np.maximum(event_slice["l"] - (event_slice["s"] + event_slice["w"] + G), 0)
        )
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


def stress_event_generator(lsw_df, event_start, event_duration, event_shortfall, G, scaling_ratio, verbose=False):
    """
    event_shortfall: target increase in shortfall energy over the event period (GWh)
    scaling_ratio: ratio of fractional load increase to fractional renewable decrease,
                   applied until renewables reach zero (beyond that only load is scaled further)
    """
    event_end = event_start + event_duration
    assert event_shortfall > 0, f"event_shortfall must be positive, got {event_shortfall}"
    assert event_start >= lsw_df.index[0], f"event_start {event_start} is before data start {lsw_df.index[0]}"
    assert event_end <= lsw_df.index[-1], f"event_end {event_end} is after data end {lsw_df.index[-1]}"

    period = lsw_df.index[1] - lsw_df.index[0]
    delta = period.total_seconds() / 3600
    ev = lsw_df.loc[event_start:event_end - period]
    l_ev = ev["l"].to_numpy()
    r_ev = (ev["s"] + ev["w"]).to_numpy()
    sf_base = delta * np.sum(np.maximum(l_ev - r_ev - G, 0))

    def _sf(alpha):
        load_scaling = 1 + alpha
        renewable_scaling = max(0.0, 1 - alpha / scaling_ratio)
        return delta * np.sum(np.maximum(load_scaling * l_ev - renewable_scaling * r_ev - G, 0))

    alpha_hi = 1.0
    while _sf(alpha_hi) - sf_base < event_shortfall:
        alpha_hi *= 2
        if alpha_hi > 1e6:
            raise ValueError(f"Cannot achieve event_shortfall={event_shortfall:.2f} GWh in this event period")

    alpha_lo = 0
    alpha = alpha_hi
    tol = 1e-3
    max_iter = 100
    for i in range(max_iter):
        alpha = (alpha_lo + alpha_hi) / 2
        residual = _sf(alpha) - sf_base - event_shortfall
        if abs(residual) < tol:
            break
        if residual < 0:
            alpha_lo = alpha
        else:
            alpha_hi = alpha
    else:
        raise RuntimeError(f"Bisection did not converge after {max_iter} iterations (residual={residual:.4f} GWh)")
    load_scaling = 1.0 + alpha
    renewable_scaling = max(0.0, 1.0 - alpha / scaling_ratio)

    lsw_scaled_df = lsw_df.copy()
    lsw_scaled_df.loc[event_start:event_end - period, "l"] *= load_scaling
    lsw_scaled_df.loc[event_start:event_end - period, "s"] *= renewable_scaling
    lsw_scaled_df.loc[event_start:event_end - period, "w"] *= renewable_scaling

    if verbose:
        sf_after = delta * np.sum(np.maximum(l_ev * load_scaling - r_ev * renewable_scaling - G, 0))
        print(f"load scaling:        {load_scaling:.4f}")
        print(f"renewable scaling:   {renewable_scaling:.4f}")
        print(f"alpha:               {alpha}")
        print(f"shortfall before:    {sf_base:.2f} GWh")
        print(f"shortfall after:     {sf_after:.2f} GWh")

    return lsw_scaled_df


def sample_stress_event(
    lsw_df,
    G,
    scaling_ratio,
    duration_range_days,
    shortfall_range_gwh,
    start_buffer,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    period = lsw_df.index[1] - lsw_df.index[0]
    max_duration = pd.Timedelta(days=duration_range_days[1])
    all_dates = pd.DatetimeIndex(sorted(set(lsw_df.index.normalize())))
    valid_dates = all_dates[
        (all_dates >= all_dates[0] + start_buffer) &
        (all_dates + max_duration - period <= lsw_df.index[-1])
    ]
    event_start = valid_dates[rng.integers(len(valid_dates))]
    duration_days = int(rng.integers(duration_range_days[0], duration_range_days[1] + 1))
    event_duration = pd.Timedelta(days=duration_days)
    event_shortfall = float(rng.uniform(shortfall_range_gwh[0], shortfall_range_gwh[1]))
    lsw_stressed = stress_event_generator(lsw_df, event_start, event_duration, event_shortfall, G, scaling_ratio)
    return lsw_stressed, event_start, event_duration, event_shortfall
