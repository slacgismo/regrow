from datetime import datetime as _dt

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def compute_partition(q, tidx, Q, atol=1e-6):
    """Return (decouple_points, is_loadshed) for the SOC trajectory.

    decouple_points: interior partition timestamps (alternating last_full/last_empty transitions)
    is_loadshed: bool array of length len(decouple_points)+1 -- True = F->E, False = E->F
    """
    charged = np.isclose(q, Q, atol=atol)
    decouple_mask = charged | np.isclose(q, 0, atol=atol)
    decouple_is_full = charged[decouple_mask]
    decouple_tidx = np.asarray(tidx)[decouple_mask]
    transitions = np.where(np.diff(decouple_is_full.astype(int)) != 0)[0]
    if len(transitions) == 0:
        return np.array([]), np.array([], dtype=bool)
    decouple_points = decouple_tidx[transitions]
    going_to_empty = decouple_is_full[transitions]
    is_loadshed = np.concatenate([[not decouple_is_full[0]], going_to_empty[:-1], [going_to_empty[-1]]])
    return decouple_points, is_loadshed


def plot_timeseries(l, R, shortfall, tidx, s, G, event_dates=None):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 4))
    ax[0].plot(tidx[s], l[s])
    ax[0].axhline(G, color="steelblue", ls=":", linewidth=0.8, label=f"G = {G} GW")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("load [GW]")
    ax[1].plot(tidx[s], R[s])
    ax[1].set_title("renewables [GW]")
    ax[2].plot(tidx[s], shortfall[s])
    ax[2].set_title("shortfall [GW]")
    for _ax in ax:
        _ax.tick_params(axis="x", rotation=45)
    if event_dates is not None:
        t_start = pd.Timestamp(event_dates[0])
        t_end = pd.Timestamp(event_dates[-1]) + pd.Timedelta(days=1)
        t_start = max(t_start, pd.Timestamp(tidx[s][0]))
        t_end = min(t_end, pd.Timestamp(tidx[s][-1]))
        if t_start < t_end:
            ax[-1].plot(
                [t_start, t_end], [-0.05, -0.05],
                transform=ax[-1].get_xaxis_transform(),
                color="red", linewidth=6, solid_capstyle="butt",
                clip_on=False,
            )
    plt.tight_layout()
    return fig


def plot_solution(solution, tidx, s, Q, B, alpha, beta, efficiency, supertitle=None, event_dates=None, atol=1e-6):
    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    if supertitle is not None:
        fig.suptitle(supertitle)
    solution = {k: np.round(v, 6) for k, v in solution.items()}

    q = solution["q"][s]
    charged = np.isclose(q, Q, atol=atol)
    discharged = np.isclose(q, 0, atol=atol)

    decouple_points, is_loadshed = compute_partition(q, tidx[s], Q)
    partition_axes = [ax[3], ax[4]]
    if len(decouple_points) > 0:
        all_times = [tidx[s][0]] + list(decouple_points) + [tidx[s][-1]]
        for t0, t1, orange in zip(all_times[:-1], all_times[1:], is_loadshed):
            for _ax in partition_axes:
                _ax.axvspan(t0, t1, alpha=0.12, color="orange" if orange else "blue", zorder=0)
        for _ax in partition_axes:
            for t in decouple_points:
                _ax.axvline(t, color="gray", alpha=0.5, linewidth=0.8, zorder=0)
        ax[3].legend(
            handles=[
                Patch(color="orange", alpha=0.4, label="load shed zone"),
                Patch(color="blue", alpha=0.4, label="non-dispatch curtail zone"),
            ],
            fontsize=7,
            loc="upper right",
        )

    ax[0].plot(tidx[s], q)
    ax[0].plot(tidx[s][charged], q[charged], ls="none", marker=".", color="blue")
    ax[0].plot(tidx[s][discharged], q[discharged], ls="none", marker=".", color="orange")
    ax[0].axhline(0, color="red", ls="--", linewidth=0.5)
    ax[0].axhline(Q, color="red", ls="--", linewidth=0.5)
    ax[0].axhline(0.5 * Q, color="orange", ls=":", linewidth=0.5)
    ax[0].set_title("battery SOC [GWh]")

    b = solution["b"][s]
    b_out = solution["b_out"][s]
    b_in = solution["b_in"][s]
    ax[1].plot(tidx[s], b)
    ax[1].plot(tidx[s], b_out, linewidth=0.5)
    ax[1].plot(tidx[s], -b_in, linewidth=0.5)
    ax[1].axhline(B, color="red", ls="--", linewidth=0.5)
    ax[1].axhline(-B, color="red", ls="--", linewidth=0.5)
    ax[1].axhline(0, color="orange", ls=":", linewidth=0.5)
    dumped_power = np.max([np.abs(b_out), np.abs(b_in)], axis=0) - np.abs(b)
    ax[1].set_title(f"battery power, dumped = {(1 - efficiency**2) * np.sum(dumped_power):.2f} GWh")

    g = solution["g"][s]
    ax[2].plot(tidx[s], g)
    ax[2].plot(tidx[s][charged], g[charged], ls="none", marker=".", color="blue")
    ax[2].plot(tidx[s][discharged], g[discharged], ls="none", marker=".", color="orange")
    ax[2].set_ylim(-0.1, 1.1)
    utility_cost = np.sum(alpha * g + beta * g**2)
    ax[2].set_title(f"utility power, cost = {utility_cost:.2f}")

    c = solution["c"][s]
    ax[3].plot(tidx[s], c)
    ax[3].plot(tidx[s][charged], c[charged], ls="none", marker=".", color="blue")
    ax[3].plot(tidx[s][discharged], c[discharged], ls="none", marker=".", color="orange")
    ax[3].set_title(f"curtailed non-dispatched generation, total = {np.sum(c):.2f} GWh")

    sv = solution["s"][s]
    ax[4].plot(tidx[s], sv)
    ax[4].plot(tidx[s][charged], sv[charged], ls="none", marker=".", color="blue")
    ax[4].plot(tidx[s][discharged], sv[discharged], ls="none", marker=".", color="orange")
    ax[4].set_title(f"load shedding, total = {np.sum(sv):.2f} GWh")

    for _ax in ax:
        _ax.tick_params(axis="x", rotation=45)
    if event_dates is not None:
        t_start = pd.Timestamp(event_dates[0])
        t_end = pd.Timestamp(event_dates[-1]) + pd.Timedelta(days=1)
        t_start = max(t_start, pd.Timestamp(tidx[s][0]))
        t_end = min(t_end, pd.Timestamp(tidx[s][-1]))
        if t_start < t_end:
            ax[-1].plot(
                [t_start, t_end], [-0.05, -0.05],
                transform=ax[-1].get_xaxis_transform(),
                color="red", linewidth=6, solid_capstyle="butt",
                clip_on=False,
            )
    plt.tight_layout()
    return fig


def plot_heatmap(tidx, values, title=None, cmap=None, center=None, vmin=None, vmax=None, ax=None, event_dates=None):
    _tidx = pd.DatetimeIndex(tidx)
    pivot = pd.DataFrame(
        {
            "value": values,
            "time": _tidx.time,
            "date": _tidx.date,
        }
    ).pivot(index="time", columns="date", values="value")

    n_slots = len(pivot.index)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()
    if vmin is None and vmax is None and center is not None:
        half = max(abs(pivot.values.max() - center), abs(pivot.values.min() - center))
        vmin, vmax = center - half, center + half
    n_days = len(pivot.columns)
    mesh = ax.pcolormesh(
        np.arange(n_days + 1),
        np.arange(n_slots + 1),
        pivot.values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax)
    slot_minutes = (_dt.combine(_dt.min, pivot.index[1]) - _dt.combine(_dt.min, pivot.index[0])).seconds // 60
    tick_every_y = max(1, 6 * 60 // slot_minutes)
    ytick_idx = np.arange(0, n_slots, tick_every_y)
    ax.set_yticks(ytick_idx + 0.5, labels=[str(pivot.index[i]) for i in ytick_idx])
    tick_every = max(1, n_days // 20)
    tick_idx = np.arange(0, n_days, tick_every)
    ax.set_xticks(tick_idx + 0.5, labels=[str(pivot.columns[i]) for i in tick_idx], rotation=90)
    ax.set_ylabel("time of day")
    ax.set_xlabel("date")
    if event_dates:
        col_index = pd.Index(pivot.columns)
        start_dt = pd.Timestamp(event_dates[0]).date()
        end_dt = pd.Timestamp(event_dates[-1]).date()
        loc_start = col_index.searchsorted(start_dt)
        loc_end = min(col_index.searchsorted(end_dt, side="right"), n_days)
        if loc_start < loc_end:
            ax.plot(
                [loc_start, loc_end], [-0.05, -0.05],
                transform=ax.get_xaxis_transform(),
                color="red", linewidth=6, solid_capstyle="butt",
                clip_on=False,
            )
    if title is not None:
        ax.set_title(title)
    if standalone:
        plt.tight_layout()
    return fig


def solution_diff_heatmaps(solution_a, solution_b, tidx, Q, B, G, label_a="a", label_b="b", event_dates=None):
    vars_config = [
        ("b", "b [GWh]", B),
        ("g", "g [GWh]", G),
        ("s", "s [GWh]", G / 100),
        ("c", "c [GWh]", G),
        ("q", "SOC [%]", 100),
    ]
    fig, axes = plt.subplots(len(vars_config), 1, figsize=(14, 3 * len(vars_config)))
    for ax, (var, unit, lim) in zip(axes, vars_config):
        if var == "q":
            a_vals = 100 * solution_a[var][1:] / Q
            b_vals = 100 * solution_b[var][1:] / Q
        else:
            a_vals = solution_a[var]
            b_vals = solution_b[var]
        diff = np.round(a_vals - b_vals, 6)
        plot_heatmap(
            tidx,
            diff,
            title=f"diff {unit} ({label_a} - {label_b})",
            cmap="RdBu_r",
            center=0,
            vmin=-lim,
            vmax=lim,
            ax=ax,
            event_dates=event_dates,
        )
    plt.tight_layout()
    return fig


def solution_heatmaps(solution, tidx, Q, B, G, label="", event_dates=None):
    prefix = f"{label} " if label else ""
    vars_config = [
        ("b", "b [GWh]", "RdBu_r", None, -B, B),
        ("g", "g [GWh]", "Oranges", None, 0, G),
        ("s", "s [GWh]", "Oranges", None, 0, None),
        ("c", "c [GWh]", "Greens", None, 0, None),
        ("q", "SOC [%]", "viridis", None, 0, 100),
    ]
    fig, axes = plt.subplots(len(vars_config), 1, figsize=(14, 3 * len(vars_config)))
    for ax, (var, title, cmap, center, vmin, vmax) in zip(axes, vars_config):
        values = np.round(100 * (solution[var][1:] / Q) if var == "q" else solution[var], 6)
        plot_heatmap(
            tidx,
            values,
            title=f"{prefix}{title}",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            event_dates=event_dates,
        )
    plt.tight_layout()
    return fig
