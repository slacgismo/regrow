import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_solution(solution, tidx, s, Q, B, alpha, beta, efficiency, supertitle=None):
    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))
    if supertitle is not None:
        fig.suptitle(supertitle)

    q = solution["q"][s]
    charged = np.isclose(q, Q, atol=1e-3)
    discharged = np.isclose(q, 0, atol=1e-3)

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
    ax[3].set_title(f"curtailed renewable power, total = {np.sum(c):.2f} GWh")

    sv = solution["s"][s]
    ax[4].plot(tidx[s], sv)
    ax[4].plot(tidx[s][charged], sv[charged], ls="none", marker=".", color="blue")
    ax[4].plot(tidx[s][discharged], sv[discharged], ls="none", marker=".", color="orange")
    ax[4].set_title(f"curtailed load, total = {np.sum(sv):.2f} GWh")

    plt.tight_layout()
    return fig


def plot_heatmap(tidx, values, title=None, cmap=None, center=None):
    _tidx = pd.DatetimeIndex(tidx)
    pivot = pd.DataFrame(
        {
            "value": values,
            "time": _tidx.time,
            "date": _tidx.date,
        }
    ).pivot(index="time", columns="date", values="value")

    n_slots = len(pivot.index)
    fig, ax = plt.subplots(figsize=(14, 5))
    vmin = vmax = None
    if center is not None:
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
    from datetime import datetime as _dt
    slot_minutes = (_dt.combine(_dt.min, pivot.index[1]) - _dt.combine(_dt.min, pivot.index[0])).seconds // 60
    tick_every_y = max(1, 180 // slot_minutes)
    ytick_idx = np.arange(0, n_slots, tick_every_y)
    ax.set_yticks(ytick_idx + 0.5, labels=[str(pivot.index[i]) for i in ytick_idx])
    tick_every = max(1, n_days // 20)
    tick_idx = np.arange(0, n_days, tick_every)
    ax.set_xticks(tick_idx + 0.5, labels=[str(pivot.columns[i]) for i in tick_idx], rotation=90)
    ax.set_ylabel("time of day")
    ax.set_xlabel("date")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    return fig
