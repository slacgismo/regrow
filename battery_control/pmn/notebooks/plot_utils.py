import matplotlib.pyplot as plt
import numpy as np


def plot_solution(solution, tidx, s, Q, B, alpha, beta, efficiency):
    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 6))

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
