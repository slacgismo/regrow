import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from controllers.data_utils import build_lsw_df, stress_event_generator, sample_stress_event

    data_path = str(pathlib.Path(__file__).parent.parent.parent / "single_node_data.csv")
    return (
        build_lsw_df,
        data_path,
        mo,
        pd,
        plt,
        sample_stress_event,
        stress_event_generator,
    )


@app.cell
def _(build_lsw_df, data_path):
    lsw_df = build_lsw_df(data_path=data_path, data_start="2018", data_end="2020")
    return (lsw_df,)


@app.cell
def _(mo):
    scaling_ratio_input = mo.ui.number(start=0.01, stop=10.0, step=0.05, label="scaling ratio (load / renewable)", value=0.667)
    scaling_ratio_input
    return (scaling_ratio_input,)


@app.cell
def _(mo):
    mo.md("""
    ## Manual event
    """)
    return


@app.cell
def _(mo):
    event_start_input = mo.ui.text(value="2019-08-15", label="event start")
    event_duration_input = mo.ui.number(start=1, step=1, label="event duration [days]", value=2)
    energy_input = mo.ui.number(start=0.0, stop=500.0, step=1.0, label="event energy [GWh]", value=20.0)
    mo.vstack([event_start_input, event_duration_input, energy_input])
    return energy_input, event_duration_input, event_start_input


@app.cell
def _(
    energy_input,
    event_duration_input,
    event_start_input,
    lsw_df,
    pd,
    scaling_ratio_input,
    stress_event_generator,
):
    event_start = pd.Timestamp(event_start_input.value)
    event_duration = pd.Timedelta(days=int(event_duration_input.value))
    lsw_stressed, phase2 = stress_event_generator(
        lsw_df,
        event_start=event_start,
        event_duration=event_duration,
        event_energy=energy_input.value,
        scaling_ratio=scaling_ratio_input.value,
        verbose=True,
    )
    print(f"phase 2 used (load-only scaling): {phase2}")
    return event_duration, event_start, lsw_stressed


@app.cell
def _(plt):
    def plot_event(lsw_df, lsw_stressed, event_start, event_duration):
        event_end = event_start + event_duration
        pad = event_duration / 4
        before = lsw_df.loc[event_start - pad:event_end + pad]
        after = lsw_stressed.loc[event_start - pad:event_end + pad]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
        for ax, col, label in zip(axes, ["l", "s", "w"], ["load [GW]", "solar [GW]", "wind [GW]"]):
            ax.plot(before.index, before[col], label="baseline")
            ax.plot(after.index, after[col], label="stressed")
            ax.axvline(event_start, color="k", linestyle="--", linewidth=0.8, label="event")
            ax.axvline(event_end, color="k", linestyle="--", linewidth=0.8)
            ax.set_ylabel(label)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(axis="x", rotation=45)
        net_before = before["l"] - before["s"] - before["w"]
        net_after = after["l"] - after["s"] - after["w"]
        axes[3].plot(before.index, net_before, label="baseline")
        axes[3].plot(after.index, net_after, label="stressed")
        axes[3].axvline(event_start, color="k", linestyle="--", linewidth=0.8, label="event")
        axes[3].axvline(event_end, color="k", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("net load [GW]")
        axes[3].legend(fontsize=7, loc="upper right")
        axes[3].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    return (plot_event,)


@app.cell
def _(event_duration, event_start, lsw_df, lsw_stressed, plot_event):
    plot_event(lsw_df, lsw_stressed, event_start, event_duration)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sampled event
    """)
    return


@app.cell
def _(mo):
    sample_button = mo.ui.button(label="Sample event")
    duration_min_input = mo.ui.number(start=1, stop=30, step=1, label="min duration [days]", value=1)
    duration_max_input = mo.ui.number(start=1, stop=30, step=1, label="max duration [days]", value=5)
    energy_min_input = mo.ui.number(start=0.0, stop=500.0, step=5.0, label="min energy [GWh]", value=5.0)
    energy_max_input = mo.ui.number(start=0.0, stop=500.0, step=5.0, label="max energy [GWh]", value=200.0)
    mo.vstack([
        duration_min_input,
        duration_max_input,
        energy_min_input,
        energy_max_input,
        sample_button,
    ])
    return (
        duration_max_input,
        duration_min_input,
        energy_max_input,
        energy_min_input,
        sample_button,
    )


@app.cell
def _(
    duration_max_input,
    duration_min_input,
    energy_max_input,
    energy_min_input,
    lsw_df,
    sample_button,
    sample_stress_event,
    scaling_ratio_input,
):
    sample_button.value
    lsw_sampled, s_start, s_duration, s_energy, s_phase2 = sample_stress_event(
        lsw_df,
        scaling_ratio=scaling_ratio_input.value,
        duration_range_days=(int(duration_min_input.value), int(duration_max_input.value)),
        energy_range_gwh=(energy_min_input.value, energy_max_input.value),
    )
    print(f"event start:    {s_start.date()}")
    print(f"duration:       {int(s_duration.days)} days")
    print(f"energy:         {s_energy:.2f} GWh")
    print(f"phase 2 used:   {s_phase2}")
    return lsw_sampled, s_duration, s_start


@app.cell
def _(lsw_df, lsw_sampled, plot_event, s_duration, s_start):
    plot_event(lsw_df, lsw_sampled, s_start, s_duration)
    return


if __name__ == "__main__":
    app.run()
