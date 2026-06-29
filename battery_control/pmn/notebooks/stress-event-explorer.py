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
    G = 1
    lsw_df = build_lsw_df(data_path=data_path, data_start="2018", data_end="2020")
    return G, lsw_df


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
    shortfall_input = mo.ui.number(label="event shortfall [GWh]", value=10.0)
    mo.vstack([event_start_input, event_duration_input, shortfall_input])
    return event_duration_input, event_start_input, shortfall_input


@app.cell
def _(
    G,
    event_duration_input,
    event_start_input,
    lsw_df,
    mo,
    pd,
    scaling_ratio_input,
    shortfall_input,
    stress_event_generator,
):
    event_start = pd.Timestamp(event_start_input.value)
    event_duration = pd.Timedelta(days=int(event_duration_input.value))
    lsw_stressed = stress_event_generator(
        lsw_df,
        event_start=event_start,
        event_duration=event_duration,
        event_shortfall=shortfall_input.value,
        G=G,
        scaling_ratio=scaling_ratio_input.value,
    )
    _period = lsw_df.index[1] - lsw_df.index[0]
    _delta = _period.total_seconds() / 3600
    _ev_base = lsw_df.loc[event_start:event_start + event_duration - _period]
    _ev_after = lsw_stressed.loc[event_start:event_start + event_duration - _period]
    _sf_base = _delta * (_ev_base["l"] - _ev_base["s"] - _ev_base["w"] - G).clip(lower=0).sum()
    _sf_after = _delta * (_ev_after["l"] - _ev_after["s"] - _ev_after["w"] - G).clip(lower=0).sum()
    mo.md(f"**baseline shortfall:** {_sf_base:.2f} GWh | **added shortfall:** {_sf_after - _sf_base:.2f} GWh")
    return event_duration, event_start, lsw_stressed


@app.cell
def _(plt):
    def plot_event(lsw_df, lsw_stressed, event_start, event_duration, G):
        event_end = event_start + event_duration
        pad = event_duration / 4
        before = lsw_df.loc[event_start - pad:event_end + pad]
        after = lsw_stressed.loc[event_start - pad:event_end + pad]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
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
        axes[3].plot(before.index, (net_before - G).clip(lower=0), label="baseline")
        axes[3].plot(after.index, (net_after - G).clip(lower=0), label="stressed")
        axes[3].axvline(event_start, color="k", linestyle="--", linewidth=0.8, label="event")
        axes[3].axvline(event_end, color="k", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("shortfall [GW]")
        axes[3].legend(fontsize=7, loc="upper right")
        axes[3].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    return (plot_event,)


@app.cell
def _(G, event_duration, event_start, lsw_df, lsw_stressed, plot_event):
    plot_event(lsw_df, lsw_stressed, event_start, event_duration, G)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sampled event
    start date: uniform over feasible

    duration: uniform (whole days)

    added shortfall: uniform (GWh)
    """)
    return


@app.cell
def _(mo):
    sample_button = mo.ui.button(label="Sample event")
    duration_min_input = mo.ui.number(start=1, stop=30, step=1, label="min duration [days]", value=1)
    duration_max_input = mo.ui.number(start=1, stop=30, step=1, label="max duration [days]", value=5)
    shortfall_min_input = mo.ui.number(start=0.0, label="min shortfall [GWh]", value=10.0)
    shortfall_max_input = mo.ui.number(start=0.0, label="max shortfall [GWh]", value=20.0)
    mo.vstack([
        duration_min_input,
        duration_max_input,
        shortfall_min_input,
        shortfall_max_input,
    ])
    return (
        duration_max_input,
        duration_min_input,
        sample_button,
        shortfall_max_input,
        shortfall_min_input,
    )


@app.cell
def _(sample_button):
    sample_button
    return


@app.cell
def _(
    G,
    duration_max_input,
    duration_min_input,
    lsw_df,
    mo,
    pd,
    sample_button,
    sample_stress_event,
    scaling_ratio_input,
    shortfall_max_input,
    shortfall_min_input,
):
    sample_button.value
    lsw_sampled, s_start, s_duration, s_shortfall = sample_stress_event(
        lsw_df,
        G=G,
        scaling_ratio=scaling_ratio_input.value,
        duration_range_days=(int(duration_min_input.value), int(duration_max_input.value)),
        shortfall_range_gwh=(shortfall_min_input.value, shortfall_max_input.value),
        start_buffer=pd.Timedelta(days=1),
    )
    _period = lsw_df.index[1] - lsw_df.index[0]
    _delta = _period.total_seconds() / 3600
    _ev_base = lsw_df.loc[s_start:s_start + s_duration - _period]
    _ev_after = lsw_sampled.loc[s_start:s_start + s_duration - _period]
    _sf_base = _delta * (_ev_base["l"] - _ev_base["s"] - _ev_base["w"] - G).clip(lower=0).sum()
    _sf_after = _delta * (_ev_after["l"] - _ev_after["s"] - _ev_after["w"] - G).clip(lower=0).sum()
    mo.md(f"**start:** {s_start.date()} | **duration:** {int(s_duration.days)} days | **baseline shortfall:** {_sf_base:.2f} GWh | **added shortfall:** {_sf_after - _sf_base:.2f} GWh")
    return lsw_sampled, s_duration, s_start


@app.cell
def _(G, lsw_df, lsw_sampled, plot_event, s_duration, s_start):
    plot_event(lsw_df, lsw_sampled, s_start, s_duration, G)
    return


if __name__ == "__main__":
    app.run()
