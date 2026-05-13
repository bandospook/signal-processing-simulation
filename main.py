"""Entry point for the wideband BPSK nonlinear amplifier simulation."""

import matplotlib.pyplot as plt
from pathlib import Path

from sim.config import load_config
from sim.simulation import wideband_bpsk_simulation
from sim.plots import (plot_wideband_results, plot_nl_tables, plot_channel_response,
                       print_metrics_table, plot_sweep_results)
from sim.sweep import parameter_sweep


def main():
    """Load config, run wideband simulation, and plot results."""
    plt.close('all')

    cfg = load_config("simulation.toml")

    carriers = cfg["carrier"]       # list from [[carrier]] blocks
    wb  = cfg["wideband"]
    amp = cfg["amplifier"]
    ola = cfg["ola"]
    sim = cfg["simulation"]
    out = cfg["output"]

    out_dir = Path(out.get("output_dir", "."))
    out_dir.mkdir(exist_ok=True)

    def out_path(name: str | None) -> str | None:
        return str(out_dir / name) if name else None

    results = wideband_bpsk_simulation(
        carriers           = carriers,
        sample_rate        = wb["sample_rate"],
        am_am_cfg          = amp["am_am"],
        am_pm_cfg          = amp["am_pm"],
        input_backoff_db   = amp["input_backoff_db"],
        noise_density_dbfs = wb.get("noise_density_dbfs"),
        ola_filter_span    = ola["filter_span"],
        ola_block_size     = ola["block_size"],
        seed               = sim["seed"],
    )

    print_metrics_table(results["carriers"])

    plot_wideband_results(results, sample_rate=wb["sample_rate"],
                          save_path=out_path(out.get("wideband")))

    plot_nl_tables(amp["am_am"], amp["am_pm"],
                   input_backoff_db=amp["input_backoff_db"],
                   save_path=out_path(out.get("nl_tables")))

    for carr in carriers:
        ch_cfg = carr.get("channel")
        if ch_cfg and ch_cfg.get("enabled", True):
            native_rate = carr["sps"] * carr["symbol_rate"]
            signal_bw   = (1 + carr["rolloff"]) * carr["symbol_rate"]
            plot_channel_response(
                native_rate, signal_bw, ch_cfg,
                title=f"{carr['name']}  ({carr['symbol_rate']/1e6:.3g} MHz sym/s)",
                save_path=out_path(ch_cfg.get("plot")),
            )

    sweep_cfg = cfg.get("sweep", {})
    ibo_sweep   = sweep_cfg.get("ibo_db", [])
    noise_sweep = sweep_cfg.get("noise_density_dbfs", [])
    if len(ibo_sweep) > 0 and len(noise_sweep) > 0:
        print(f"\nRunning sweep: {len(ibo_sweep)} IBO × {len(noise_sweep)} noise "
              f"= {len(ibo_sweep) * len(noise_sweep)} points …")
        sweep_results = parameter_sweep(
            carriers=carriers,
            sample_rate=wb["sample_rate"],
            am_am_cfg=amp["am_am"],
            am_pm_cfg=amp["am_pm"],
            ibo_db_values=ibo_sweep,
            noise_density_dbfs_values=noise_sweep,
            ola_filter_span=ola["filter_span"],
            ola_block_size=ola["block_size"],
            seed=sim["seed"],
        )
        plot_sweep_results(sweep_results, save_path=out_path(out.get("sweep")))

    plt.show()


if __name__ == "__main__":
    main()
