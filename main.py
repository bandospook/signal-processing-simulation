"""Entry point for the wideband BPSK nonlinear amplifier simulation."""

import matplotlib.pyplot as plt

from config import load_config
from simulation import wideband_bpsk_simulation
from plots import plot_wideband_results, plot_nl_tables, plot_channel_response


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

    plot_wideband_results(results, sample_rate=wb["sample_rate"],
                          save_path=out.get("save_path"))

    plot_nl_tables(amp["am_am"], amp["am_pm"],
                   input_backoff_db=amp["input_backoff_db"])

    for carr in carriers:
        ch_cfg = carr.get("channel")
        if ch_cfg and ch_cfg.get("enabled", True):
            native_rate = carr["sps"] * carr["symbol_rate"]
            signal_bw   = (1 + carr["rolloff"]) * carr["symbol_rate"]
            plot_channel_response(
                native_rate, signal_bw, ch_cfg,
                title=f"{carr['name']}  ({carr['symbol_rate']/1e6:.3g} MHz sym/s)",
            )

    plt.show()


if __name__ == "__main__":
    main()
