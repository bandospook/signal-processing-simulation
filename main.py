"""Entry point for the wideband BPSK nonlinear amplifier simulation."""

import matplotlib.pyplot as plt

from config import load_config
from simulation import wideband_bpsk_simulation
from plots import plot_wideband_results, plot_nl_tables, plot_channel_response


def main():
    """Load config, run wideband simulation, and plot results."""
    plt.close('all')

    cfg = load_config("simulation.toml")

    sig = cfg["signals"]
    wb  = cfg["wideband"]
    amp = cfg["amplifier"]
    ola = cfg["ola"]
    sim = cfg["simulation"]
    out = cfg["output"]
    ch  = cfg.get("channel", {})

    results = wideband_bpsk_simulation(
        num_symbols_slow  = sim["num_symbols_slow"],
        symbol_rate_slow  = sig["symbol_rate_slow"],
        sample_rate       = wb["sample_rate"],
        freq_slow         = wb["freq_slow"],
        freq_fast         = wb["freq_fast"],
        sps               = sig["sps"],
        rolloff           = sig["rolloff"],
        filter_span       = sig["filter_span"],
        input_backoff_db  = amp["input_backoff_db"],
        am_am_cfg         = amp["am_am"],
        am_pm_cfg         = amp["am_pm"],
        channel_slow_cfg  = ch.get("slow"),
        channel_fast_cfg  = ch.get("fast"),
        rate_ratio        = sig["rate_ratio"],
        ola_filter_span   = ola["filter_span"],
        ola_block_size    = ola["block_size"],
        seed              = sim["seed"],
    )

    native_rate_slow = sig["sps"] * sig["symbol_rate_slow"]
    native_rate_fast = sig["sps"] * sig["rate_ratio"] * sig["symbol_rate_slow"]
    signal_bw_slow   = (1 + sig["rolloff"]) * sig["symbol_rate_slow"]
    signal_bw_fast   = (1 + sig["rolloff"]) * sig["rate_ratio"] * sig["symbol_rate_slow"]

    plot_wideband_results(
        results,
        sample_rate      = wb["sample_rate"],
        native_rate_slow = native_rate_slow,
        native_rate_fast = native_rate_fast,
        symbol_rate_slow = sig["symbol_rate_slow"],
        symbol_rate_fast = sig["rate_ratio"] * sig["symbol_rate_slow"],
        save_path        = out.get("save_path"),
    )

    plot_nl_tables(
        amp["am_am"],
        amp["am_pm"],
        input_backoff_db = amp["input_backoff_db"],
    )

    slow_cfg = ch.get("slow")
    if slow_cfg and slow_cfg.get("enabled", True):
        plot_channel_response(
            native_rate_slow, signal_bw_slow, slow_cfg,
            title=f"Slow carrier  ({sig['symbol_rate_slow']/1e6:.3g} MHz sym/s)",
        )

    fast_cfg = ch.get("fast")
    if fast_cfg and fast_cfg.get("enabled", True):
        plot_channel_response(
            native_rate_fast, signal_bw_fast, fast_cfg,
            title=f"Fast carrier  ({sig['rate_ratio'] * sig['symbol_rate_slow']/1e6:.3g} MHz sym/s)",
        )

    plt.show()


if __name__ == "__main__":
    main()
