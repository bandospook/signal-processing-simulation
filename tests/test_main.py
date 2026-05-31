"""Smoke tests for main(), load_config(), and plot file-save branches."""
from unittest.mock import patch
import main as main_module
from sim.config import load_config

_AM_AM = {
    "input":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "output": [0.000, 0.119, 0.238, 0.356, 0.473, 0.586, 0.692,
               0.788, 0.873, 0.944, 1.000],
}
_AM_PM = {
    "input":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "phase_deg": [0.000, 0.050, 0.200, 0.450, 0.800, 1.250, 1.800,
                  2.450, 3.200, 4.050, 5.000],
}


def _make_cfg(tmp_path, extra_carriers=None, plots=True):
    carriers = [
        dict(name="c1", symbol_rate=1e6, sps=4, rolloff=0.35, filter_span=8,
             power_db=0.0, freq=-3e6, sweep_demod=True,
             channel=dict(enabled=True, ripple_db=0.5, ripple_cycles=2.0,
                          max_phase_dev_deg=5.0, phase_poly_order=2)),
        # c2 has sweep_demod=False (default): contributes to the composite but is
        # not demodulated, exercising the skip branches in main.py and simulation.py.
        dict(name="c2", symbol_rate=1e6, sps=4, rolloff=0.35, filter_span=8,
             power_db=0.0, freq=+3e6),
    ]
    if extra_carriers:
        carriers.extend(extra_carriers)
    return {
        "carrier": carriers,
        "sweep": {
            "sample_rate":        16e6,
            "ibo_db":             [3.0, 6.0],
            "noise_density_dbfs": [-160.0, -150.0],
        },
        "amplifier": {"am_am": _AM_AM, "am_pm": _AM_PM},
        "ola": {"filter_span": 8, "block_size": 1024},
        "simulation": {
            "seed":                   42,
            # Tiny budget keeps tests fast (100 symbols × 4 sps = 400 samples).
            "max_block_size_samples": 400,
            "target_ci_half_width":   0.5,   # huge → converge in 1 iteration
            "confidence":             0.95,
            "min_errors":             0,
            "max_iterations":         2,
        },
        "output": {
            "output_dir": str(tmp_path),
            "plots":      plots,
        },
    }


def test_main_runs(tmp_path):
    """End-to-end: main runs and writes fixed-named plot files."""
    with patch("main.load_config", return_value=_make_cfg(tmp_path)), \
         patch("matplotlib.pyplot.show"):
        main_module.main()

    assert (tmp_path / "wideband.png").exists()
    assert (tmp_path / "amplifier.png").exists()
    assert (tmp_path / "c1_detector.png").exists()
    assert (tmp_path / "c1_channel.png").exists()
    # Each of the six detector panels is also saved as a standalone PNG.
    for panel in ("ber_vs_ibo", "evm_vs_ibo", "db_vs_ibo",
                  "ber_vs_cnr", "evm_vs_cnr", "db_vs_cnr"):
        assert (tmp_path / f"c1_detector_{panel}.png").exists()


def test_main_writes_phase_noise_plot(tmp_path):
    """A carrier with a `[carrier.phase_noise]` block triggers a.

    `<name>_phase_noise.png` next to the channel and detector plots.
    """
    cfg = _make_cfg(tmp_path)
    cfg["carrier"][0]["phase_noise"] = {
        "enabled":    True,
        "offset_hz":  [1e3, 1e4, 1e5, 1e6],
        "dbc_per_hz": [-60.0, -80.0, -100.0, -120.0],
    }
    with patch("main.load_config", return_value=cfg), \
         patch("matplotlib.pyplot.show"):
        main_module.main()
    assert (tmp_path / "c1_phase_noise.png").exists()
    # Disabled-or-absent carriers don't get a plot.
    assert not (tmp_path / "c2_phase_noise.png").exists()


def test_main_plots_disabled(tmp_path):
    """plots=false skips image files; report.md is still written."""
    with patch("main.load_config", return_value=_make_cfg(tmp_path, plots=False)), \
         patch("matplotlib.pyplot.show"):
        main_module.main()

    assert not (tmp_path / "wideband.png").exists()
    assert not (tmp_path / "amplifier.png").exists()
    assert not (tmp_path / "c1_detector.png").exists()
    assert not (tmp_path / "c1_channel.png").exists()
    assert (tmp_path / "report.md").exists()


def test_main_carrier_name_slug(tmp_path):
    """Spaces in carrier names are replaced with underscores in plot filenames."""
    spaced = dict(name="my carrier", symbol_rate=1e6, sps=4, rolloff=0.35,
                  filter_span=8, power_db=0.0, freq=0.0,
                  modulation="BPSK", sweep_demod=True)
    cfg = _make_cfg(tmp_path, extra_carriers=[spaced])
    with patch("main.load_config", return_value=cfg), \
         patch("matplotlib.pyplot.show"):
        main_module.main()

    assert (tmp_path / "my_carrier_detector.png").exists()


def test_main_progress_callback(tmp_path):
    """progress_callback receives (frac, msg) calls throughout main()."""
    calls: list[tuple] = []
    with patch("main.load_config", return_value=_make_cfg(tmp_path)), \
         patch("matplotlib.pyplot.show"):
        main_module.main(progress_callback=lambda f, m: calls.append((f, m)))

    assert len(calls) > 0
    fracs = [f for f, _ in calls]
    assert fracs[0] == 0.0
    assert fracs[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fracs)


def test_main_demod_writes_report(tmp_path):
    """A carrier with sweep_demod=True is demodulated and writes report.md."""
    fixed_carrier = dict(
        name="fd", symbol_rate=1e6, sps=4, rolloff=0.35, filter_span=8,
        power_db=0.0, freq=0.0,
        modulation="BPSK", sweep_demod=True,
    )
    cfg = _make_cfg(tmp_path, extra_carriers=[fixed_carrier])
    cfg["output"]["report"] = "report.md"

    with patch("main.load_config", return_value=cfg), \
         patch("matplotlib.pyplot.show"):
        main_module.main()

    assert (tmp_path / "report.md").exists()


def test_main_raises_on_empty_sweep(tmp_path):
    """main() rejects a config with no sweep points configured."""
    cfg = _make_cfg(tmp_path)
    cfg["sweep"]["ibo_db"] = []
    with patch("main.load_config", return_value=cfg), \
         patch("matplotlib.pyplot.show"):
        try:
            main_module.main()
        except ValueError as e:
            assert "sweep" in str(e).lower()
            return
        raise AssertionError("Expected ValueError for empty sweep")


_MINIMAL_TOML = """\
[simulation]
seed = 99

[sweep]
sample_rate        = 16
ibo_db             = [6.0]
noise_density_dbfs = [-160.0]

[amplifier.am_am]
input  = [0.0, 1.0]
output = [0.0, 1.0]

[amplifier.am_pm]
input     = [0.0, 1.0]
phase_deg = [0.0, 5.0]

[ola]
filter_span = 8
block_size  = 1024

[output]
output_dir = "."

[[carrier]]
name        = "test"
symbol_rate = 1
sps         = 4
rolloff     = 0.35
filter_span = 8
power_db    = 0.0
freq        = -3
"""


def test_load_config(tmp_path):
    """load_config parses a TOML file and converts MHz fields to Hz."""
    path = tmp_path / "test.toml"
    path.write_text(_MINIMAL_TOML, encoding="utf-8")

    cfg = load_config(path)

    assert cfg["simulation"]["seed"] == 99
    # Defaults populated for fields the user didn't set
    assert cfg["simulation"]["max_block_size_samples"] == 16_777_216
    assert cfg["simulation"]["confidence"] == 0.95
    assert cfg["sweep"]["sample_rate"] == 16_000_000  # 16 MHz -> Hz
    assert cfg["sweep"]["ibo_db"] == [6.0]
    assert cfg["sweep"]["noise_density_dbfs"] == [-160.0]
    assert cfg["amplifier"]["am_am"]["input"] == [0.0, 1.0]
    assert cfg["amplifier"]["am_pm"]["phase_deg"] == [0.0, 5.0]
    assert cfg["ola"]["filter_span"] == 8
    assert cfg["carrier"][0]["symbol_rate"] == 1_000_000  # 1 MHz -> Hz
    assert cfg["carrier"][0]["freq"] == -3_000_000        # -3 MHz -> Hz
