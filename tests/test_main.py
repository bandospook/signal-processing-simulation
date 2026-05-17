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


def _make_cfg(tmp_path):
    return {
        "carrier": [
            dict(name="c1", symbol_rate=1e6, sps=4, rolloff=0.35, filter_span=8,
                 num_symbols=100, power_db=0.0, freq=-3e6,
                 channel=dict(enabled=True, ripple_db=0.5, ripple_cycles=2.0,
                              max_phase_dev_deg=5.0, phase_poly_order=2,
                              plot="channel_c1.png")),
            dict(name="c2", symbol_rate=1e6, sps=4, rolloff=0.35, filter_span=8,
                 num_symbols=100, power_db=0.0, freq=+3e6),
        ],
        "wideband": {"sample_rate": 16e6, "noise_density_dbfs": -160.0},
        "amplifier": {"input_backoff_db": 3.0, "am_am": _AM_AM, "am_pm": _AM_PM},
        "ola": {"filter_span": 8, "block_size": 1024},
        "simulation": {"seed": 42},
        "output": {
            "output_dir": str(tmp_path),
            "wideband":  "wideband.png",
            "nl_tables": "nl.png",
            "sweep":     "sweep.png",
        },
        "sweep": {
            "ibo_db":             [3.0, 6.0],
            "noise_density_dbfs": [-160.0, -150.0],
        },
    }


def test_main_runs(tmp_path):
    """End-to-end: main runs, all four plot files are written to disk."""
    with patch("main.load_config", return_value=_make_cfg(tmp_path)), \
         patch("matplotlib.pyplot.show"):
        main_module.main()

    assert (tmp_path / "wideband.png").exists()
    assert (tmp_path / "nl.png").exists()
    assert (tmp_path / "sweep.png").exists()
    assert (tmp_path / "channel_c1.png").exists()


_MINIMAL_TOML = """\
[simulation]
seed = 99

[wideband]
sample_rate = 16_000_000

[amplifier]
input_backoff_db = 6.0

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
"""


def test_load_config(tmp_path):
    """load_config parses a TOML file and returns the expected structure."""
    path = tmp_path / "test.toml"
    path.write_text(_MINIMAL_TOML, encoding="utf-8")

    cfg = load_config(path)

    assert cfg["simulation"]["seed"] == 99
    assert cfg["wideband"]["sample_rate"] == 16_000_000
    assert cfg["amplifier"]["input_backoff_db"] == 6.0
    assert cfg["amplifier"]["am_am"]["input"] == [0.0, 1.0]
    assert cfg["amplifier"]["am_pm"]["phase_deg"] == [0.0, 5.0]
    assert cfg["ola"]["filter_span"] == 8
