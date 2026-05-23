"""Tests for sim.plots text-output and helper functions.

Covers: psd_db, _fmt_metric, _enabled_carrier_names,
        write_sweep_report, write_detector_results.
Rendering functions (plot_*) are not tested here — they require a display
and are exercised indirectly through test_main.py.
"""
import numpy as np
from pathlib import Path

from sim.plots import (
    psd_db,
    plot_sweep_results,
    write_sweep_report,
    write_detector_results,
    _fmt_metric,
    _enabled_carrier_names,
)


# ── psd_db ────────────────────────────────────────────────────────────────────

def test_psd_db_shape():
    sig = np.exp(2j * np.pi * 0.1 * np.arange(4096))
    f, psd = psd_db(sig, fs=1.0, nfft=512)
    assert len(f) == len(psd) == 512


def test_psd_db_signal_shorter_than_nfft():
    """When the signal is shorter than nfft the output length equals len(sig)."""
    sig = np.ones(100, dtype=complex)
    f, psd = psd_db(sig, fs=1.0, nfft=512)
    assert len(f) == len(psd) == 100


# ── _fmt_metric ───────────────────────────────────────────────────────────────

def test_fmt_metric_none():
    assert _fmt_metric("cnr_db", None) == "—"

def test_fmt_metric_nan():
    assert _fmt_metric("cnr_db", float("nan")) == "—"

def test_fmt_metric_inf():
    assert _fmt_metric("cnr_db", float("inf")) == "∞"

def test_fmt_metric_ber_zero():
    assert _fmt_metric("ber", 0) == "0"

def test_fmt_metric_ber_nonzero():
    assert "1.00e-02" in _fmt_metric("ber", 0.01)

def test_fmt_metric_evm():
    assert "4.50" in _fmt_metric("evm_rms", 4.5)

def test_fmt_metric_normal_float():
    assert "35.5" in _fmt_metric("cnr_db", 35.5)


# ── _enabled_carrier_names ────────────────────────────────────────────────────

_SWEEP_FINITE = [
    {"ibo_db": 3.0, "noise_density_dbfs": -160.0,
     "carriers": [{"name": "c1", "cnir_db": 40.0, "cnr_db": 80.0,
                   "cir_db": 40.0, "ber": 0.0, "evm_rms": 3.0}]},
]

_SWEEP_NAN = [
    {"ibo_db": 3.0, "noise_density_dbfs": -160.0,
     "carriers": [{"name": "c1", "cnir_db": float("nan"), "cnr_db": float("nan"),
                   "cir_db": float("nan"), "ber": None, "evm_rms": None}]},
]


def test_enabled_carrier_names_with_data():
    assert _enabled_carrier_names(_SWEEP_FINITE) == ["c1"]


def test_enabled_carrier_names_no_finite_data():
    """Carrier with all-NaN cnir_db across every sweep point is excluded."""
    assert _enabled_carrier_names(_SWEEP_NAN) == []


def test_plot_sweep_results_empty_carriers_early_return():
    """plot_sweep_results returns before creating any figure when all carriers
    have non-finite metrics (carrier_names is empty)."""
    plot_sweep_results(_SWEEP_NAN, save_path=None)


def test_plot_sweep_results_single_ibo():
    """plot_sweep_results with a single IBO value skips set_xlim rather than
    triggering the 'identical low and high xlims' UserWarning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_sweep_results(_SWEEP_FINITE, save_path=None)


# ── write_sweep_report ────────────────────────────────────────────────────────

_CFG = {
    "simulation": {"seed": 42},
    "sweep": {"sample_rate": 16e6, "ibo_db": [3.0], "noise_density_dbfs": [-160.0]},
    "amplifier": {},
    "ola": {"filter_span": 8, "block_size": 1024},
    "carrier": [
        {"name": "c1", "symbol_rate": 1e6, "sps": 4,
         "num_symbols": 100, "sweep_demod": True},
    ],
}


def test_write_sweep_report_creates_file(tmp_path):
    path = str(tmp_path / "report.md")
    write_sweep_report(_SWEEP_FINITE, _CFG, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "Simulation Sweep Report" in text
    assert "c1" in text


def test_write_sweep_report_no_path():
    """save_path=None → early return, no error."""
    write_sweep_report(_SWEEP_FINITE, _CFG, save_path=None)


def test_write_sweep_report_no_finite_carriers(tmp_path):
    """All-NaN sweep: excluded carrier produces no performance-summary section,
    but Sweep Results header is still written."""
    path = str(tmp_path / "report.md")
    write_sweep_report(_SWEEP_NAN, _CFG, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "Sweep Results" in text


# ── write_detector_results ────────────────────────────────────────────────────

_ROWS_NO_THEORY = [
    dict(name="c1", ibo_db=3.0, noise_density_dbfs=-160.0, ber=0.0,
         effective_ebn0_db=12.0, theory_ebn0_db=None,
         implementation_loss_db=None, cnr_db=80.0, cir_db=40.0, cnir_db=40.0,
         evm_rms=3.0),
]

_ROWS_WITH_THEORY = [
    dict(name="c1", ibo_db=3.0, noise_density_dbfs=-120.0, ber=0.01,
         effective_ebn0_db=9.5, theory_ebn0_db=9.4,
         implementation_loss_db=0.1, cnr_db=60.0, cir_db=40.0, cnir_db=39.9,
         evm_rms=5.0),
]


def test_write_detector_results_no_path():
    """save_path=None → early return."""
    write_detector_results(_ROWS_NO_THEORY, save_path=None)


def test_write_detector_results_empty():
    """Empty rows list → early return."""
    write_detector_results([], save_path="irrelevant.md")


def test_write_detector_results_basic(tmp_path):
    """Row with theory=None formats those columns as em-dash; IBO column included."""
    path = str(tmp_path / "det.md")
    write_detector_results(_ROWS_NO_THEORY, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "Detector Results" in text
    assert "IBO (dB)" in text     # new column header
    assert "| 3.0" in text        # ibo_db value formatted
    assert "—" in text            # theory_ebn0_db is None


def test_write_detector_results_with_theory(tmp_path):
    """Row with theory and impl-loss values formats them as floats."""
    path = str(tmp_path / "det.md")
    write_detector_results(_ROWS_WITH_THEORY, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "1.00e-02" in text     # BER scientific notation
    assert "9.40" in text         # theory_ebn0_db
    assert "0.10" in text         # implementation_loss_db


def test_write_detector_results_inf_cnr(tmp_path):
    """Infinite CNR value is formatted as ∞."""
    rows = [dict(_ROWS_NO_THEORY[0], cnr_db=float("inf"))]
    path = str(tmp_path / "det.md")
    write_detector_results(rows, save_path=path)
    assert "∞" in Path(path).read_text(encoding="utf-8")


def test_write_detector_results_append(tmp_path):
    """append=True adds content after existing file text."""
    path = str(tmp_path / "det.md")
    Path(path).write_text("existing\n", encoding="utf-8")
    write_detector_results(_ROWS_NO_THEORY, save_path=path, append=True)
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("existing")
    assert "Detector Results" in text


# ── write_sweep_report: multi-carrier and missing-carrier paths ───────────────

# Two-carrier sweep where:
#   - c2 is absent from the second result  →  cr=None continue paths
#   - c1 has evm_rms=None everywhere       →  "no data" line
#   - two carriers in carrier_names        →  per-carrier sub-headers
_SWEEP_MULTI = [
    {"ibo_db": 3.0, "noise_density_dbfs": -160.0,
     "carriers": [
         {"name": "c1", "cnir_db": 40.0, "cnr_db": 80.0, "cir_db": 40.0,
          "ber": 0.0, "evm_rms": None},
         {"name": "c2", "cnir_db": 38.0, "cnr_db": 78.0, "cir_db": 38.0,
          "ber": 0.0, "evm_rms": 3.0},
     ]},
    {"ibo_db": 6.0, "noise_density_dbfs": -160.0,
     "carriers": [
         {"name": "c1", "cnir_db": 45.0, "cnr_db": 85.0, "cir_db": 45.0,
          "ber": 0.0, "evm_rms": None},
         # c2 intentionally absent → cr=None paths
     ]},
]

_CFG_MULTI = {
    "simulation": {"seed": 42},
    "sweep": {"sample_rate": 16e6, "ibo_db": [3.0, 6.0], "noise_density_dbfs": [-160.0]},
    "amplifier": {},
    "ola": {"filter_span": 8, "block_size": 1024},
    "carrier": [
        {"name": "c1", "symbol_rate": 1e6, "sps": 4,
         "num_symbols": 100, "sweep_demod": True},
        {"name": "c2", "symbol_rate": 1e6, "sps": 4,
         "num_symbols": 100, "sweep_demod": True},
    ],
}


def test_write_sweep_report_multi_carrier(tmp_path):
    """Two-carrier report: per-carrier sub-headers written; missing-carrier
    and no-data-metric branches exercised."""
    path = str(tmp_path / "report.md")
    write_sweep_report(_SWEEP_MULTI, _CFG_MULTI, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "### c1" in text
    assert "### c2" in text
    assert "no data" in text         # EVM has no values for c1
