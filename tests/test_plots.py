"""Tests for sim.plots text-output and helper functions.

Covers: psd_db, _fmt_metric, _enabled_carrier_names, write_report,
        plot_carrier_detector.
Rendering functions (plot_*) are mostly exercised indirectly through
test_main.py; here we add the targeted no-data / single-IBO paths.
"""
import numpy as np
from pathlib import Path

from sim.plots import (
    psd_db,
    plot_carrier_detector,
    write_report,
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


def test_plot_carrier_detector_no_data_early_return():
    """plot_carrier_detector returns before creating any figure when the named
    carrier has non-finite metrics across the sweep."""
    plot_carrier_detector(_SWEEP_NAN, "c1", save_path=None)


def test_plot_carrier_detector_unknown_carrier_early_return():
    """plot_carrier_detector returns silently when asked for a carrier that is
    not in the sweep results at all."""
    plot_carrier_detector(_SWEEP_FINITE, "missing", save_path=None)


def test_plot_carrier_detector_single_ibo():
    """plot_carrier_detector with a single IBO value skips set_xlim rather than
    triggering the 'identical low and high xlims' UserWarning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_carrier_detector(_SWEEP_FINITE, "c1", save_path=None)


def test_plot_carrier_detector_zero_ber_with_inf_cnr():
    """CNR-row zero-BER annotation skips points where cnr_db is non-finite
    (e.g. zero-noise → CNR = inf), avoiding an inf x-coordinate annotation."""
    sweep = [
        {"ibo_db": 3.0, "noise_density_dbfs": -160.0,
         "carriers": [{"name": "c1", "cnir_db": 40.0, "cnr_db": float("inf"),
                       "cir_db": 40.0, "ber": 0.0, "evm_rms": 3.0}]},
    ]
    plot_carrier_detector(sweep, "c1", save_path=None)


# ── write_report ──────────────────────────────────────────────────────────────

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


def test_write_report_no_path():
    """save_path=None → early return."""
    write_report(_ROWS_NO_THEORY, save_path=None)


def test_write_report_empty():
    """Empty rows list → early return."""
    write_report([], save_path="irrelevant.md")


def test_write_report_basic(tmp_path):
    """Row with theory=None formats those columns as em-dash; IBO column included."""
    path = str(tmp_path / "report.md")
    write_report(_ROWS_NO_THEORY, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "## Results" in text
    assert "IBO (dB)" in text
    assert "| 3.0" in text         # ibo_db value formatted
    assert "—" in text             # theory_ebn0_db is None


def test_write_report_zero_ber_no_upper_renders_as_0(tmp_path):
    """ber == 0 without a ber_upper_95 falls through to the literal '0' branch."""
    rows = [dict(_ROWS_NO_THEORY[0])]
    rows[0]["ber_upper_95"] = None
    path = str(tmp_path / "report.md")
    write_report(rows, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    # Cell appears between pipes — anchor on the surrounding pipes to avoid
    # matching the literal "0" elsewhere (e.g. in IBO 3.0).
    assert "| 0 |" in text or "| 0 " in text


def test_write_report_zero_ber_with_upper_renders_as_lt(tmp_path):
    """ber == 0 with a ber_upper_95 set renders as '< x.xe-y'."""
    rows = [dict(_ROWS_NO_THEORY[0])]
    rows[0]["ber_upper_95"] = 3e-6
    path = str(tmp_path / "report.md")
    write_report(rows, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "< 3.00e-06" in text


def test_write_report_capped_iter_marks_asterisk(tmp_path):
    """converged=False appends '*' to the iteration count cell."""
    rows = [dict(_ROWS_NO_THEORY[0])]
    rows[0]["iterations"] = 100
    rows[0]["converged"]  = False
    path = str(tmp_path / "report.md")
    write_report(rows, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "100*" in text


def test_write_report_ber_none_renders_em_dash(tmp_path):
    """ber=None (e.g. demod skipped) renders as '—' in the BER cell."""
    rows = [dict(_ROWS_NO_THEORY[0])]
    rows[0]["ber"] = None
    path = str(tmp_path / "report.md")
    write_report(rows, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    # The BER cell sits between two pipes
    assert "| — |" in text or "| —" in text


def test_write_report_with_theory(tmp_path):
    """Row with theory and impl-loss values formats them as floats."""
    path = str(tmp_path / "report.md")
    write_report(_ROWS_WITH_THEORY, save_path=path)
    text = Path(path).read_text(encoding="utf-8")
    assert "1.00e-02" in text      # BER scientific notation
    assert "9.40" in text          # theory_ebn0_db
    assert "0.10" in text          # implementation_loss_db


def test_write_report_inf_cnr(tmp_path):
    """Infinite CNR value is formatted as ∞."""
    rows = [dict(_ROWS_NO_THEORY[0], cnr_db=float("inf"))]
    path = str(tmp_path / "report.md")
    write_report(rows, save_path=path)
    assert "∞" in Path(path).read_text(encoding="utf-8")


def test_write_report_append(tmp_path):
    """append=True adds content after existing file text."""
    path = str(tmp_path / "report.md")
    Path(path).write_text("existing\n", encoding="utf-8")
    write_report(_ROWS_NO_THEORY, save_path=path, append=True)
    text = Path(path).read_text(encoding="utf-8")
    assert text.startswith("existing")
    assert "## Results" in text
