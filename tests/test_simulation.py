"""System-level tests for wideband_bpsk_simulation: NL distortion vs drive level."""
import numpy as np
import pytest
from sim.simulation import wideband_bpsk_simulation, _WelchState, _decimate

# Production AM-AM / AM-PM from simulation.toml
_AM_AM = {
    "input":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "output": [0.000, 0.119, 0.238, 0.356, 0.473, 0.586, 0.692, 0.788, 0.873, 0.944, 1.000],
}
_AM_PM = {
    "input":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "phase_deg": [0.000, 0.050, 0.200, 0.450, 0.800, 1.250, 1.800, 2.450, 3.200, 4.050, 5.000],
}

# Minimal two-carrier config (two carriers needed to generate inter-carrier IM products).
# native_rate = sps * symbol_rate = 4 * 1e6 = 4 MHz → upsample factor L=4 to 16 MHz.
_CARRIERS = [
    dict(name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
         rolloff=0.35, filter_span=8, num_symbols=300, power_db=0.0, freq=-3e6),
    dict(name="c2", modulation="BPSK", symbol_rate=1e6, sps=4,
         rolloff=0.35, filter_span=8, num_symbols=300, power_db=0.0, freq=+3e6),
]
_SAMPLE_RATE = 16e6


def _run_no_noise(ibo_db: float) -> dict:
    return wideband_bpsk_simulation(
        carriers=_CARRIERS,
        sample_rate=_SAMPLE_RATE,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        input_backoff_db=ibo_db,
        noise_density_dbfs=None,      # no noise — isolates NL distortion
        ola_filter_span=8,
        ola_block_size=1024,
        seed=42,
    )


def test_simulation_returns_expected_keys():
    """Smoke test: result dict has the expected top-level and per-carrier keys."""
    result = _run_no_noise(ibo_db=6.0)
    for key in ("psd_pre_nl", "psd_post_nl", "psd_noisy", "has_noise", "carriers"):
        assert key in result, f"Missing key: {key}"
    f, p = result["psd_pre_nl"]
    assert len(f) == len(p) and len(f) > 0
    for cr in result["carriers"]:
        for key in ("name", "ber", "evm_rms", "cir_db", "cnr_db", "cnir_db"):
            assert key in cr, f"Missing per-carrier key: {key}"


def test_noiseless_cnr_is_infinite():
    """With no noise, CNR must be infinite (no noise floor)."""
    result = _run_no_noise(ibo_db=6.0)
    for cr in result["carriers"]:
        assert not np.isfinite(cr["cnr_db"]), \
            f"Expected infinite CNR with no noise, got {cr['cnr_db']:.1f} dB"


def test_welch_state_empty_result():
    """_WelchState.result() before any data is fed returns a -100 dB floor."""
    ws = _WelchState(nfft=16)
    f, psd = ws.result(sample_rate=1.0)
    assert len(f) == 16
    assert np.all(psd == -100.0)


def test_decimate_offset_ge_len():
    """_decimate returns empty array and adjusts offset when offset >= len(filtered)."""
    sig = np.ones(3, dtype=complex)
    out, new_offset = _decimate(sig, L=2, offset=5)
    assert len(out) == 0
    assert new_offset == 5 - 3


def test_simulation_raises_sample_rate_below_native():
    """Carrier with sample_rate < sps * symbol_rate must raise ValueError."""
    bad_carrier = dict(name="c1", modulation="BPSK", symbol_rate=1e6, sps=8,
                       rolloff=0.35, filter_span=4, num_symbols=50,
                       power_db=0.0, freq=0.0)
    with pytest.raises(ValueError, match="sample_rate / native_rate"):
        wideband_bpsk_simulation(
            carriers=[bad_carrier],
            sample_rate=4e6,            # 4 MHz < 8 MHz native rate
            am_am_cfg={"input": [0.0, 1.0], "output": [0.0, 1.0]},
            am_pm_cfg={"input": [0.0, 1.0], "phase_deg": [0.0, 0.0]},
            input_backoff_db=6.0,
        )


def test_distortion_increases_with_drive():
    """
    CIR must decrease monotonically as input backoff decreases (harder drive
    → more NL compression → more inter-carrier IM distortion).

    Runs the wideband simulation at 5 IBO levels with no AWGN so that only
    NL-induced distortion contributes to CIR.
    """
    ibos = [12.0, 9.0, 6.0, 3.0, 0.0]   # decreasing = harder drive
    cirs = [
        np.mean([cr["cir_db"] for cr in _run_no_noise(ibo)["carriers"]])
        for ibo in ibos
    ]
    for i in range(1, len(cirs)):
        assert cirs[i] < cirs[i-1], (
            f"CIR should decrease as drive increases: "
            f"IBO={ibos[i]:.0f} dB → CIR={cirs[i]:.1f} dB "
            f"not < IBO={ibos[i-1]:.0f} dB → CIR={cirs[i-1]:.1f} dB"
        )
