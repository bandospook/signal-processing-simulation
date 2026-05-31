"""System-level tests for simulate: NL distortion vs drive level."""
import math
import numpy as np
import pytest
from sim.simulation import simulate, _WelchState, _decimate

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
         rolloff=0.35, filter_span=8, power_db=0.0, freq=-3e6),
    dict(name="c2", modulation="BPSK", symbol_rate=1e6, sps=4,
         rolloff=0.35, filter_span=8, power_db=0.0, freq=+3e6),
]
_SAMPLE_RATE = 16e6
# Budget chosen so each carrier's native buffer = 300 symbols × 4 sps = 1200 samples.
_BUDGET = 1200


def _run_no_noise(ibo_db: float) -> dict:
    return simulate(
        carriers=_CARRIERS,
        sample_rate=_SAMPLE_RATE,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        max_block_size_samples=_BUDGET,
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
                       rolloff=0.35, filter_span=4,
                       power_db=0.0, freq=0.0)
    with pytest.raises(ValueError, match="sample_rate / native_rate"):
        simulate(
            carriers=[bad_carrier],
            sample_rate=4e6,            # 4 MHz < 8 MHz native rate
            am_am_cfg={"input": [0.0, 1.0], "output": [0.0, 1.0]},
            am_pm_cfg={"input": [0.0, 1.0], "phase_deg": [0.0, 0.0]},
            max_block_size_samples=400,
            input_backoff_db=6.0,
        )


def test_coded_carrier_decodes():
    """A convolutionally-coded carrier is FEC-encoded into the chain and decoded out.

    block_length=400 → coded_bits=(400+6)*2=812 per frame; sps=4 → ~3248 samples/frame.
    Budget of 7000 fits 2 frames.
    """
    coded = dict(name="cc", modulation="BPSK", symbol_rate=1e6, sps=4,
                 rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
                 coding=dict(scheme="convolutional", block_length=400))
    result = simulate(
        carriers=[coded], sample_rate=16e6,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        max_block_size_samples=7000,
        input_backoff_db=12.0,
        noise_density_dbfs=-100.0,
        ola_filter_span=8, ola_block_size=1024, seed=1,
        demod_carriers={"cc"})
    cr = result["carriers"][0]
    assert cr["ber"] is not None and 0.0 <= cr["ber"] <= 1.0   # post-decoder BER
    assert "uncoded_ber" in cr
    assert cr["ber"] < 0.05                                    # coded carrier decodes
    assert cr["n_bits"] > 0 and cr["uncoded_n_bits"] > 0       # error counts exposed


def test_derive_block_counts_uncoded():
    """Uncoded carrier: num_symbols = budget // sps, n_frames=0, code=None."""
    from sim.simulation import _derive_block_counts
    carr = dict(name="x")    # no coding key
    num_symbols, n_frames, code = _derive_block_counts(
        carr, sps=4, bps=1, budget_samples=4000)
    assert num_symbols == 1000
    assert n_frames == 0
    assert code is None


def test_derive_block_counts_ldpc():
    """LDPC branch builds the generator before computing frame count."""
    from sim.simulation import _derive_block_counts
    carr = dict(name="x", coding=dict(scheme="ldpc"))   # default matrix
    num_symbols, n_frames, code = _derive_block_counts(
        carr, sps=4, bps=1, budget_samples=200_000)
    assert n_frames >= 1
    assert code is not None
    assert code.k > 0                       # generator was built
    assert num_symbols == n_frames * math.ceil(code.coded_bits / 1)


def test_phase_noise_degrades_evm():
    """Adding phase noise rotates received symbols off the constellation, raising EVM vs. the baseline.

    Phase noise is intentionally applied to bb_ch BEFORE the chunk pipeline
    forks into reference / NL / noisy branches, so it travels with the signal
    through every branch.  The projection-based CIR therefore cannot see it
    (both ref and NL carry the same φ(t) and cancel in the projection); EVM
    measures actual deviation from the ideal constellation and does pick it up.

    Phase noise lives in each carrier's own config block.  Only one of the
    two carriers gets it here so the per-carrier wiring is exercised — the
    untouched carrier should keep its baseline EVM.
    """
    base = _run_no_noise(ibo_db=12.0)
    base_evms = {cr["name"]: cr["evm_rms"] for cr in base["carriers"]}

    pn_cfg = {
        "enabled":    True,
        "offset_hz":  [1e3, 1e4, 1e5, 1e6],
        "dbc_per_hz": [-40.0, -60.0, -80.0, -100.0],   # aggressive mask
    }
    carriers_with_pn = [
        {**_CARRIERS[0], "phase_noise": pn_cfg},
        {**_CARRIERS[1]},                              # no phase noise on c2
    ]
    with_pn = simulate(
        carriers=carriers_with_pn, sample_rate=_SAMPLE_RATE,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        max_block_size_samples=_BUDGET,
        input_backoff_db=12.0, noise_density_dbfs=None,
        ola_filter_span=8, ola_block_size=1024, seed=42,
    )
    pn_evms = {cr["name"]: cr["evm_rms"] for cr in with_pn["carriers"]}
    assert pn_evms["c1"] > base_evms["c1"] * 2.0, (
        f"Phase noise should at least double EVM on c1: "
        f"baseline {base_evms['c1']:.2f}% -> with PN {pn_evms['c1']:.2f}%"
    )
    # c2 has no [phase_noise] block → its EVM stays at the baseline.
    assert abs(pn_evms["c2"] - base_evms["c2"]) < 0.5, (
        f"c2 has no phase_noise block and should match baseline EVM: "
        f"baseline {base_evms['c2']:.2f}% vs got {pn_evms['c2']:.2f}%"
    )


def test_distortion_increases_with_drive():
    """CIR must decrease monotonically as input backoff decreases.

    Harder drive → more NL compression → more inter-carrier IM distortion.
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
