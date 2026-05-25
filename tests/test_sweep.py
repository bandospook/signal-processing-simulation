"""Tests for the adaptive parameter-sweep accumulation logic in sim.sweep."""
import math

from sim.sweep import _ErrorAccumulator, parameter_sweep


_AM_AM = {"input":  [0.0, 0.5, 1.0], "output": [0.0, 0.45, 0.85]}
_AM_PM = {"input":  [0.0, 0.5, 1.0], "phase_deg": [0.0, 1.0, 3.0]}


def test_accumulator_ber_none_when_empty():
    """ber is None when no bits have been accumulated."""
    a = _ErrorAccumulator()
    assert a.ber is None
    assert a.uncoded_ber is None
    assert math.isinf(a.half_width(0.95))
    assert not a.converged(target=0.01, confidence=0.95, min_errors=0)


def test_accumulator_uncoded_ber_none_when_zero():
    """uncoded_ber is None when uncoded_n_bits == 0 (uncoded carriers don't track it)."""
    a = _ErrorAccumulator()
    a.add(n_bits=1000, n_errors=10)
    assert a.uncoded_ber is None
    assert a.ber == 0.01


def test_accumulator_uncoded_ber_when_positive():
    """uncoded_ber returns n_errors/n_bits ratio when uncoded counts > 0 (coded path)."""
    a = _ErrorAccumulator()
    a.add(n_bits=500, n_errors=5, uncoded_n_bits=1000, uncoded_n_errors=80)
    assert a.uncoded_ber == 0.08


def test_accumulator_not_converged_below_min_errors():
    """Wilson half-width can be tight but min_errors floor blocks convergence."""
    a = _ErrorAccumulator()
    a.add(n_bits=10_000_000, n_errors=5)   # tight CI, but only 5 errors
    assert not a.converged(target=1e-3, confidence=0.95, min_errors=50)
    a.add(n_bits=10_000_000, n_errors=45)   # bump total to 50
    assert a.converged(target=1e-3, confidence=0.95, min_errors=50)


def test_parameter_sweep_capped_iterations():
    """When the BER is too low to reach min_errors, the loop exits at max_iterations
    and the result is flagged with converged=False."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    _, results = parameter_sweep(
        carriers=carriers,
        sample_rate=16e6,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0],
        noise_density_dbfs_values=[-200.0],   # essentially noiseless → 0 errors
        max_block_size_samples=400,            # 100 symbols per iter
        target_ci_half_width=1e-12,            # impossible CI target
        confidence=0.95,
        min_errors=1,                          # require ≥1 error
        max_iterations=2,                      # cap quickly
        ola_filter_span=8,
        ola_block_size=1024,
        seed=0,
    )
    pt = results[0]
    assert pt["iterations"] == 2
    assert pt["converged"] is False
    cr = pt["carriers"][0]
    # No errors expected at -200 dBFS/Hz noise → BER == 0 and an upper-95 bound set.
    assert cr["ber"] == 0.0 or cr["ber"] is None
    if cr["n_errors"] == 0 and cr["n_bits"] > 0:
        assert cr["ber_upper_95"] is not None and cr["ber_upper_95"] > 0


def test_parameter_sweep_chunk_print_carries_iter_prefix_and_tally():
    """chunk_print receives both per-iteration chunk lines (prefixed with the
    iteration count) and a cumulative tally line after each iteration completes."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    lines: list[str] = []
    _, results = parameter_sweep(
        carriers=carriers,
        sample_rate=16e6,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0],
        noise_density_dbfs_values=[-200.0],
        max_block_size_samples=400,
        target_ci_half_width=1e-12,           # impossible → drives 2 iterations
        confidence=0.95,
        min_errors=1,
        max_iterations=2,
        ola_filter_span=8,
        ola_block_size=1024,
        seed=0,
        chunk_print=lines.append,
    )
    chunk_lines = [ln for ln in lines if "chunk" in ln and "done" not in ln]
    tally_lines = [ln for ln in lines if "done:" in ln]
    assert any(ln.startswith("iter 1/2: chunk ") for ln in chunk_lines)
    assert any(ln.startswith("iter 2/2: chunk ") for ln in chunk_lines)
    assert len(tally_lines) == 2
    assert tally_lines[0].startswith("iter 1/2 done: c1 ")
    assert tally_lines[1].startswith("iter 2/2 done: c1 ")
    # At -200 dBFS/Hz we expect zero errors, so the human-readable form is used.
    assert "BER=0 (no errors)" in tally_lines[0]
    # min_errors=1 not met → no "(target met)" suffix on either iter.
    assert not any("(target met)" in ln for ln in tally_lines)
    assert results[0]["iterations"] == 2


def test_parameter_sweep_tally_flags_target_met_when_converged():
    """The tally line for the converging iteration ends with '(target met)'."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    lines: list[str] = []
    parameter_sweep(
        carriers=carriers,
        sample_rate=16e6,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0],
        noise_density_dbfs_values=[-160.0],
        max_block_size_samples=400,
        target_ci_half_width=0.5,             # trivially satisfied
        confidence=0.95,
        min_errors=0,
        max_iterations=3,
        ola_filter_span=8,
        ola_block_size=1024,
        seed=0,
        chunk_print=lines.append,
    )
    tally_lines = [ln for ln in lines if "done:" in ln]
    assert len(tally_lines) == 1
    assert tally_lines[0].endswith("(target met)")


def test_parameter_sweep_converges_in_one_iteration_with_iter_cb():
    """Permissive CI target converges in iteration 1; iter_cb is called with the
    iteration index and grid position."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    calls: list[tuple[int, int, int]] = []
    _, results = parameter_sweep(
        carriers=carriers,
        sample_rate=16e6,
        am_am_cfg=_AM_AM,
        am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0],
        noise_density_dbfs_values=[-160.0],
        max_block_size_samples=400,
        target_ci_half_width=0.5,             # huge → converges immediately
        confidence=0.95,
        min_errors=0,
        max_iterations=5,
        ola_filter_span=8,
        ola_block_size=1024,
        seed=0,
        iter_cb=lambda i, b, n: calls.append((i, b, n)),
    )
    pt = results[0]
    assert pt["iterations"] == 1
    assert pt["converged"] is True
    assert calls == [(1, 0, 0)]
