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


def test_accumulator_relative_target_meets_at_high_ber():
    """target_rel can declare convergence when the absolute target is far too
    tight to ever be met at the operating BER but the relative target is."""
    a = _ErrorAccumulator()
    # 50000 trials, 500 errors → BER = 0.01, Wilson half-width ≈ 8.7e-4.
    a.add(n_bits=50_000, n_errors=500)
    # Absolute target unreachable; without target_rel we don't converge.
    assert not a.converged(target=1e-6, confidence=0.95, min_errors=50)
    # Relative target of 10% (hw/ber ≤ 0.10): 8.7e-4 / 1e-2 ≈ 0.087 < 0.10 → meets.
    assert a.converged(target=1e-6, confidence=0.95, min_errors=50,
                       target_rel=0.10)


def test_accumulator_relative_target_does_not_meet_with_few_errors():
    """target_rel still requires min_errors and a wide enough relative ratio."""
    a = _ErrorAccumulator()
    # Few trials, single error → BER = 0.01 but hw ≈ 0.055; ratio ≈ 5.5 ≫ 0.10.
    a.add(n_bits=100, n_errors=1)
    assert not a.converged(target=1e-6, confidence=0.95, min_errors=0,
                           target_rel=0.10)


def test_accumulator_relative_target_inert_when_zero_errors():
    """With k=0 the relative test is undefined; only the absolute can converge."""
    a = _ErrorAccumulator()
    a.add(n_bits=10_000, n_errors=0)
    # Absolute hw at k=0, n=10000, 95%: ≈ 3.8e-4 — meets 1e-3 target.
    assert a.converged(target=1e-3, confidence=0.95, min_errors=0,
                       target_rel=0.01)
    # Tight absolute target → no convergence even with a generous relative one
    # (the latter does nothing when BER is zero).
    assert not a.converged(target=1e-9, confidence=0.95, min_errors=0,
                           target_rel=0.10)


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
    """chunk_print receives a unified status line for both in-flight chunks
    and post-iteration tallies; both forms share fixed-width columns for the
    cumulative bits/errors/BER/CI fields."""
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
    chunk_lines = [ln for ln in lines if ": chunk " in ln]
    tally_lines = [ln for ln in lines if ": done" in ln]
    assert any(ln.startswith("iter 1/2: chunk") for ln in chunk_lines)
    assert any(ln.startswith("iter 2/2: chunk") for ln in chunk_lines)
    assert len(tally_lines) == 2
    assert tally_lines[0].startswith("iter 1/2: done")
    assert tally_lines[1].startswith("iter 2/2: done")
    # Every status line — chunk or done — carries the running carrier tail.
    assert all("c1" in ln for ln in chunk_lines + tally_lines)
    # Iter-1 chunk lines fire BEFORE iter 1's demod runs, so the cumulative
    # accumulator is still empty → bits=0 and BER/CI render as '---'.
    iter1_chunks = [ln for ln in chunk_lines if ln.startswith("iter 1/2: chunk")]
    assert iter1_chunks, "expected at least one iter-1 chunk line"
    assert "bits=         0" in iter1_chunks[0]
    assert "BER=     ---" in iter1_chunks[0]
    assert "CI±=    ---" in iter1_chunks[0]
    # Iter-2 chunk lines see iter-1's stale tally (n_bits > 0 from iter 1).
    iter2_chunks = [ln for ln in chunk_lines if ln.startswith("iter 2/2: chunk")]
    assert iter2_chunks, "expected at least one iter-2 chunk line"
    assert "BER=0.00e+00" in iter2_chunks[0]
    # Zero errors at -200 dBFS/Hz → BER=0.00e+00 with explicit errors=0.
    assert "errors=       0" in tally_lines[0]
    assert "BER=0.00e+00" in tally_lines[0]
    # All cumulative columns line up vertically across chunk and done lines.
    bits_col = tally_lines[0].index("bits=")
    err_col  = tally_lines[0].index("errors=")
    ber_col  = tally_lines[0].index("BER=")
    ci_col   = tally_lines[0].index("CI±=")
    for ln in chunk_lines + tally_lines:
        assert ln.index("bits=")   == bits_col
        assert ln.index("errors=") == err_col
        assert ln.index("BER=")    == ber_col
        assert ln.index("CI±=")    == ci_col
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
    tally_lines = [ln for ln in lines if ": done" in ln]
    assert len(tally_lines) == 1
    assert tally_lines[0].endswith("(target met)")


def test_parameter_sweep_relative_target_short_circuits_high_ber_point():
    """At a high-BER operating point the relative target can be met long
    before the absolute target — the sweep exits as soon as either fires."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    _, capped = parameter_sweep(
        carriers=carriers, sample_rate=16e6,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0],
        noise_density_dbfs_values=[-40.0],   # extreme noise → ~half BER on BPSK
        max_block_size_samples=2000,
        target_ci_half_width=1e-6,           # absolute unreachable here
        confidence=0.95, min_errors=5, max_iterations=3,
        ola_filter_span=8, ola_block_size=1024, seed=0,
    )
    assert capped[0]["converged"] is False
    assert capped[0]["iterations"] == 3
    # With a generous relative target convergence is declared on iter 1.
    _, met = parameter_sweep(
        carriers=carriers, sample_rate=16e6,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0], noise_density_dbfs_values=[-40.0],
        max_block_size_samples=2000,
        target_ci_half_width=1e-6, target_ci_relative=10.0,
        confidence=0.95, min_errors=5, max_iterations=3,
        ola_filter_span=8, ola_block_size=1024, seed=0,
    )
    assert met[0]["converged"] is True
    assert met[0]["iterations"] == 1


def test_parameter_sweep_point_summary_carries_ber_ci_ebn0(capsys):
    """The per-point summary line printed at each sweep point includes the
    carrier name, BER (or rule-of-three bound), CI half-width, and Eb/N0."""
    carriers = [dict(
        name="c1", modulation="BPSK", symbol_rate=1e6, sps=4,
        rolloff=0.35, filter_span=8, power_db=0.0, freq=0.0,
        sweep_demod=True,
    )]
    # Noisy point → BER > 0 → numeric BER + CI on the summary line.
    parameter_sweep(
        carriers=carriers, sample_rate=16e6,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0], noise_density_dbfs_values=[-40.0],
        max_block_size_samples=2000,
        target_ci_half_width=0.5, confidence=0.95,
        min_errors=0, max_iterations=1,
        ola_filter_span=8, ola_block_size=1024, seed=0,
    )
    out = capsys.readouterr().out
    summary = next(ln for ln in out.splitlines() if ln.lstrip().startswith("["))
    assert "c1: BER=" in summary
    assert "CI±=" in summary
    assert "Eb/N0=" in summary
    assert " dB" in summary.split("Eb/N0=", 1)[1]

    # Zero-errors point → BER is rendered as a rule-of-three upper bound.
    parameter_sweep(
        carriers=carriers, sample_rate=16e6,
        am_am_cfg=_AM_AM, am_pm_cfg=_AM_PM,
        ibo_db_values=[6.0], noise_density_dbfs_values=[-200.0],
        max_block_size_samples=400,
        target_ci_half_width=0.5, confidence=0.95,
        min_errors=0, max_iterations=1,
        ola_filter_span=8, ola_block_size=1024, seed=0,
    )
    out = capsys.readouterr().out
    summary = next(ln for ln in out.splitlines() if ln.lstrip().startswith("["))
    assert "BER<" in summary
    assert "Eb/N0=" in summary


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
