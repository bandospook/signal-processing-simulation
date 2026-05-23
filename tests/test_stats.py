"""Tests for the Wilson-CI and rule-of-three helpers in sim.stats."""
import math
import pytest

from sim.stats import rule_of_three_upper, wilson_half_width, z_score


def test_z_score_95_pct():
    """z for 95% two-sided is ~1.96."""
    assert math.isclose(z_score(0.95), 1.959964, abs_tol=1e-4)


def test_z_score_rejects_out_of_range():
    """Confidence outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError):
        z_score(0.0)
    with pytest.raises(ValueError):
        z_score(1.0)


def test_wilson_zero_n_is_inf():
    """No trials → undefined CI, return +inf so convergence checks reject."""
    assert math.isinf(wilson_half_width(0, 0))


def test_wilson_rejects_k_out_of_range():
    """k must satisfy 0 ≤ k ≤ n."""
    with pytest.raises(ValueError):
        wilson_half_width(-1, 10)
    with pytest.raises(ValueError):
        wilson_half_width(11, 10)


def test_wilson_shrinks_with_n():
    """Half-width monotonically decreases as n grows at fixed p̂."""
    p_hat = 0.01
    widths = [wilson_half_width(int(p_hat * n), n) for n in (1_000, 10_000, 100_000)]
    assert widths[0] > widths[1] > widths[2]


def test_rule_of_three_default():
    """For 95% confidence the bound is -ln(0.05)/n ≈ 2.996/n."""
    assert math.isclose(rule_of_three_upper(1000), 0.002996, abs_tol=1e-5)


def test_rule_of_three_zero_n_is_inf():
    """No trials → upper bound is +inf."""
    assert math.isinf(rule_of_three_upper(0))
