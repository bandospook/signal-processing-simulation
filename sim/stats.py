"""Statistical helpers for BER convergence and confidence intervals."""
import math

from scipy.stats import norm


def z_score(confidence: float) -> float:
    """Two-sided z-value for the given confidence level (e.g. 0.95 → 1.959964)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def wilson_half_width(k: int, n: int, confidence: float = 0.95) -> float:
    """Half-width of the Wilson score interval for a binomial proportion.

    k        : number of successes (errors)
    n        : number of trials (bits)
    confidence : two-sided confidence level, e.g. 0.95

    Returns the radius of the symmetric (about the Wilson center) interval on p.
    Returns +inf when n == 0 (no data yet).
    """
    if n <= 0:
        return float("inf")
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range [0, n={n}]")
    z = z_score(confidence)
    p_hat = k / n
    denom = 1.0 + z * z / n
    radius_num = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return radius_num / denom


def rule_of_three_upper(n: int, confidence: float = 0.95) -> float:
    """Upper bound on BER when k = 0 errors observed in n bits.

    For 95% confidence the standard rule is 3/n; this generalises to
    -ln(1-confidence) / n.  Returns +inf when n == 0.
    """
    if n <= 0:
        return float("inf")
    return -math.log(1.0 - confidence) / n
