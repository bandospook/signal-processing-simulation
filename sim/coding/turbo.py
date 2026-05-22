"""Rate-1/3 parallel-concatenated turbo code with an iterative BCJR decoder."""
import numpy as np
from numba import njit, prange


def _rsc_trellis() -> tuple[np.ndarray, np.ndarray]:
    """Trellis of the LTE/3GPP turbo constituent RSC.

    Recursive systematic convolutional code: feedback polynomial 0o13
    (1 + D^2 + D^3), feedforward 0o15 (1 + D + D^3), 3 memory bits / 8 states.
    Returns next_state[s, u] and parity[s, u] for state s and input bit u.
    """
    next_state = np.zeros((8, 2), dtype=np.int64)
    parity = np.zeros((8, 2), dtype=np.int64)
    for s in range(8):
        r0, r1, r2 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        for u in range(2):
            a = u ^ r1 ^ r2                      # recursive bit (feedback)
            parity[s, u] = a ^ r0 ^ r2           # feedforward parity
            next_state[s, u] = a | (r0 << 1) | (r1 << 2)
    return next_state, parity


# _siso runs as Numba-compiled native code, invisible to coverage.py's line
# tracer; it is exercised by the turbo tests in tests/test_coding.py.
@njit(cache=True)
def _siso(lc_sys, lc_par, la, next_state, parity, n_states):  # pragma: no cover
    """Max-log-MAP soft-in/soft-out decoder for one constituent RSC.

    Runs the forward (alpha) and backward (beta) recursions over the trellis
    with max-log branch metrics, then returns the extrinsic LLR per bit —
    the a posteriori LLR minus the a priori and systematic-channel terms.
    Each stage is normalised by its maximum to keep the metrics bounded.
    """
    k = len(lc_sys)
    neg = -1.0e18
    alpha = np.full((k + 1, n_states), neg)
    alpha[0, 0] = 0.0
    for i in range(k):
        ls = la[i] + lc_sys[i]
        lp = lc_par[i]
        for s in range(n_states):
            if alpha[i, s] <= neg * 0.5:
                continue
            for u in range(2):
                g = 0.5 * (1.0 - 2.0 * u) * ls + 0.5 * (1.0 - 2.0 * parity[s, u]) * lp
                val = alpha[i, s] + g
                sp = next_state[s, u]
                if val > alpha[i + 1, sp]:
                    alpha[i + 1, sp] = val
        amax = neg
        for s in range(n_states):
            if alpha[i + 1, s] > amax:
                amax = alpha[i + 1, s]
        for s in range(n_states):
            if alpha[i + 1, s] > neg * 0.5:
                alpha[i + 1, s] -= amax

    beta = np.full((k + 1, n_states), neg)
    for s in range(n_states):
        beta[k, s] = 0.0                          # unterminated: uniform end state
    for i in range(k - 1, -1, -1):
        ls = la[i] + lc_sys[i]
        lp = lc_par[i]
        bmax = neg
        for s in range(n_states):
            best = neg
            for u in range(2):
                g = 0.5 * (1.0 - 2.0 * u) * ls + 0.5 * (1.0 - 2.0 * parity[s, u]) * lp
                val = beta[i + 1, next_state[s, u]] + g
                if val > best:
                    best = val
            beta[i, s] = best
            if best > bmax:
                bmax = best
        for s in range(n_states):
            beta[i, s] -= bmax

    extrinsic = np.empty(k)
    for i in range(k):
        ls = la[i] + lc_sys[i]
        lp = lc_par[i]
        m0 = neg
        m1 = neg
        for s in range(n_states):
            for u in range(2):
                g = 0.5 * (1.0 - 2.0 * u) * ls + 0.5 * (1.0 - 2.0 * parity[s, u]) * lp
                metric = alpha[i, s] + g + beta[i + 1, next_state[s, u]]
                if u == 0:
                    if metric > m0:
                        m0 = metric
                elif metric > m1:
                    m1 = metric
        extrinsic[i] = (m0 - m1) - la[i] - lc_sys[i]
    return extrinsic


@njit(cache=True)
def _turbo_decode_one(llrs, perm, inv_perm, next_state, parity,  # pragma: no cover
                      n_states, iterations):
    """Iterative max-log-MAP turbo decode of one frame; returns the k data bits."""
    k = len(perm)
    lc_sys = llrs[:k]
    lc_par1 = llrs[k:2 * k]
    lc_par2 = llrs[2 * k:3 * k]
    lc_sys_il = lc_sys[perm]
    la1 = np.zeros(k)
    le1 = np.zeros(k)
    for _ in range(iterations):
        le1 = _siso(lc_sys, lc_par1, la1, next_state, parity, n_states)
        le2 = _siso(lc_sys_il, lc_par2, le1[perm], next_state, parity, n_states)
        la1 = le2[inv_perm]
    out = np.empty(k, dtype=np.int64)
    for i in range(k):
        out[i] = 1 if (lc_sys[i] + le1[i] + la1[i]) < 0.0 else 0
    return out


@njit(parallel=True, cache=True)
def _turbo_decode_batch(llrs_2d, perm, inv_perm, next_state,  # pragma: no cover
                        parity, n_states, iterations):
    """Decode a batch of equal-length frames, one per parallel thread."""
    n_frames = llrs_2d.shape[0]
    out = np.empty((n_frames, len(perm)), dtype=np.int64)
    for f in prange(n_frames):
        out[f] = _turbo_decode_one(llrs_2d[f], perm, inv_perm, next_state,
                                   parity, n_states, iterations)
    return out


class TurboCode:
    """Rate-1/3 parallel-concatenated turbo code (two LTE-style RSC encoders).

    Two recursive systematic convolutional encoders — the second fed a random
    interleave of the data — produce the systematic stream plus two parity
    streams.  Decoding iterates two max-log-MAP (BCJR) decoders that exchange
    extrinsic information.  The constituent trellises are not terminated; the
    backward recursion starts from a uniform end state.
    """

    def __init__(self, k: int, interleaver_seed: int = 0xC0DE) -> None:
        self.k = k
        self.n_states = 8
        self.rate = 1.0 / 3.0
        self._next, self._parity = _rsc_trellis()
        perm = np.random.default_rng(interleaver_seed).permutation(k)
        self._perm = perm.astype(np.int64)
        self._inv_perm = np.argsort(perm).astype(np.int64)

    def _rsc_parity(self, bits: np.ndarray) -> np.ndarray:
        """Run the RSC encoder over bits, returning the parity stream."""
        state = 0
        par = np.empty(len(bits), dtype=np.int64)
        for i in range(len(bits)):
            u = int(bits[i])
            par[i] = self._parity[state, u]
            state = int(self._next[state, u])
        return par

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode k data bits to 3k coded bits: systematic, parity 1, parity 2."""
        data = np.asarray(data_bits, dtype=np.int64)
        p1 = self._rsc_parity(data)
        p2 = self._rsc_parity(data[self._perm])
        return np.concatenate([data, p1, p2])

    def decode(self, llrs: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Iterative max-log-MAP turbo decode; returns the k data bits.

        llrs holds 3k channel LLRs ordered systematic, parity 1, parity 2.
        """
        return _turbo_decode_one(np.asarray(llrs, dtype=np.float64),
                                 self._perm, self._inv_perm, self._next,
                                 self._parity, self.n_states, iterations)

    def decode_batch(self, llrs_batch: np.ndarray, iterations: int = 8) -> np.ndarray:
        """Decode many equal-length frames in parallel across CPU cores.

        llrs_batch is a 2-D array (n_frames x 3k); returns an (n_frames x k)
        array of decoded data bits.
        """
        arr = np.ascontiguousarray(llrs_batch, dtype=np.float64)
        return _turbo_decode_batch(arr, self._perm, self._inv_perm, self._next,
                                   self._parity, self.n_states, iterations)

    def decode_data(self, llrs_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of frames to an (n_frames x k) array of data bits."""
        return self.decode_batch(llrs_batch)
