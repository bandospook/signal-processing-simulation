"""Tail-terminated convolutional code with a soft-decision Viterbi decoder."""
import math

import numpy as np
from numba import njit, prange


# _viterbi_core runs as Numba-compiled native code, which coverage.py's line
# tracer cannot observe; it is exercised by the tests in tests/test_coding.py.
@njit(cache=True)
def _viterbi_core(llrs, next_state, out_bits, n_states, n_gen, n_steps):  # pragma: no cover
    """Maximum-likelihood Viterbi sweep over the trellis, then traceback.

    Branch metric for an expected coded bit c against received LLR L is
    (1 - 2c)*L — positive when the branch agrees with the soft decision — and
    the path metric is their running sum, maximised.  The encoder is tail-
    terminated, so the surviving path is traced back from state 0.
    """
    neg = -1.0e18
    pm = np.full(n_states, neg)
    pm[0] = 0.0
    # pred/inbit need no zero-init: every entry is written by the add-compare-
    # select before traceback reads it (traceback only visits surviving states).
    pred = np.empty((n_steps, n_states), dtype=np.int64)
    inbit = np.empty((n_steps, n_states), dtype=np.int64)

    for step in range(n_steps):
        new_pm = np.full(n_states, neg)
        base = step * n_gen
        for s in range(n_states):
            if pm[s] <= neg * 0.5:
                continue                       # state unreachable so far
            for u in range(2):
                bm = 0.0
                for j in range(n_gen):
                    bm += (1.0 - 2.0 * out_bits[s, u, j]) * llrs[base + j]
                ns = next_state[s, u]
                cand = pm[s] + bm
                if cand > new_pm[ns]:
                    new_pm[ns] = cand
                    pred[step, ns] = s
                    inbit[step, ns] = u
        pm = new_pm

    decoded = np.empty(n_steps, dtype=np.int64)
    s = 0
    for step in range(n_steps - 1, -1, -1):
        decoded[step] = inbit[step, s]
        s = pred[step, s]
    return decoded


@njit(parallel=True, cache=True)
def _viterbi_batch(llrs_2d, next_state, out_bits, n_states, n_gen, n_steps):  # pragma: no cover
    """Decode a batch of equal-length frames, one per parallel thread."""
    n_frames = llrs_2d.shape[0]
    out = np.empty((n_frames, n_steps), dtype=np.int64)
    for f in prange(n_frames):
        out[f] = _viterbi_core(llrs_2d[f], next_state, out_bits, n_states, n_gen, n_steps)
    return out


class ConvolutionalCode:
    """Tail-terminated convolutional code with a soft-decision Viterbi decoder.

    Defaults to the rate-1/2, constraint-length-7 (171, 133)-octal code — the
    CCSDS / Voyager standard, free distance 10.  The K-1 zero tail bits flush
    the encoder back to state 0 so the decoder can traceback from a known end.
    """

    def __init__(self, generators: tuple[int, ...] = (0o171, 0o133), K: int = 7) -> None:
        self.K = K
        self.generators = tuple(generators)
        self.n = len(self.generators)            # coded bits per input bit
        self.rate = 1.0 / self.n
        self.n_states = 1 << (K - 1)
        self._build_trellis()

    def _build_trellis(self) -> None:
        """Precompute next-state and output-bit tables for the Viterbi decoder."""
        next_state = np.zeros((self.n_states, 2), dtype=np.int64)
        out_bits = np.zeros((self.n_states, 2, self.n), dtype=np.int64)
        for s in range(self.n_states):
            for u in range(2):
                sr = (u << (self.K - 1)) | s     # shift register: new bit + state
                next_state[s, u] = sr >> 1
                for j, g in enumerate(self.generators):
                    out_bits[s, u, j] = (sr & g).bit_count() & 1
        self._next_state = next_state
        self._out_bits = out_bits

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode data bits to coded bits, appending K-1 zero tail bits.

        Each generator output is a mod-2 convolution of the (tail-padded) input
        with the generator taps; the n streams are interleaved per input bit.
        """
        data = np.asarray(data_bits, dtype=np.int64)
        u = np.concatenate([data, np.zeros(self.K - 1, dtype=np.int64)])
        streams = []
        for g in self.generators:
            taps = np.array([(g >> (self.K - 1 - i)) & 1 for i in range(self.K)],
                            dtype=np.int64)
            streams.append(np.convolve(u, taps)[:len(u)] & 1)
        return np.stack(streams, axis=1).ravel()

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """Soft-decision Viterbi decode; returns the data bits (tail removed).

        llrs holds one log-likelihood ratio per coded bit (positive favours 0),
        ordered as the encoder emitted them.
        """
        llrs = np.asarray(llrs, dtype=np.float64)
        n_steps = len(llrs) // self.n
        decoded = _viterbi_core(llrs, self._next_state, self._out_bits,
                                self.n_states, self.n, n_steps)
        return decoded[:n_steps - (self.K - 1)]

    def decode_batch(self, llrs_batch: np.ndarray) -> np.ndarray:
        """Decode many equal-length frames in parallel across CPU cores.

        llrs_batch is a 2-D array (n_frames x coded_bits); returns an
        (n_frames x data_bits) array of decoded bits.
        """
        arr = np.ascontiguousarray(llrs_batch, dtype=np.float64)
        n_steps = arr.shape[1] // self.n
        decoded = _viterbi_batch(arr, self._next_state, self._out_bits,
                                 self.n_states, self.n, n_steps)
        return decoded[:, :n_steps - (self.K - 1)]

    def weight_spectrum(self, d_max: int = 24) -> np.ndarray:
        """Information-weighted output-weight spectrum B_d, for d up to d_max.

        B_d is the total number of nonzero information bits over all error
        events — trellis paths that leave the all-zero state and remerge with
        it — of output Hamming weight d.  It feeds the soft-decision Viterbi
        union bound.  This code is non-catastrophic, so the enumeration ends.
        """
        out_weight = self._out_bits.sum(axis=2)
        next_state = self._next_state
        spectrum = np.zeros(d_max + 1, dtype=np.int64)
        # dp[state] = {output_weight: (path_count, information_weight_sum)}
        dp: dict[int, dict[int, tuple[int, int]]] = {
            int(next_state[0, 1]): {int(out_weight[0, 1]): (1, 1)}}
        for _ in range(3000):
            if not dp:
                break
            nxt: dict[int, dict[int, tuple[int, int]]] = {}
            for s, by_weight in dp.items():
                for d, (count, info) in by_weight.items():
                    for u in (0, 1):
                        d2 = d + int(out_weight[s, u])
                        if d2 > d_max:
                            continue
                        info2 = info + u * count
                        s2 = int(next_state[s, u])
                        if s2 == 0:
                            spectrum[d2] += info2
                        else:
                            cell = nxt.setdefault(s2, {})
                            c0, i0 = cell.get(d2, (0, 0))
                            cell[d2] = (c0 + count, i0 + info2)
            dp = nxt
        return spectrum

    def union_bound_ber(self, ebn0_db: float, d_max: int = 24) -> float:
        """Soft-decision Viterbi BER union bound at the given Eb/N0 (dB).

        Sums B_d * Q(sqrt(2 d R Eb/N0)) over the weight spectrum — an upper
        bound on BER that is tight at moderate-to-high SNR.
        """
        spectrum = self.weight_spectrum(d_max)
        ebn0 = 10.0 ** (ebn0_db / 10.0)
        ber = 0.0
        for d in range(d_max + 1):
            if spectrum[d] > 0:
                arg = math.sqrt(2.0 * d * self.rate * ebn0)
                ber += float(spectrum[d]) * 0.5 * math.erfc(arg / math.sqrt(2.0))
        return ber
