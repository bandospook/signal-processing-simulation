"""Tail-terminated convolutional code with a soft-decision Viterbi decoder."""
import numpy as np
from numba import njit


# _viterbi_core runs as Numba-compiled native code, which coverage.py's line
# tracer cannot observe; it is exercised by the tests in tests/test_coding.py.
@njit
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
    pred = np.zeros((n_steps, n_states), dtype=np.int64)
    inbit = np.zeros((n_steps, n_states), dtype=np.int64)

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

    decoded = np.zeros(n_steps, dtype=np.int64)
    s = 0
    for step in range(n_steps - 1, -1, -1):
        decoded[step] = inbit[step, s]
        s = pred[step, s]
    return decoded


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
