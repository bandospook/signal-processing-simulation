"""Classic concatenated code: Reed-Solomon outer + convolutional inner."""
import galois
import numpy as np

from .convolutional import ConvolutionalCode


class ConcatenatedCode:
    """Reed-Solomon (255, 223) outer code wrapping a convolutional inner code.

    The RS code over GF(2^8) corrects byte errors the inner code leaves behind;
    a random bit interleaver between the two spreads Viterbi burst errors across
    RS symbols so they stay within the t=16 correction power.  The inner code is
    the rate-1/2 (171, 133) convolutional code with soft-decision Viterbi
    decoding.  Overall rate is about 0.436.
    """

    def __init__(self, interleaver_seed: int = 0xCAFE) -> None:
        self.rs = galois.ReedSolomon(255, 223)
        self.conv = ConvolutionalCode()
        self.k_data_bits = self.rs.k * 8
        self._rs_coded_bits = self.rs.n * 8
        self.coded_bits = (self._rs_coded_bits + self.conv.K - 1) * self.conv.n
        self.rate = self.k_data_bits / self.coded_bits
        perm = np.random.default_rng(interleaver_seed).permutation(self._rs_coded_bits)
        self._perm = perm.astype(np.int64)

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode k_data_bits data bits to coded_bits coded bits."""
        data = np.asarray(data_bits, dtype=np.uint8)
        rs_codeword = np.asarray(self.rs.encode(np.packbits(data)), dtype=np.uint8)
        rs_bits = np.unpackbits(rs_codeword)
        return self.conv.encode(rs_bits[self._perm])

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """Decode coded LLRs: soft-decision Viterbi inner, then algebraic RS outer."""
        inner = self.conv.decode(llrs).astype(np.uint8)
        rs_bits = np.empty(self._rs_coded_bits, dtype=np.uint8)
        rs_bits[self._perm] = inner                       # deinterleave
        message = self.rs.decode(np.packbits(rs_bits))
        return np.unpackbits(np.asarray(message, dtype=np.uint8)).astype(np.int64)
