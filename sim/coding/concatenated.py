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
        """Build RS(255,223) + convolutional with a seeded random interleaver."""
        self.rs = galois.ReedSolomon(255, 223)
        self.conv = ConvolutionalCode()
        self.k = self.rs.k * 8
        self._rs_coded_bits = self.rs.n * 8
        self.coded_bits = (self._rs_coded_bits + self.conv.K - 1) * self.conv.n
        self.rate = self.k / self.coded_bits
        perm = np.random.default_rng(interleaver_seed).permutation(self._rs_coded_bits)
        self._perm = perm.astype(np.int64)

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """Encode k data bits to coded_bits coded bits."""
        data = np.asarray(data_bits, dtype=np.uint8)
        rs_codeword = np.asarray(self.rs.encode(np.packbits(data)), dtype=np.uint8)
        rs_bits = np.unpackbits(rs_codeword)
        return self.conv.encode(rs_bits[self._perm])

    def _rs_outer(self, inner: np.ndarray) -> np.ndarray:
        """Deinterleave inner-decoder bits and run the algebraic RS outer decode."""
        rs_bits = np.empty(self._rs_coded_bits, dtype=np.uint8)
        rs_bits[self._perm] = inner.astype(np.uint8)      # deinterleave
        message = self.rs.decode(np.packbits(rs_bits))
        return np.unpackbits(np.asarray(message, dtype=np.uint8)).astype(np.int64)

    def decode(self, llrs: np.ndarray) -> np.ndarray:
        """Decode coded LLRs: soft-decision Viterbi inner, then algebraic RS outer."""
        return self._rs_outer(self.conv.decode(llrs))

    def decode_batch(self, llrs_batch: np.ndarray) -> np.ndarray:
        """Decode many equal-length frames; the inner Viterbi runs in parallel.

        The convolutional inner code is decoded across CPU cores; the
        algebraic RS outer decode (galois) is then applied per frame.
        """
        inner = self.conv.decode_batch(llrs_batch)
        return np.stack([self._rs_outer(row) for row in inner])

    def decode_data(self, llrs_batch: np.ndarray) -> np.ndarray:
        """Decode a batch of frames to an (n_frames x k) array of data bits."""
        return self.decode_batch(llrs_batch)
