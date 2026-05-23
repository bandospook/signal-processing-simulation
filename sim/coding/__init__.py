"""Forward error correction: convolutional, turbo, and LDPC codes.

See docs/coding_design.md for the design rationale and staging plan.
"""
import numpy as np

from .concatenated import ConcatenatedCode
from .convolutional import ConvolutionalCode
from .ldpc import LDPCCode
from .turbo import TurboCode

__all__ = ["ConcatenatedCode", "ConvolutionalCode", "LDPCCode", "TurboCode",
           "build_code", "encode_frames", "decode_frames"]


def build_code(coding_cfg: dict) -> ConvolutionalCode | ConcatenatedCode | LDPCCode | TurboCode:
    """Construct an FEC code object from a [carrier.coding] config block.

    coding_cfg["scheme"] selects the family; scheme-specific keys:
      turbo -> block_length (data bits per frame); ldpc -> matrix (alist path).
    convolutional and concatenated take no extra keys.
    """
    scheme = str(coding_cfg["scheme"]).lower()
    if scheme == "convolutional":
        return ConvolutionalCode(block_length=int(coding_cfg.get("block_length", 1024)))
    if scheme == "concatenated":
        return ConcatenatedCode()
    if scheme == "turbo":
        return TurboCode(int(coding_cfg["block_length"]))
    if scheme == "ldpc":
        return LDPCCode(coding_cfg["matrix"])
    raise ValueError(f"Unknown coding scheme '{scheme}'")


def encode_frames(code, n_frames: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_frames random data frames and FEC-encode them.

    Returns (data_bits, coded_bits) as flat arrays, frames concatenated.
    """
    if isinstance(code, LDPCCode):
        code.build_generator()              # sets .k and enables encode()
    k = code.k
    data = rng.integers(0, 2, n_frames * k)
    coded = np.concatenate([code.encode(data[f * k:(f + 1) * k])
                            for f in range(n_frames)])
    return data, coded


def decode_frames(code, llrs: np.ndarray, n_frames: int) -> np.ndarray:
    """Decode flat channel LLRs (n_frames concatenated frames) to flat data bits.

    The received LLR count may be slightly shorter than n_frames * code.coded_bits
    because the simulation strips transient samples from both ends of the received
    signal.  When code.coded_bits is available, pad with zeros (neutral LLRs) so
    each frame is always exactly coded_bits long.
    """
    llrs = np.asarray(llrs, dtype=float)
    coded_per_frame: int = getattr(code, "coded_bits", len(llrs) // n_frames)
    total = coded_per_frame * n_frames
    if len(llrs) < total:
        llrs = np.pad(llrs, (0, total - len(llrs)))
    frames = llrs[:total].reshape(n_frames, coded_per_frame)
    return code.decode_data(frames).ravel()
