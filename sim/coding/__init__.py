"""Forward error correction: convolutional, turbo, and LDPC codes.

See docs/coding_design.md for the design rationale and staging plan.
"""
from .concatenated import ConcatenatedCode
from .convolutional import ConvolutionalCode
from .ldpc import LDPCCode
from .turbo import TurboCode

__all__ = ["ConcatenatedCode", "ConvolutionalCode", "LDPCCode", "TurboCode", "build_code"]


def build_code(coding_cfg: dict) -> ConvolutionalCode | ConcatenatedCode | LDPCCode | TurboCode:
    """Construct an FEC code object from a [carrier.coding] config block.

    coding_cfg["scheme"] selects the family; scheme-specific keys:
      turbo -> block_length (data bits per frame); ldpc -> matrix (alist path).
    convolutional and concatenated take no extra keys.
    """
    scheme = str(coding_cfg["scheme"]).lower()
    if scheme == "convolutional":
        return ConvolutionalCode()
    if scheme == "concatenated":
        return ConcatenatedCode()
    if scheme == "turbo":
        return TurboCode(int(coding_cfg["block_length"]))
    if scheme == "ldpc":
        return LDPCCode(coding_cfg["matrix"])
    raise ValueError(f"Unknown coding scheme '{scheme}'")
