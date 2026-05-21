"""Forward error correction: convolutional, turbo, and LDPC codes.

See docs/coding_design.md for the design rationale and staging plan.
"""
from .convolutional import ConvolutionalCode
from .ldpc import LDPCCode

__all__ = ["ConvolutionalCode", "LDPCCode"]
