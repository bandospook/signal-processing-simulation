"""Benchmark FEC decoder throughput — serial vs parallel batch decoding.

Run from the repo root:  .venv\\Scripts\\python.exe tools/benchmark_coding.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from sim.coding import ConcatenatedCode, ConvolutionalCode, LDPCCode, TurboCode  # noqa: E402
from sim.receiver import soft_demap  # noqa: E402

_ALIST = Path(__file__).resolve().parent.parent / "data" / "ldpc" / "mackay_13298.alist"
_BATCH = 64


def _llrs(coded_bits, ebn0_db, rate, rng):
    """BPSK + AWGN soft LLRs for a coded-bit array at the given information Eb/N0."""
    sigma = np.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * rate))
    tx = 1.0 - 2.0 * np.asarray(coded_bits, dtype=float)
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    return soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)


def _row(label, decode, decode_batch, batch):
    """Time serial (per-frame) and parallel-batch decoding; print the comparison."""
    frames = list(batch)
    decode(frames[0])                            # warm up serial JIT
    start = time.perf_counter()
    for f in frames:
        decode(f)
    serial = (time.perf_counter() - start) / len(frames)

    decode_batch(batch[:4])                      # warm up batch JIT
    start = time.perf_counter()
    decode_batch(batch)
    batched = (time.perf_counter() - start) / len(batch)

    print(f"{label:<16}{serial * 1e3:>14.2f}{batched * 1e3:>14.2f}{serial / batched:>9.1f}x")


def main():
    rng = np.random.default_rng(0)
    print(f"{'decoder':<16}{'serial ms/fr':>14}{'batch ms/fr':>14}{'speedup':>10}")
    print("-" * 54)

    conv = ConvolutionalCode()
    batch = np.stack([_llrs(conv.encode(rng.integers(0, 2, 4000)), 3.0, conv.rate, rng)
                      for _ in range(_BATCH)])
    _row("convolutional", conv.decode, conv.decode_batch, batch)

    turbo = TurboCode(2000)
    batch = np.stack([_llrs(turbo.encode(rng.integers(0, 2, turbo.k)), 2.0, turbo.rate, rng)
                      for _ in range(_BATCH)])
    _row("turbo", turbo.decode, turbo.decode_batch, batch)

    concat = ConcatenatedCode()
    batch = np.stack([_llrs(concat.encode(rng.integers(0, 2, concat.k_data_bits)),
                            3.5, concat.rate, rng) for _ in range(_BATCH)])
    _row("concatenated", concat.decode, concat.decode_batch, batch)

    ldpc = LDPCCode(_ALIST)
    zero = np.zeros(ldpc.n, dtype=int)
    for ebn0 in (1.0, 2.0, 4.0):
        batch = np.stack([_llrs(zero, ebn0, ldpc.design_rate, rng) for _ in range(_BATCH)])
        _row(f"ldpc @ {ebn0:.0f} dB", ldpc.decode, ldpc.decode_batch, batch)


if __name__ == "__main__":
    main()
