"""Benchmark FEC decoder throughput.

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
_TRIALS = 30


def _llrs(coded_bits, ebn0_db, rate, rng):
    """BPSK + AWGN soft LLRs for a coded-bit array at the given information Eb/N0."""
    sigma = np.sqrt(1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0) * rate))
    tx = 1.0 - 2.0 * np.asarray(coded_bits, dtype=float)
    rx = tx + sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    return soft_demap(rx, "BPSK", noise_var=2.0 * sigma ** 2)


def _mean_seconds(decode, inputs):
    """Mean wall-clock seconds per decode call, excluding one JIT warmup call."""
    decode(inputs[0])
    start = time.perf_counter()
    for x in inputs:
        decode(x)
    return (time.perf_counter() - start) / len(inputs)


def _report(label, info_bits, seconds):
    print(f"{label:<18}{info_bits:>12}{seconds * 1e3:>14.2f}"
          f"{info_bits / seconds / 1e6:>16.3f}")


def main():
    rng = np.random.default_rng(0)
    print(f"{'decoder':<18}{'info bits':>12}{'ms / frame':>14}{'info Mbit/s':>16}")
    print("-" * 60)

    conv = ConvolutionalCode()
    frames = [_llrs(conv.encode(rng.integers(0, 2, 4000)), 3.0, conv.rate, rng)
              for _ in range(_TRIALS)]
    _report("convolutional", 4000, _mean_seconds(conv.decode, frames))

    turbo = TurboCode(2000)
    frames = [_llrs(turbo.encode(rng.integers(0, 2, turbo.k)), 2.0, turbo.rate, rng)
              for _ in range(_TRIALS)]
    _report("turbo", turbo.k, _mean_seconds(turbo.decode, frames))

    concat = ConcatenatedCode()
    frames = [_llrs(concat.encode(rng.integers(0, 2, concat.k_data_bits)),
                    3.5, concat.rate, rng) for _ in range(_TRIALS)]
    _report("concatenated", concat.k_data_bits, _mean_seconds(concat.decode, frames))

    ldpc = LDPCCode(_ALIST)
    info = int(round(ldpc.design_rate * ldpc.n))
    zero = np.zeros(ldpc.n, dtype=int)
    for ebn0 in (1.0, 2.0, 4.0):
        frames = [_llrs(zero, ebn0, ldpc.design_rate, rng) for _ in range(_TRIALS)]
        _report(f"ldpc @ {ebn0:.0f} dB", info, _mean_seconds(ldpc.decode, frames))

    start = time.perf_counter()
    ldpc.build_generator()
    print(f"\nLDPC generator build (one-time): {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
