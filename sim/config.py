"""TOML config loader with unit conversions and adaptive-sweep defaults."""
import tomllib
from pathlib import Path

_SIM_DEFAULTS = {
    "seed":                    42,
    "max_block_size_samples":  16_777_216,   # ~256 MB per native-rate buffer
    "target_ci_half_width":    2e-3,         # absolute half-width on BER
    "confidence":              0.95,
    "min_errors":              50,
    "max_iterations":          100,
}


def load_config(path: str | Path = "simulation.toml") -> dict:
    with open(Path(path), "rb") as f:
        cfg = tomllib.load(f)
    # TOML stores human-readable MHz; convert to Hz for the simulation.
    cfg["sweep"]["sample_rate"] = cfg["sweep"]["sample_rate"] * 1_000_000
    for carr in cfg.get("carrier", []):
        carr["freq"]        = carr["freq"]        * 1_000_000
        carr["symbol_rate"] = carr["symbol_rate"] * 1_000_000
    # Fill [simulation] defaults for keys the user hasn't set.
    sim = cfg.setdefault("simulation", {})
    for k, v in _SIM_DEFAULTS.items():
        sim.setdefault(k, v)
    return cfg
