import tomllib
from pathlib import Path


def load_config(path: str | Path = "simulation.toml") -> dict:
    with open(Path(path), "rb") as f:
        cfg = tomllib.load(f)
    # TOML stores human-readable MHz; convert to Hz for the simulation.
    cfg["sweep"]["sample_rate"] = cfg["sweep"]["sample_rate"] * 1_000_000
    for carr in cfg.get("carrier", []):
        carr["freq"]        = carr["freq"]        * 1_000_000
        carr["symbol_rate"] = carr["symbol_rate"] * 1_000_000
    return cfg
