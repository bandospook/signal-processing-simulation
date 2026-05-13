import tomllib
from pathlib import Path


def load_config(path: str | Path = "simulation.toml") -> dict:
    with open(Path(path), "rb") as f:
        return tomllib.load(f)
