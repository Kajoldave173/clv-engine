"""Common I/O helpers and path management for the CLV pipeline."""

from pathlib import Path
from typing import Any

import yaml
import pandas as pd


# Project root is two levels up from this file (src/utils/io.py -> clv-engine/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard directory paths
PATHS = {
    "raw": PROJECT_ROOT / "data" / "raw",
    "interim": PROJECT_ROOT / "data" / "interim",
    "processed": PROJECT_ROOT / "data" / "processed",
    "predictions": PROJECT_ROOT / "data" / "predictions",
    "models": PROJECT_ROOT / "models",
    "reports": PROJECT_ROOT / "reports",
    "figures": PROJECT_ROOT / "reports" / "figures",
    "configs": PROJECT_ROOT / "configs",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Path to params.yaml. Defaults to configs/params.yaml
                     relative to the project root.

    Returns:
        Dictionary of configuration parameters.
    """
    if config_path is None:
        config_path = PATHS["configs"] / "params.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Save a DataFrame to CSV, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        path: Output file path.
        **kwargs: Additional arguments passed to df.to_csv().

    Returns:
        The resolved Path where the file was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    return path.resolve()


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments passed to pd.read_csv().

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, **kwargs)
