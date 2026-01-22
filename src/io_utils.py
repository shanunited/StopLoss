from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def safe_folder_name(name: str, ticker: str) -> str:
    """Generate a filesystem-safe folder name from name and ticker."""
    raw = f"{name}_{ticker}"
    cleaned = []
    for ch in raw.upper():
        cleaned.append(ch if ch.isalnum() else "_")
    joined = "".join(cleaned)
    collapsed = "_".join(part for part in joined.split("_") if part)
    return collapsed or "INSTRUMENT"


def save_csv(df: pd.DataFrame, path: Path, index_label: str | None = "Date") -> None:
    """Save a dataframe to CSV, ensuring parent directory exists."""
    ensure_dir(path.parent)
    df.to_csv(path, index_label=index_label)


def write_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Write metadata JSON to disk."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
