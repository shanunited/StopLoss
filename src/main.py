from __future__ import annotations

from pathlib import Path

from .config import get_project_root, load_config
from .stoploss.run_pipeline import run_pipeline


def main() -> None:
    root = get_project_root()
    config, warnings = load_config()
    for warning in warnings:
        print(f"WARNING: {warning}")

    output_dir = Path(config.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    run_pipeline(
        instruments=None,
        start_date=config.get("start"),
        end_date=config.get("end"),
        output_dir=str(output_dir),
        config_path=None,
        enable_charts=True,
    )


if __name__ == "__main__":
    main()
