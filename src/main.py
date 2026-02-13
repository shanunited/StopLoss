from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_project_root, load_config
from .stoploss.run_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StopLoss pipeline.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD).")
    parser.add_argument(
        "--full-backfill-from",
        default=None,
        help="Rebuild raw data from this date (YYYY-MM-DD) through end date.",
    )
    args = parser.parse_args()

    root = get_project_root()
    config, warnings = load_config()
    for warning in warnings:
        print(f"WARNING: {warning}")

    output_dir = Path(args.output_dir or config.get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    start_date = args.start if args.start is not None else config.get("start")
    end_date = args.end if args.end is not None else config.get("end")
    full_backfill_from = (
        args.full_backfill_from
        if args.full_backfill_from is not None
        else config.get("full_backfill_from")
    )

    run_pipeline(
        instruments=None,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(output_dir),
        config_path=None,
        enable_charts=True,
        full_backfill_from=full_backfill_from,
    )


if __name__ == "__main__":
    main()
