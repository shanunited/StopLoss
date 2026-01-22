from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

from .config import get_project_root, load_config
from .indicators import compute_indicators
from .io_utils import ensure_dir, safe_folder_name, save_csv, write_metadata
from .plotting import plot_instrument

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure a logger that writes to a file and stdout."""
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def parse_date(value: Any, default: dt.date, logger: logging.Logger, label: str) -> dt.date:
    """Parse a date from config; fall back to default if invalid."""
    if value in (None, ""):
        return default
    try:
        return pd.to_datetime(value).date()
    except Exception:
        logger.warning("Invalid %s date '%s'; using default %s", label, value, default)
        return default


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data using yfinance."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def normalize_ohlcv_columns(
    df: pd.DataFrame, ticker: str, logger: logging.Logger
) -> pd.DataFrame:
    """Normalize yfinance MultiIndex columns to single-level OHLCV columns."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    cols = df.columns
    if cols.nlevels >= 2:
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)

        if ticker in set(lvl1) and set(REQUIRED_COLUMNS).issubset(set(lvl0)):
            return df.xs(ticker, level=1, axis=1).copy()

        if ticker in set(lvl0) and set(REQUIRED_COLUMNS).issubset(set(lvl1)):
            return df.xs(ticker, level=0, axis=1).copy()

        if len(set(lvl1)) == 1:
            return df.droplevel(1, axis=1)

        if len(set(lvl0)) == 1:
            return df.droplevel(0, axis=1)

    logger.warning("Unexpected MultiIndex columns for %s; columns=%s", ticker, cols)
    return df


def process_instrument(
    name: str,
    ticker: str,
    output_dir: Path,
    start: dt.date,
    end: dt.date,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Download, compute indicators, save outputs, and plot for a single instrument."""
    folder = safe_folder_name(name, ticker)
    instrument_dir = output_dir / folder
    ensure_dir(instrument_dir)

    status: Dict[str, Any] = {
        "name": name,
        "ticker": ticker,
        "folder": folder,
        "status": "fail",
        "rows_downloaded": 0,
        "rows_saved": 0,
        "start": "",
        "end": "",
        "error_message": "",
    }

    start_str = start.isoformat()
    end_str = end.isoformat()

    df_raw = download_ohlcv(ticker, start_str, end_str)
    df_raw = normalize_ohlcv_columns(df_raw, ticker, logger)
    if df_raw is None or df_raw.empty:
        raise ValueError("No data returned from yfinance")

    status["rows_downloaded"] = int(len(df_raw))

    missing = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before_drop = len(df_raw)
    df = df_raw.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    missing_rows_dropped = before_drop - len(df)
    if df.empty:
        raise ValueError("All rows dropped after removing missing OHLC")

    if len(df) < 150:
        logger.warning(
            "%s (%s): only %s rows available; SMA150 will be mostly NaN",
            name,
            ticker,
            len(df),
        )
    if len(df) < 15:
        logger.warning(
            "%s (%s): only %s rows available; Donchian(15) will be mostly NaN",
            name,
            ticker,
            len(df),
        )
    if len(df) < 14:
        logger.warning(
            "%s (%s): only %s rows available; ATR14 will be mostly NaN",
            name,
            ticker,
            len(df),
        )

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].round(2)

    indicators = compute_indicators(df)
    indicators_rounded = indicators.round(2)

    ohlc = df[["Open", "High", "Low", "Close", "Volume"]]
    merged = pd.concat([ohlc, indicators_rounded], axis=1)

    save_csv(ohlc, instrument_dir / "ohlc_2dp.csv")
    save_csv(indicators_rounded, instrument_dir / "indicators.csv")
    save_csv(merged, instrument_dir / "merged_with_indicators.csv")

    metadata = {
        "ticker": ticker,
        "name": name,
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "rows": int(len(df)),
        "missing_rows_dropped": int(missing_rows_dropped),
        "yfinance_params": {
            "start": start_str,
            "end": end_str,
            "interval": "1d",
            "auto_adjust": True,
            "progress": False,
        },
    }
    write_metadata(metadata, instrument_dir / "metadata.json")

    plot_instrument(merged, instrument_dir / "chart_last10y.png", f"{name} ({ticker})")

    status.update(
        {
            "status": "success",
            "rows_saved": int(len(df)),
            "start": metadata["start_date"],
            "end": metadata["end_date"],
            "error_message": "",
        }
    )

    if missing_rows_dropped:
        logger.info(
            "%s (%s): dropped %s rows with missing OHLC",
            name,
            ticker,
            missing_rows_dropped,
        )

    return status


def main() -> None:
    root = get_project_root()
    config, warnings = load_config()

    output_dir = root / str(config.get("output_dir", "output"))
    summary_dir = output_dir / "_summary"
    ensure_dir(summary_dir)

    logger = setup_logger(summary_dir / "run_log.txt")
    for warning in warnings:
        logger.warning(warning)

    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 10)

    start = parse_date(config.get("start"), default_start, logger, "start")
    end = parse_date(config.get("end"), today, logger, "end")

    instruments: List[Dict[str, str]] = config.get("instruments", [])
    if not instruments:
        logger.error("No instruments configured. Exiting.")
        return

    status_rows: List[Dict[str, Any]] = []

    for inst in instruments:
        name = inst.get("name", "")
        ticker = inst.get("ticker", "")
        if not name or not ticker:
            logger.error("Invalid instrument entry: %s", inst)
            status_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "folder": "",
                    "status": "fail",
                    "rows_downloaded": 0,
                    "rows_saved": 0,
                    "start": "",
                    "end": "",
                    "error_message": "Invalid instrument entry",
                }
            )
            continue

        try:
            status = process_instrument(name, ticker, output_dir, start, end, logger)
            status_rows.append(status)
        except Exception as exc:
            logger.exception("Failed processing %s (%s)", name, ticker)
            status_rows.append(
                {
                    "name": name,
                    "ticker": ticker,
                    "folder": safe_folder_name(name, ticker),
                    "status": "fail",
                    "rows_downloaded": 0,
                    "rows_saved": 0,
                    "start": "",
                    "end": "",
                    "error_message": str(exc),
                }
            )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(summary_dir / "status_report.csv", index=False)


if __name__ == "__main__":
    main()
