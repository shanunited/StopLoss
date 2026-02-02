"""StopLoss pipeline entry point for Airflow and CLI usage."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf

from ..config import default_config, load_config
from ..indicators import compute_indicators
from ..io_utils import ensure_dir, write_metadata
from ..plotting import plot_instrument

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"stoploss.pipeline.{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _parse_date(value: Any, default: dt.date, logger: logging.Logger, label: str) -> dt.date:
    if value in (None, ""):
        return default
    try:
        return pd.to_datetime(value).date()
    except Exception:
        logger.warning("Invalid %s date '%s'; using default %s", label, value, default)
        return default


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def _normalize_ohlcv_columns(
    df: pd.DataFrame, ticker: str, logger: logging.Logger
) -> pd.DataFrame:
    if df is None:
        return df
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


def _parse_instruments(items: Optional[Iterable[dict]]) -> List[Dict[str, str]]:
    if not items:
        return []
    result: List[Dict[str, str]] = []
    for raw in items:
        if raw is None:
            continue
        if isinstance(raw, dict) and "ticker" in raw:
            name = str(raw.get("name") or raw.get("ticker")).strip()
            ticker = str(raw.get("ticker") or "").strip()
        else:
            text = str(raw).strip()
            if not text:
                continue
            if "|" in text:
                name, ticker = [part.strip() for part in text.split("|", 1)]
            elif ":" in text:
                name, ticker = [part.strip() for part in text.split(":", 1)]
            else:
                name, ticker = text, text
        if name and ticker:
            result.append({"name": name, "ticker": ticker})
    return result


def _read_raw_ndjson(path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in %s", path)
    except Exception as exc:
        logger.warning("Failed reading raw OHLC at %s: %s", path, exc)
        return None
    if not records:
        return None
    df = pd.DataFrame.from_records(records)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()
    return df


def _write_raw_ndjson(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df_out = df.copy()
    df_out = df_out.reset_index()
    df_out["Date"] = df_out["Date"].dt.strftime("%Y-%m-%d")
    with path.open("w", encoding="utf-8") as handle:
        for record in df_out.to_dict(orient="records"):
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _ensure_ohlcv_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _calculate_date_range(
    existing_df: Optional[pd.DataFrame],
    start_date: Optional[dt.date],
    end_date: dt.date,
    logger: logging.Logger,
) -> Optional[tuple[dt.date, dt.date]]:
    if existing_df is not None and not existing_df.empty:
        last_date = existing_df.index.max().date()
        if start_date is None or start_date <= last_date:
            download_start = last_date + dt.timedelta(days=1)
        else:
            download_start = start_date
    else:
        if start_date is None:
            download_start = end_date - dt.timedelta(days=3650)
        else:
            download_start = start_date

    if download_start > end_date:
        logger.info("No new data needed (start %s > end %s)", download_start, end_date)
        return None

    return download_start, end_date


def run_pipeline(
    instruments: Optional[List[Dict[str, str]]],
    start_date: Optional[str],
    end_date: Optional[str],
    output_dir: str,
    config_path: Optional[str],
    enable_charts: bool = True,
) -> Dict[str, Any]:
    """Run the stoploss pipeline and return a status dict.

    Args:
        instruments: Optional list of instruments (dicts with name/ticker).
        start_date: Optional ISO date string. If None, uses incremental logic.
        end_date: Optional ISO date string. If None, defaults to today.
        output_dir: Base output directory.
        config_path: Optional path to config.yaml for default instruments.
        enable_charts: Whether to render charts.
    """
    cfg, warnings = load_config(Path(config_path) if config_path else None)

    output_base = Path(output_dir or cfg.get("output_dir") or "output")
    summary_dir = output_base / "_summary"
    ensure_dir(summary_dir)

    log_path = summary_dir / "run_log.txt"
    logger = _setup_logger(log_path)
    for warning in warnings:
        logger.warning(warning)

    today = dt.date.today()
    end_dt = _parse_date(end_date, today, logger, "end")
    default_start = end_dt - dt.timedelta(days=3650)
    start_dt = _parse_date(start_date, default_start, logger, "start") if start_date else None

    instrument_defs = _parse_instruments(instruments)
    if not instrument_defs:
        instrument_defs = cfg.get("instruments", default_config()["instruments"])

    run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID") or os.getenv("AIRFLOW_CTX_RUN_ID")
    logical_date = (
        os.getenv("AIRFLOW_CTX_LOGICAL_DATE")
        or os.getenv("AIRFLOW_CTX_EXECUTION_DATE")
        or (end_dt.isoformat() if end_dt else "")
    )

    status_by_ticker: Dict[str, Dict[str, Any]] = {}

    for inst in instrument_defs:
        name = inst.get("name", "")
        ticker = inst.get("ticker", "")
        if not name or not ticker:
            logger.error("Invalid instrument entry: %s", inst)
            status_by_ticker[ticker or name or "unknown"] = {
                "status": "failed",
                "rows_added": 0,
                "last_date": "",
                "error": "Invalid instrument entry",
            }
            continue

        instrument_dir = output_base / ticker
        ensure_dir(instrument_dir)
        raw_path = instrument_dir / "raw_ohlc.json"
        final_path = instrument_dir / "final.csv"

        existing_df = _read_raw_ndjson(raw_path, logger)
        existing_rows = int(len(existing_df)) if existing_df is not None else 0

        status = {
            "status": "failed",
            "rows_added": 0,
            "last_date": "",
            "error": "",
        }

        try:
            date_range = _calculate_date_range(existing_df, start_dt, end_dt, logger)
            if date_range is None:
                last_date = existing_df.index.max().date() if existing_df is not None else None
                status.update(
                    {
                        "status": "no_new_data",
                        "rows_added": 0,
                        "last_date": str(last_date) if last_date else "",
                        "error": "",
                    }
                )
                logger.info("%s (%s): no new data to download", name, ticker)

                metadata = {
                    "ticker": ticker,
                    "name": name,
                    "last_updated": dt.datetime.utcnow().isoformat(),
                    "last_date_in_raw": str(last_date) if last_date else "",
                    "rows_total": existing_rows,
                    "rows_added_in_run": 0,
                    "run_id": run_id,
                    "logical_date": logical_date,
                    "buy_calls_in_run": 0,
                    "last_buy_call_date": "",
                    "strategy_version": "buy_ma30_donch20_v1",
                    "status": "no_new_data",
                    "error_message": "",
                }
                write_metadata(metadata, instrument_dir / "metadata.json")
                status_by_ticker[ticker] = status
                continue

            download_start, download_end = date_range
            yfinance_end = (download_end + dt.timedelta(days=1)).isoformat()

            logger.info(
                "%s (%s): downloading %s to %s (inclusive)",
                name,
                ticker,
                download_start,
                download_end,
            )

            df_raw = _download_ohlcv(ticker, download_start.isoformat(), yfinance_end)
            df_raw = _normalize_ohlcv_columns(df_raw, ticker, logger)

            if df_raw is None or df_raw.empty:
                last_date = existing_df.index.max().date() if existing_df is not None else None
                status.update(
                    {
                        "status": "no_new_data",
                        "rows_added": 0,
                        "last_date": str(last_date) if last_date else "",
                        "error": "",
                    }
                )
                logger.info(
                    "%s (%s): no new data returned for %s to %s",
                    name,
                    ticker,
                    download_start,
                    download_end,
                )
                metadata = {
                    "ticker": ticker,
                    "name": name,
                    "last_updated": dt.datetime.utcnow().isoformat(),
                    "last_date_in_raw": str(last_date) if last_date else "",
                    "rows_total": existing_rows,
                    "rows_added_in_run": 0,
                    "run_id": run_id,
                    "logical_date": logical_date,
                    "buy_calls_in_run": 0,
                    "last_buy_call_date": "",
                    "strategy_version": "buy_ma30_donch20_v1",
                    "status": "no_new_data",
                    "error_message": "",
                }
                write_metadata(metadata, instrument_dir / "metadata.json")
                status_by_ticker[ticker] = status
                continue

            _ensure_ohlcv_columns(df_raw)

            before_drop = len(df_raw)
            df_new = df_raw.dropna(subset=["Open", "High", "Low", "Close"]).copy()
            if df_new.empty:
                raise ValueError("All rows dropped after removing missing OHLC")

            for col in ["Open", "High", "Low", "Close"]:
                df_new[col] = df_new[col].round(2)

            df_new.index = pd.to_datetime(df_new.index)

            if existing_df is not None:
                combined = pd.concat([existing_df, df_new], axis=0)
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = df_new

            combined = combined[REQUIRED_COLUMNS].copy()

            rows_added = max(0, len(combined) - existing_rows)

            _write_raw_ndjson(combined, raw_path)

            final_df = compute_indicators(combined)

            final_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "SMA150",
                "DonchianLow15",
                "ATR14",
                "StopLoss",
                "MA30",
                "DonchianHigh20",
                "DonchianHigh20_prev",
                "BuyCall",
                "EntryPrice",
                "StopLoss_Entry",
            ]
            final_df = final_df[final_columns].copy()
            final_df["SMA150"] = final_df["SMA150"].round(2)
            final_df["DonchianLow15"] = final_df["DonchianLow15"].round(2)
            final_df["ATR14"] = final_df["ATR14"].round(2)
            final_df["StopLoss"] = final_df["StopLoss"].round(2)
            final_df["MA30"] = final_df["MA30"].round(2)
            final_df["DonchianHigh20"] = final_df["DonchianHigh20"].round(2)
            final_df["DonchianHigh20_prev"] = final_df["DonchianHigh20_prev"].round(2)
            final_df["EntryPrice"] = final_df["EntryPrice"].round(2)
            final_df["StopLoss_Entry"] = final_df["StopLoss_Entry"].round(2)
            final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
            final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
            final_df.to_csv(final_path, index=False)

            if enable_charts:
                chart_df = compute_indicators(combined)
                plot_instrument(
                    chart_df,
                    instrument_dir / "chart_last10y.png",
                    f"{name} ({ticker})",
                )

            last_date = combined.index.max().date() if not combined.empty else None
            status.update(
                {
                    "status": "success",
                    "rows_added": rows_added,
                    "last_date": str(last_date) if last_date else "",
                    "error": "",
                }
            )

            buy_calls = int(final_df["BuyCall"].sum())
            last_buy_date = ""
            if buy_calls:
                last_buy_date = (
                    final_df.loc[final_df["BuyCall"] == 1, "Date"].iloc[-1]
                )

            metadata = {
                "ticker": ticker,
                "name": name,
                "last_updated": dt.datetime.utcnow().isoformat(),
                "last_date_in_raw": str(last_date) if last_date else "",
                "rows_total": int(len(combined)),
                "rows_added_in_run": rows_added,
                "run_id": run_id,
                "logical_date": logical_date,
                "buy_calls_in_run": buy_calls,
                "last_buy_call_date": last_buy_date,
                "strategy_version": "buy_ma30_donch20_v1",
                "status": "success",
                "error_message": "",
            }
            write_metadata(metadata, instrument_dir / "metadata.json")

            if before_drop != len(df_new):
                logger.info(
                    "%s (%s): dropped %s rows with missing OHLC",
                    name,
                    ticker,
                    before_drop - len(df_new),
                )

        except Exception as exc:
            logger.exception("Failed processing %s (%s)", name, ticker)
            status.update(
                {
                    "status": "failed",
                    "rows_added": 0,
                    "error": str(exc),
                }
            )
            metadata = {
                "ticker": ticker,
                "name": name,
                "last_updated": dt.datetime.utcnow().isoformat(),
                "last_date_in_raw": "",
                "rows_total": existing_rows,
                "rows_added_in_run": 0,
                "run_id": run_id,
                "logical_date": logical_date,
                "buy_calls_in_run": 0,
                "last_buy_call_date": "",
                "strategy_version": "buy_ma30_donch20_v1",
                "status": "failed",
                "error_message": str(exc),
            }
            write_metadata(metadata, instrument_dir / "metadata.json")

        status_by_ticker[ticker] = status

    return status_by_ticker
