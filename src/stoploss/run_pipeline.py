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
from ..strategy import compute_trade_state

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SPECIAL_START_TICKERS = {
    "NIFTYMIDSML400.NS": dt.date(2015, 1, 1),
    "^CNXSC": dt.date(2015, 1, 1),
}
MANUAL_INPUT_DIRS = [
    Path("data/manual_inputs"),
    Path("data/stoploss_output/Manual Data"),
]
MANUAL_TICKERS = {
    "NIFTYMIDSML400.NS": [
        "NIFTYMIDSML400.NS.csv",
        "Nifty Midcap 150 Historical Data.csv",
    ],
    "^CNXSC": [
        "CNXSC.csv",
        "NIFTY Smallcap 100 Historical Data.csv",
    ],
}
FINAL_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA150",
    "DonchianLow15",
    "ATR14",
    "RSI14",
    "StopLoss",
    "MA30",
    "DonchianHigh20",
    "DonchianHigh20_prev",
    "BuyCall",
    "EntryPrice",
    "StopLoss_Entry",
    "CandidateStop",
    "TrailingStop",
    "InPosition",
    "EntryDate",
    "ExitSignal",
    "ExitPrice",
    "ExitReason",
    "TradeId",
    "HoldingDays",
    "PnL_pct",
]
ROUND_COLUMNS = [
    "SMA150",
    "DonchianLow15",
    "ATR14",
    "RSI14",
    "StopLoss",
    "MA30",
    "DonchianHigh20",
    "DonchianHigh20_prev",
    "EntryPrice",
    "StopLoss_Entry",
    "CandidateStop",
    "TrailingStop",
    "ExitPrice",
    "PnL_pct",
]


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


def _effective_start_date(ticker: str, start_dt: Optional[dt.date]) -> Optional[dt.date]:
    special = SPECIAL_START_TICKERS.get(ticker)
    if not special:
        return start_dt
    if start_dt is None:
        return special
    return special if start_dt > special else start_dt


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def _find_manual_path(ticker: str) -> Optional[Path]:
    filenames = MANUAL_TICKERS.get(ticker)
    if not filenames:
        return None
    for directory in MANUAL_INPUT_DIRS:
        for name in filenames:
            path = directory / name
            if path.exists():
                return path
    return None


def _parse_manual_ohlcv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Support standard schema or Investing.com-style exports
    if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        data = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    elif {"Date", "Price", "Open", "High", "Low", "Vol."}.issubset(df.columns):
        data = df[["Date", "Price", "Open", "High", "Low", "Vol."]].copy()
        data = data.rename(columns={"Price": "Close", "Vol.": "Volume"})
    else:
        raise ValueError(
            f"Manual input schema not recognized at {path}. "
            "Expected columns: Date, Open, High, Low, Close, Volume "
            "or Date, Price, Open, High, Low, Vol."
        )

    # Normalize numeric fields (strip commas)
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"": None, "nan": None})
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Volume"] = (
        data["Volume"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("B", "", regex=False)
        .str.replace("M", "", regex=False)
        .replace({"": None, "nan": None, "None": None})
    )
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0)

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=True)
    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"])
    data = data.set_index("Date")
    data = data[REQUIRED_COLUMNS].copy()

    for col in ["Open", "High", "Low", "Close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").round(2)

    logger.info("Loaded manual input from %s (%s rows)", path, len(data))
    return data


def _load_manual_ohlcv(
    ticker: str, logger: logging.Logger
) -> Optional[tuple[pd.DataFrame, Path]]:
    path = _find_manual_path(ticker)
    if not path:
        return None
    df = _parse_manual_ohlcv(path, logger)
    return df, path


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


def _round_numeric_columns(df: pd.DataFrame, columns: list[str], digits: int = 2) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(digits)
    return df


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
    full_backfill_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the stoploss pipeline and return a status dict.

    Args:
        instruments: Optional list of instruments (dicts with name/ticker).
        start_date: Optional ISO date string. If None, uses incremental logic.
        end_date: Optional ISO date string. If None, defaults to today.
        output_dir: Base output directory.
        config_path: Optional path to config.yaml for default instruments.
        enable_charts: Whether to render charts.
        full_backfill_from: Optional ISO date string. If set, rebuilds raw data
            from this date through end_date and overwrites raw_ohlc.json.
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
    full_backfill_dt = (
        _parse_date(full_backfill_from, default_start, logger, "full_backfill_from")
        if full_backfill_from
        else None
    )
    if full_backfill_dt and full_backfill_dt > end_dt:
        raise ValueError("full_backfill_from cannot be after end_date")

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

        existing_df = None if full_backfill_dt else _read_raw_ndjson(raw_path, logger)
        existing_rows = int(len(existing_df)) if existing_df is not None else 0

        status = {
            "status": "failed",
            "rows_added": 0,
            "last_date": "",
            "error": "",
        }

        try:
            manual = _load_manual_ohlcv(ticker, logger)
            if manual is not None:
                if full_backfill_dt:
                    logger.info(
                        "%s (%s): manual input detected; full_backfill_from ignored",
                        name,
                        ticker,
                    )
                combined, manual_path = manual
                _write_raw_ndjson(combined, raw_path)
                final_df = compute_indicators(combined)
                final_df = compute_trade_state(final_df)
                final_df = final_df[FINAL_COLUMNS].copy()
                final_df = _round_numeric_columns(final_df, ROUND_COLUMNS)
                final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
                final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
                if "EntryDate" in final_df.columns:
                    final_df["EntryDate"] = pd.to_datetime(
                        final_df["EntryDate"], errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
                final_df.to_csv(final_path, index=False)
                if enable_charts:
                    plot_instrument(
                        final_df.set_index("Date"),
                        instrument_dir / "chart_last10y.png",
                        f"{name} ({ticker})",
                    )
                last_date = combined.index.max().date() if not combined.empty else None
                status.update(
                    {
                        "status": "success",
                        "rows_added": int(len(combined)),
                        "last_date": str(last_date) if last_date else "",
                        "error": "",
                    }
                )
                metadata = {
                    "ticker": ticker,
                    "name": name,
                    "last_updated": dt.datetime.utcnow().isoformat(),
                    "last_date_in_raw": str(last_date) if last_date else "",
                    "rows_total": int(len(combined)),
                    "rows_added_in_run": int(len(combined)),
                    "run_id": run_id,
                    "logical_date": logical_date,
                    "buy_calls_in_run": int(final_df["BuyCall"].sum()),
                    "last_buy_call_date": final_df.loc[final_df["BuyCall"] == 1, "Date"].iloc[-1]
                    if int(final_df["BuyCall"].sum()) > 0
                    else "",
                    "last_rsi_value_on_last_buy": final_df.loc[final_df["BuyCall"] == 1, "RSI14"].iloc[-1]
                    if int(final_df["BuyCall"].sum()) > 0
                    else "",
                    "total_buy_calls_in_dataset": int(final_df["BuyCall"].sum()),
                    "total_trades": int(final_df["TradeId"].dropna().nunique())
                    if "TradeId" in final_df.columns
                    else 0,
                    "last_trade_entry_date": final_df.loc[final_df["BuyCall"] == 1, "Date"].iloc[-1]
                    if int(final_df["BuyCall"].sum()) > 0
                    else "",
                    "last_trade_exit_date": final_df.loc[final_df["ExitSignal"] == 1, "Date"].iloc[-1]
                    if "ExitSignal" in final_df.columns and int(final_df["ExitSignal"].sum()) > 0
                    else "",
                    "last_trade_pnl_pct": final_df.loc[final_df["ExitSignal"] == 1, "PnL_pct"].iloc[-1]
                    if "ExitSignal" in final_df.columns and int(final_df["ExitSignal"].sum()) > 0
                    else "",
                    "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                    "status": "success",
                    "error_message": "",
                    "data_source": "manual_csv",
                    "manual_input_path": str(manual_path.resolve()),
                }
                write_metadata(metadata, instrument_dir / "metadata.json")
                status_by_ticker[ticker] = status
                continue

            if full_backfill_dt:
                download_start = full_backfill_dt
                download_end = end_dt
            else:
                ticker_start = _effective_start_date(ticker, start_dt)
                date_range = _calculate_date_range(existing_df, ticker_start, end_dt, logger)
                if date_range is None:
                    last_date = (
                        existing_df.index.max().date() if existing_df is not None else None
                    )
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
                        "last_rsi_value_on_last_buy": "",
                        "total_buy_calls_in_dataset": 0,
                        "total_trades": 0,
                        "last_trade_entry_date": "",
                        "last_trade_exit_date": "",
                        "last_trade_pnl_pct": "",
                        "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                        "status": "no_new_data",
                        "error_message": "",
                    }
                    write_metadata(metadata, instrument_dir / "metadata.json")
                    if existing_df is not None:
                        existing_df = existing_df[REQUIRED_COLUMNS].copy()
                        final_df = compute_indicators(existing_df)
                        final_df = compute_trade_state(final_df)
                        final_df = final_df[FINAL_COLUMNS].copy()
                        final_df = _round_numeric_columns(final_df, ROUND_COLUMNS)
                        final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
                        final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
                        if "EntryDate" in final_df.columns:
                            final_df["EntryDate"] = pd.to_datetime(
                                final_df["EntryDate"], errors="coerce"
                            ).dt.strftime("%Y-%m-%d")
                        final_df.to_csv(final_path, index=False)
                        if enable_charts:
                            plot_instrument(
                                final_df.set_index("Date"),
                                instrument_dir / "chart_last10y.png",
                                f"{name} ({ticker})",
                            )
                    elif not final_path.exists():
                        pd.DataFrame(columns=["Date"] + FINAL_COLUMNS).to_csv(final_path, index=False)
                    status_by_ticker[ticker] = status
                    continue

                download_start, download_end = date_range

            if download_start > end_dt:
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
                    "last_rsi_value_on_last_buy": "",
                    "total_buy_calls_in_dataset": 0,
                    "total_trades": 0,
                    "last_trade_entry_date": "",
                    "last_trade_exit_date": "",
                    "last_trade_pnl_pct": "",
                    "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                    "status": "no_new_data",
                    "error_message": "",
                }
                write_metadata(metadata, instrument_dir / "metadata.json")
                if existing_df is not None:
                    existing_df = existing_df[REQUIRED_COLUMNS].copy()
                    final_df = compute_indicators(existing_df)
                    final_df = compute_trade_state(final_df)
                    final_df = final_df[FINAL_COLUMNS].copy()
                    final_df = _round_numeric_columns(final_df, ROUND_COLUMNS)
                    final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
                    final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
                    if "EntryDate" in final_df.columns:
                        final_df["EntryDate"] = pd.to_datetime(
                            final_df["EntryDate"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                    final_df.to_csv(final_path, index=False)
                    if enable_charts:
                        plot_instrument(
                            final_df.set_index("Date"),
                            instrument_dir / "chart_last10y.png",
                            f"{name} ({ticker})",
                        )
                elif not final_path.exists():
                    pd.DataFrame(columns=["Date"] + FINAL_COLUMNS).to_csv(final_path, index=False)
                status_by_ticker[ticker] = status
                continue

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
                    "last_rsi_value_on_last_buy": "",
                    "total_buy_calls_in_dataset": 0,
                    "total_trades": 0,
                    "last_trade_entry_date": "",
                    "last_trade_exit_date": "",
                    "last_trade_pnl_pct": "",
                    "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                    "status": "no_new_data",
                    "error_message": "",
                }
                write_metadata(metadata, instrument_dir / "metadata.json")
                if existing_df is not None:
                    existing_df = existing_df[REQUIRED_COLUMNS].copy()
                    final_df = compute_indicators(existing_df)
                    final_df = compute_trade_state(final_df)
                    final_df = final_df[FINAL_COLUMNS].copy()
                    final_df = _round_numeric_columns(final_df, ROUND_COLUMNS)
                    final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
                    final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
                    if "EntryDate" in final_df.columns:
                        final_df["EntryDate"] = pd.to_datetime(
                            final_df["EntryDate"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                    final_df.to_csv(final_path, index=False)
                    if enable_charts:
                        plot_instrument(
                            final_df.set_index("Date"),
                            instrument_dir / "chart_last10y.png",
                            f"{name} ({ticker})",
                        )
                elif not final_path.exists():
                    pd.DataFrame(columns=["Date"] + FINAL_COLUMNS).to_csv(final_path, index=False)
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

            if full_backfill_dt:
                combined = df_new
            else:
                if existing_df is not None:
                    combined = pd.concat([existing_df, df_new], axis=0)
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                else:
                    combined = df_new

            combined = combined[REQUIRED_COLUMNS].copy()
            if combined.index.duplicated().any():
                combined = combined[~combined.index.duplicated(keep="last")]
                if combined.index.duplicated().any():
                    raise ValueError("Duplicate dates remain after dedupe")

            rows_added = int(len(combined)) if full_backfill_dt else max(0, len(combined) - existing_rows)

            if full_backfill_dt:
                earliest = combined.index.min().date() if not combined.empty else None
                if earliest:
                    if earliest < full_backfill_dt:
                        logger.warning(
                            "%s (%s): earliest date %s is before requested %s",
                            name,
                            ticker,
                            earliest,
                            full_backfill_dt,
                        )
                    elif earliest > full_backfill_dt:
                        logger.warning(
                            "%s (%s): earliest date %s is after requested %s",
                            name,
                            ticker,
                            earliest,
                            full_backfill_dt,
                        )

            _write_raw_ndjson(combined, raw_path)

            final_df = compute_indicators(combined)
            final_df = compute_trade_state(final_df)

            final_df = final_df[FINAL_COLUMNS].copy()
            final_df = _round_numeric_columns(final_df, ROUND_COLUMNS)
            final_df = final_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
            final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")
            if "EntryDate" in final_df.columns:
                final_df["EntryDate"] = pd.to_datetime(final_df["EntryDate"], errors="coerce").dt.strftime(
                    "%Y-%m-%d"
                )
            final_df.to_csv(final_path, index=False)

            if full_backfill_dt:
                required_cols = ["MA30", "SMA150", "DonchianHigh20_prev", "RSI14"]
                if all(col in final_df.columns for col in required_cols):
                    buy_nan = final_df.loc[final_df["BuyCall"] == 1, required_cols].isna().any(axis=1)
                    if int(buy_nan.sum()) > 0:
                        logger.warning(
                            "%s (%s): %s BuyCall rows have NaN indicators",
                            name,
                            ticker,
                            int(buy_nan.sum()),
                        )

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
            total_buy_calls = buy_calls
            last_buy_date = ""
            last_rsi_value = ""
            if buy_calls:
                last_buy_date = (
                    final_df.loc[final_df["BuyCall"] == 1, "Date"].iloc[-1]
                )
                last_rsi_value = (
                    final_df.loc[final_df["BuyCall"] == 1, "RSI14"].iloc[-1]
                )

            trade_ids = final_df["TradeId"].dropna()
            total_trades = int(trade_ids.nunique()) if not trade_ids.empty else 0
            last_trade_entry_date = ""
            if total_trades:
                last_trade_id = int(trade_ids.max())
                entry_rows = final_df[
                    (final_df["TradeId"] == last_trade_id) & (final_df["BuyCall"] == 1)
                ]
                if not entry_rows.empty:
                    last_trade_entry_date = entry_rows["Date"].iloc[0]

            last_trade_exit_date = ""
            last_trade_pnl_pct = ""
            exit_rows = final_df[final_df["ExitSignal"] == 1]
            if not exit_rows.empty:
                last_trade_exit_date = exit_rows["Date"].iloc[-1]
                last_trade_pnl_pct = exit_rows["PnL_pct"].iloc[-1]

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
                "last_rsi_value_on_last_buy": last_rsi_value,
                "total_buy_calls_in_dataset": total_buy_calls,
                "total_trades": total_trades,
                "last_trade_entry_date": last_trade_entry_date,
                "last_trade_exit_date": last_trade_exit_date,
                "last_trade_pnl_pct": last_trade_pnl_pct,
                "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                "status": "success",
                "error_message": "",
            }
            write_metadata(metadata, instrument_dir / "metadata.json")

            logger.info(
                "%s (%s): total_trades=%s last_trade_pnl_pct=%s",
                name,
                ticker,
                total_trades,
                last_trade_pnl_pct,
            )

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
                "last_rsi_value_on_last_buy": "",
                "total_buy_calls_in_dataset": 0,
                "total_trades": 0,
                "last_trade_entry_date": "",
                "last_trade_exit_date": "",
                "last_trade_pnl_pct": "",
                "strategy_version": "buy_ma30_donch20_rsi60_trailing_v1",
                "status": "failed",
                "error_message": str(exc),
            }
            write_metadata(metadata, instrument_dir / "metadata.json")

        status_by_ticker[ticker] = status

    return status_by_ticker
