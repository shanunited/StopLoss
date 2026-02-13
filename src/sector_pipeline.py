from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from .indicators import compute_indicators
from .performance import _compute_metrics, _equity_curve
from .strategy import compute_trade_state


@dataclass(frozen=True)
class Sector:
    name: str
    ticker: str


SECTORS = [
    Sector("NSE_BANK", "^NSEBANK"),
    Sector("NSE_IT", "^CNXIT"),
    Sector("NSE_PHARMA", "^CNXPHARMA"),
    Sector("NSE_AUTO", "^CNXAUTO"),
    Sector("NSE_METAL", "^CNXMETAL"),
    Sector("NSE_ENERGY", "^CNXENERGY"),
    Sector("NSE_FMCG", "^CNXFMCG"),
    Sector("NSE_INFRA", "^CNXINFRA"),
    Sector("NSE_PSU", "^CNXPSU"),
    Sector("NSE_FIN_SERVICE", "^NIFTY_FIN_SERVICE.NS"),
]


FINAL_COLUMNS = [
    "Date",
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
    "Sector",
]


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def _normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    cols = df.columns
    if cols.nlevels >= 2:
        lvl0 = cols.get_level_values(0)
        lvl1 = cols.get_level_values(1)
        if ticker in set(lvl1):
            return df.xs(ticker, level=1, axis=1).copy()
        if ticker in set(lvl0):
            return df.xs(ticker, level=0, axis=1).copy()
        if len(set(lvl1)) == 1:
            return df.droplevel(1, axis=1)
        if len(set(lvl0)) == 1:
            return df.droplevel(0, axis=1)
    return df


def _round_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def _trades_summary(df: pd.DataFrame, sector: str, ticker: str) -> pd.DataFrame:
    exits = df[df["ExitSignal"] == 1].copy()
    if exits.empty:
        return pd.DataFrame(
            columns=[
                "Sector",
                "Ticker",
                "TradeId",
                "EntryDate",
                "ExitDate",
                "EntryPrice",
                "ExitPrice",
                "HoldingDays",
                "PnL_pct",
            ]
        )
    summary = exits[
        [
            "TradeId",
            "EntryDate",
            "Date",
            "EntryPrice",
            "ExitPrice",
            "HoldingDays",
            "PnL_pct",
        ]
    ].copy()
    summary = summary.rename(columns={"Date": "ExitDate"})
    summary.insert(0, "Ticker", ticker)
    summary.insert(0, "Sector", sector)
    return summary


def _write_equity_curve(trades: pd.DataFrame, output_path: Path, sector: str, ticker: str) -> None:
    if trades.empty:
        pd.DataFrame(columns=["Sector", "Ticker", "Trade", "Equity"]).to_csv(output_path, index=False)
        return
    x, equity = _equity_curve(trades)
    pd.DataFrame({"Sector": sector, "Ticker": ticker, "Trade": x, "Equity": equity}).to_csv(
        output_path, index=False
    )


def _plot_equity_curve(trades: pd.DataFrame, output_path: Path, title: str) -> None:
    if trades.empty:
        return
    x, equity = _equity_curve(trades)
    plt.figure(figsize=(10, 4))
    plt.plot(x, equity, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_outputs(sector: Sector, output_dir: Path, df: pd.DataFrame) -> None:
    sector_dir = output_dir / sector.name
    sector_dir.mkdir(parents=True, exist_ok=True)

    raw_path = sector_dir / "raw_ohlc.json"
    df_reset = df.reset_index().rename(columns={"index": "Date"})
    df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.strftime("%Y-%m-%d")
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in df_reset.to_dict(orient="records"):
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def run(
    output_dir: Path, end_date: Optional[str] = None, full_backfill_from: Optional[str] = None
) -> None:
    end_dt = pd.to_datetime(end_date).date() if end_date else date.today()
    if full_backfill_from:
        start_dt = pd.to_datetime(full_backfill_from).date()
    else:
        start_dt = end_dt - timedelta(days=3650)

    for sector in SECTORS:
        sector_dir = output_dir / sector.name
        sector_dir.mkdir(parents=True, exist_ok=True)

        df_raw = _download_ohlcv(
            sector.ticker, start_dt.isoformat(), (end_dt + timedelta(days=1)).isoformat()
        )
        df_raw = _normalize_ohlcv_columns(df_raw, sector.ticker)
        if df_raw is None or df_raw.empty:
            continue
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.sort_index()
        df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"])
        df_raw = _round_ohlc(df_raw)
        df_raw = df_raw[~df_raw.index.duplicated(keep="last")]
        if full_backfill_from:
            earliest = df_raw.index.min().date() if not df_raw.empty else None
            if earliest and earliest > start_dt:
                print(
                    f"{sector.name} ({sector.ticker}): earliest date {earliest} after requested {start_dt}"
                )

        _save_outputs(sector, output_dir, df_raw)

        indicators = compute_indicators(df_raw)
        indicators = compute_trade_state(indicators)
        indicators["Sector"] = sector.name

        # round indicator outputs
        for col in [
            "SMA150",
            "MA30",
            "ATR14",
            "RSI14",
            "DonchianHigh20",
            "DonchianHigh20_prev",
            "DonchianLow15",
            "StopLoss",
            "StopLoss_Entry",
            "CandidateStop",
            "TrailingStop",
            "ExitPrice",
            "PnL_pct",
            "EntryPrice",
        ]:
            if col in indicators.columns:
                indicators[col] = pd.to_numeric(indicators[col], errors="coerce").round(2)

        output = indicators.reset_index().rename(columns={"index": "Date"})
        output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
        if "EntryDate" in output.columns:
            output["EntryDate"] = pd.to_datetime(output["EntryDate"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
        if "ExitDate" in output.columns:
            output["ExitDate"] = pd.to_datetime(output["ExitDate"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

        output = output[FINAL_COLUMNS]
        output.to_csv(sector_dir / "final.csv", index=False)

        trades = _trades_summary(output, sector.name, sector.ticker)
        if not trades.empty:
            trades["_ExitDate"] = pd.to_datetime(trades["ExitDate"], errors="coerce")
            trades = trades.sort_values("_ExitDate").drop(columns=["_ExitDate"])
        trades.to_csv(sector_dir / "trades_summary.csv", index=False)

        metrics = _compute_metrics(trades)
        metrics_payload = {"ticker": sector.ticker, "sector": sector.name, **metrics}
        (sector_dir / "performance_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )

        _write_equity_curve(trades, sector_dir / "equity_curve.csv", sector.name, sector.ticker)
        _plot_equity_curve(trades, sector_dir / "equity_curve.png", f"{sector.name} Equity Curve")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sector-based stoploss analysis.")
    parser.add_argument(
        "--output-dir",
        default="data/stoploss_output",
        help="Base output directory.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional ISO end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--full-backfill-from",
        default=None,
        help="Rebuild raw data from this date (YYYY-MM-DD) through end date.",
    )
    args = parser.parse_args()
    run(Path(args.output_dir), args.end_date, args.full_backfill_from)


if __name__ == "__main__":
    main()
