from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .indicators import compute_indicators


FINAL_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA150",
    "MA30",
    "DonchianLow15",
    "DonchianHigh20",
    "DonchianHigh20_prev",
    "ATR14",
    "RSI14",
    "BuyCall",
    "EntryPrice",
    "CandidateStop",
    "StopLoss_Entry",
    "TrailingStop",
    "InPosition",
    "TradeId",
    "EntryDate",
    "ExitSignal",
    "ExitPrice",
    "ExitReason",
    "HoldingDays",
    "PnL_pct",
]


def _read_raw_ndjson(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])
    df = df.set_index("Date")
    return df


def _round_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def _recompute_buycall(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    breakout_today = close > df["DonchianHigh20_prev"]
    breakout_cross = close.shift(1) <= df["DonchianHigh20_prev"].shift(1)
    ma_filter = close > df["MA30"]
    sma_filter = close > df["SMA150"]
    rsi_filter = df["RSI14"] >= 60
    buycall = (breakout_today & breakout_cross & ma_filter & sma_filter & rsi_filter).fillna(False)
    return buycall.astype(int)


def _compute_candidate_stop(df: pd.DataFrame) -> pd.Series:
    return (
        0.935 * df["Close"] + (df["Close"] - 2 * df["ATR14"]) + df["DonchianLow15"]
    ) / 3.0


def _trade_engine(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    result = df.copy()
    n = len(result)

    trailing_stop: List[Any] = [pd.NA] * n
    in_position: List[int] = [0] * n
    trade_id: List[Any] = [pd.NA] * n
    entry_date: List[Any] = [pd.NA] * n
    exit_signal: List[int] = [0] * n
    exit_price: List[Any] = [pd.NA] * n
    exit_reason: List[Any] = [pd.NA] * n
    holding_days: List[Any] = [pd.NA] * n
    pnl_pct: List[Any] = [pd.NA] * n

    trades: List[Dict[str, Any]] = []

    in_pos = False
    current_trade_id = 0
    entry_idx: Optional[int] = None
    entry_px: Optional[float] = None
    entry_dt: Optional[pd.Timestamp] = None
    current_stop: Optional[float] = None

    trailing_decrease_rows: List[Dict[str, Any]] = []
    trailing_violations = 0
    exit_price_mismatch = 0
    any_exit_before_entry = False

    max_fav: Optional[float] = None
    max_adv: Optional[float] = None

    for i, (idx, row) in enumerate(result.iterrows()):
        buy_call = row.get("BuyCall") == 1
        low = row.get("Low")
        sma150 = row.get("SMA150")
        high = row.get("High")
        candidate = row.get("CandidateStop")
        stop_entry = row.get("StopLoss_Entry")

        if in_pos:
            # Exit check with previous day stop
            trailing_hit = current_stop is not None and pd.notna(current_stop) and pd.notna(low) and low <= current_stop
            sma_hit = pd.notna(sma150) and pd.notna(low) and low <= sma150
            if trailing_hit or sma_hit:
                exit_signal[i] = 1
                if trailing_hit and sma_hit:
                    exit_price[i] = max(current_stop, sma150)
                    exit_reason[i] = "TRAILING_STOP|SMA150_BREAK"
                elif trailing_hit:
                    exit_price[i] = current_stop
                    exit_reason[i] = "TRAILING_STOP"
                else:
                    exit_price[i] = sma150
                    exit_reason[i] = "SMA150_BREAK"
                in_position[i] = 1
                trade_id[i] = current_trade_id
                entry_date[i] = entry_dt
                trailing_stop[i] = current_stop
                if entry_idx is not None and entry_px is not None:
                    holding_days[i] = i - entry_idx + 1
                    pnl_pct[i] = (exit_price[i] - entry_px) / entry_px * 100
                if entry_idx is not None and entry_dt is not None:
                    trades.append(
                        {
                            "TradeId": current_trade_id,
                            "EntryDate": entry_dt,
                            "EntryPrice": entry_px,
                            "InitialStop": trailing_stop[entry_idx],
                            "ExitDate": idx,
                            "ExitPrice": exit_price[i],
                            "ExitReason": exit_reason[i],
                            "HoldingDays": holding_days[i],
                            "PnL_pct": pnl_pct[i],
                            "MaxFavorablePct": max_fav,
                            "MaxAdversePct": max_adv,
                        }
                    )
                in_pos = False
                entry_idx = None
                entry_px = None
                entry_dt = None
                current_stop = None
                max_fav = None
                max_adv = None
                continue

            # update trailing stop after exit check
            new_stop = current_stop
            if pd.notna(candidate):
                if new_stop is None or pd.isna(new_stop):
                    new_stop = candidate
                else:
                    new_stop = max(new_stop, candidate)
            if current_stop is not None and new_stop is not None:
                if pd.notna(new_stop) and pd.notna(current_stop) and new_stop < current_stop:
                    trailing_violations += 1
                    trailing_decrease_rows.append(
                        {
                            "Date": idx,
                            "TrailingStop": new_stop,
                            "PriorTrailingStop": current_stop,
                            "CandidateStop": candidate,
                        }
                    )
            current_stop = new_stop
            trailing_stop[i] = current_stop
            in_position[i] = 1
            trade_id[i] = current_trade_id
            entry_date[i] = entry_dt

            # update MFE/MAE
            if entry_px is not None:
                if pd.notna(high):
                    move = (high - entry_px) / entry_px * 100
                    max_fav = move if max_fav is None else max(max_fav, move)
                if pd.notna(low):
                    move = (low - entry_px) / entry_px * 100
                    max_adv = move if max_adv is None else min(max_adv, move)
        else:
            if buy_call and pd.notna(stop_entry):
                in_pos = True
                current_trade_id += 1
                entry_idx = i
                entry_px = float(row.get("EntryPrice"))
                entry_dt = idx
                current_stop = float(stop_entry) if pd.notna(stop_entry) else float(candidate)
                trailing_stop[i] = current_stop
                in_position[i] = 1
                trade_id[i] = current_trade_id
                entry_date[i] = entry_dt
                # initialize MFE/MAE with entry day move
                if entry_px is not None:
                    if pd.notna(high):
                        max_fav = (high - entry_px) / entry_px * 100
                    if pd.notna(low):
                        max_adv = (low - entry_px) / entry_px * 100

    if in_pos:
        any_exit_before_entry = False

    result["TrailingStop"] = trailing_stop
    result["InPosition"] = in_position
    result["TradeId"] = trade_id
    result["EntryDate"] = entry_date
    result["ExitSignal"] = exit_signal
    result["ExitPrice"] = exit_price
    result["ExitReason"] = exit_reason
    result["HoldingDays"] = holding_days
    result["PnL_pct"] = pnl_pct

    audit = {
        "trailing_violations": trailing_violations,
        "exit_price_not_prev_stop": exit_price_mismatch,
        "any_exit_before_entry": any_exit_before_entry,
        "trailing_decrease_rows": trailing_decrease_rows,
    }
    return result, trades, audit


def _compute_metrics(trades_summary: pd.DataFrame) -> Dict[str, Any]:
    trades = trades_summary.copy()
    trades["PnL_pct"] = pd.to_numeric(trades.get("PnL_pct"), errors="coerce")
    if "ExitDate" in trades.columns:
        trades["_ExitDate"] = pd.to_datetime(trades["ExitDate"], errors="coerce")
        trades = trades.sort_values("_ExitDate")

    total_trades = int(len(trades))
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "expectancy_pct": 0.0,
            "cumulative_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl = trades["PnL_pct"].fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = float(len(wins)) / float(total_trades)
    avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses.mean()) if not losses.empty else 0.0
    expectancy_pct = (win_rate * avg_win_pct) + ((1.0 - win_rate) * avg_loss_pct)

    equity = 1.0
    curve = []
    for value in pnl:
        equity *= 1.0 + value / 100.0
        curve.append(equity)

    cumulative_return_pct = (curve[-1] - 1.0) * 100.0 if curve else 0.0

    if curve:
        peak = curve[0]
        peaks = []
        for value in curve:
            peak = max(peak, value)
            peaks.append(peak)
        drawdowns = [(e - p) / p for e, p in zip(curve, peaks)]
        max_drawdown_pct = min(drawdowns) * 100.0
    else:
        max_drawdown_pct = 0.0

    return {
        "total_trades": total_trades,
        "win_rate_pct": win_rate * 100.0,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "expectancy_pct": expectancy_pct,
        "cumulative_return_pct": cumulative_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }


def _audit_checks(
    df: pd.DataFrame, trades: pd.DataFrame, audit: Dict[str, Any]
) -> Dict[str, Any]:
    buy_signals = int(pd.to_numeric(df["BuyCall"], errors="coerce").fillna(0).sum())
    trades_started = int(pd.to_numeric(df["TradeId"], errors="coerce").notna().sum())
    trades_exited = int(pd.to_numeric(df["ExitSignal"], errors="coerce").fillna(0).sum())
    open_trade_at_end = bool(pd.to_numeric(df["InPosition"], errors="coerce").fillna(0).iloc[-1] == 1)

    required_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA150",
        "MA30",
        "DonchianLow15",
        "DonchianHigh20",
        "DonchianHigh20_prev",
        "ATR14",
        "RSI14",
        "CandidateStop",
        "StopLoss_Entry",
        "TrailingStop",
    ]
    nan_cols = [col for col in required_cols if df[col].isna().any()]

    indicator_cols = ["MA30", "DonchianHigh20_prev", "RSI14"]
    buycall_nan = df.loc[df["BuyCall"] == 1, indicator_cols].isna().any(axis=1).sum()

    # exit price validation
    exit_price_mismatch = 0
    if "ExitSignal" in df.columns and "ExitReason" in df.columns:
        exit_rows = df[df["ExitSignal"] == 1].copy()
        if not exit_rows.empty:
            prev_stop = df["TrailingStop"].shift(1)
            for idx, row in exit_rows.iterrows():
                reason = str(row.get("ExitReason") or "")
                expected = None
                if "TRAILING_STOP" in reason and "SMA150_BREAK" in reason:
                    expected = max(prev_stop.loc[idx], row.get("SMA150"))
                elif "TRAILING_STOP" in reason:
                    expected = prev_stop.loc[idx]
                elif "SMA150_BREAK" in reason:
                    expected = row.get("SMA150")
                if expected is not None and pd.notna(expected):
                    if pd.notna(row.get("ExitPrice")):
                        if abs(float(row.get("ExitPrice")) - float(expected)) > 0.02:
                            exit_price_mismatch += 1

    return {
        "buy_signals": buy_signals,
        "trades_started": int(trades["TradeId"].nunique()) if not trades.empty else 0,
        "trades_exited": trades_exited,
        "open_trade_at_end": open_trade_at_end,
        "any_nan_in_required_columns": nan_cols,
        "any_exit_before_entry": audit["any_exit_before_entry"],
        "any_trailing_stop_decreased": audit["trailing_violations"],
        "any_exit_price_not_equal_prev_stop": exit_price_mismatch,
        "any_buycall_where_indicators_nan": int(buycall_nan),
    }


def _write_audit_outputs(
    ticker: str,
    output_dir: Path,
    trades: pd.DataFrame,
    audit: Dict[str, Any],
    checks: Dict[str, Any],
) -> None:
    audit_trades_path = output_dir / ticker / "audit_trades.csv"
    trades.to_csv(audit_trades_path, index=False)

    audit_checks_path = output_dir / ticker / "audit_checks.json"
    audit_checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    if audit["trailing_violations"] > 0:
        violations_path = output_dir / ticker / "audit_trailing_violations.csv"
        pd.DataFrame(audit["trailing_decrease_rows"]).to_csv(violations_path, index=False)


def _build_trades_summary(ticker: str, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
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
    summary = trades.copy()
    summary.insert(0, "Ticker", ticker)
    return summary[
        [
            "Ticker",
            "TradeId",
            "EntryDate",
            "ExitDate",
            "EntryPrice",
            "ExitPrice",
            "HoldingDays",
            "PnL_pct",
        ]
    ]


def _recompute_one(ticker: str, output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ticker_dir = output_dir / ticker
    raw_path = ticker_dir / "raw_ohlc.json"
    if not raw_path.exists():
        return {}, {"ticker": ticker, "error": "missing raw_ohlc.json"}

    df = _read_raw_ndjson(raw_path)
    if df.empty:
        return {}, {"ticker": ticker, "error": "raw_ohlc.json is empty"}

    df = _round_ohlc(df)
    df = df.sort_index()

    indicators = compute_indicators(df)
    indicators["BuyCall"] = _recompute_buycall(indicators)
    indicators["EntryPrice"] = indicators["Close"].where(indicators["BuyCall"] == 1)
    indicators["CandidateStop"] = _compute_candidate_stop(indicators)
    indicators["StopLoss_Entry"] = (
        0.935 * indicators["EntryPrice"]
        + (indicators["EntryPrice"] - 2 * indicators["ATR14"])
        + indicators["DonchianLow15"]
    ) / 3.0

    recomputed, trades_list, audit = _trade_engine(indicators)

    trades = pd.DataFrame(trades_list)
    if not trades.empty:
        trades["EntryDate"] = pd.to_datetime(trades["EntryDate"]).dt.strftime("%Y-%m-%d")
        trades["ExitDate"] = pd.to_datetime(trades["ExitDate"]).dt.strftime("%Y-%m-%d")

    # round indicator outputs for saved CSV
    for col in [
        "SMA150",
        "MA30",
        "DonchianLow15",
        "DonchianHigh20",
        "DonchianHigh20_prev",
        "ATR14",
        "RSI14",
        "CandidateStop",
        "StopLoss_Entry",
        "TrailingStop",
        "ExitPrice",
        "PnL_pct",
        "EntryPrice",
    ]:
        if col in recomputed.columns:
            recomputed[col] = pd.to_numeric(recomputed[col], errors="coerce").round(2)

    output = recomputed.copy()
    output = output.reset_index().rename(columns={"index": "Date"})
    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    if "EntryDate" in output.columns:
        output["EntryDate"] = pd.to_datetime(output["EntryDate"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    output = output[FINAL_COLUMNS]
    output.to_csv(ticker_dir / "final.csv", index=False)

    # trades summary + performance metrics
    trades_summary = _build_trades_summary(ticker, trades)
    trades_summary.to_csv(ticker_dir / "trades_summary.csv", index=False)

    metrics = _compute_metrics(trades_summary)
    metrics_payload = {"ticker": ticker, **metrics}
    (ticker_dir / "performance_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    checks = _audit_checks(output, trades_summary, audit)
    _write_audit_outputs(ticker, output_dir, trades, audit, checks)

    overview = {
        "Ticker": ticker,
        "total_trades": metrics["total_trades"],
        "win_rate_pct": metrics["win_rate_pct"],
        "cumulative_return_pct": metrics["cumulative_return_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "expectancy_pct": metrics["expectancy_pct"],
        "buy_signals": checks["buy_signals"],
        "trailing_violations": checks["any_trailing_stop_decreased"],
        "open_trade_at_end": checks["open_trade_at_end"],
    }
    return overview, checks


def run(output_dir: Path, tickers: List[str]) -> None:
    summary_dir = output_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for ticker in tickers:
        overview, checks = _recompute_one(ticker, output_dir)
        if not overview:
            failures.append(ticker)
            continue
        overview_rows.append(overview)
        print(
            f"{ticker}: trades={overview['total_trades']}, "
            f"cum_return={overview['cumulative_return_pct']:.2f}%, "
            f"maxDD={overview['max_drawdown_pct']:.2f}%, "
            f"violations={overview['trailing_violations']}"
        )
        if checks["any_trailing_stop_decreased"] > 0 or checks["any_exit_price_not_equal_prev_stop"] > 0:
            raise RuntimeError(
                f"{ticker}: violations found "
                f"(trailing_decreased={checks['any_trailing_stop_decreased']}, "
                f"exit_price_mismatch={checks['any_exit_price_not_equal_prev_stop']})"
            )

    if overview_rows:
        pd.DataFrame(overview_rows).to_csv(summary_dir / "audit_overview.csv", index=False)

    if failures:
        raise RuntimeError(f"Failed tickers: {', '.join(failures)}")


def _list_tickers(output_dir: Path) -> List[str]:
    tickers: List[str] = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name in {"_summary", "status_reports"}:
            continue
        tickers.append(d.name)
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and audit stoploss pipeline.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory containing per-ticker folders.",
    )
    parser.add_argument(
        "--tickers",
        default="ALL",
        help="Comma-separated list of tickers or ALL.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.tickers.upper() == "ALL":
        tickers = _list_tickers(output_dir)
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    run(output_dir, tickers)


if __name__ == "__main__":
    main()
