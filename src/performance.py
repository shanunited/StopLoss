from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_COLUMNS = [
    "Ticker",
    "TradeId",
    "EntryDate",
    "ExitDate",
    "EntryPrice",
    "ExitPrice",
    "HoldingDays",
    "PnL_pct",
]


def _load_final_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _build_entry_lookup(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["TradeId"] = pd.to_numeric(data.get("TradeId"), errors="coerce")
    data = data[data["TradeId"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=["TradeId", "EntryDate", "EntryPrice"])

    data["_Date"] = pd.to_datetime(data.get("Date"), errors="coerce")
    data["_EntryDate"] = pd.to_datetime(data.get("EntryDate"), errors="coerce")
    data["_EntryPrice"] = pd.to_numeric(data.get("EntryPrice"), errors="coerce")

    entries: List[Dict[str, Any]] = []
    for trade_id, group in data.groupby("TradeId", sort=False):
        entry_date = group["_EntryDate"].dropna()
        if not entry_date.empty:
            entry_dt = entry_date.iloc[0]
        else:
            entry_dt = group["_Date"].min()

        entry_row = group[group["_Date"] == entry_dt]
        if entry_row.empty:
            entry_row = group

        entry_date_str = entry_row["Date"].iloc[0]
        entry_price = entry_row["_EntryPrice"].dropna()
        if not entry_price.empty:
            entry_price_value = float(entry_price.iloc[0])
        else:
            fallback = group["_EntryPrice"].dropna()
            entry_price_value = float(fallback.iloc[0]) if not fallback.empty else None

        entries.append(
            {
                "TradeId": trade_id,
                "EntryDate": entry_date_str,
                "EntryPrice": entry_price_value,
            }
        )

    return pd.DataFrame(entries)


def _extract_trades(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    exit_signal = pd.to_numeric(df.get("ExitSignal"), errors="coerce").fillna(0)
    exits = df[exit_signal == 1].copy()
    if exits.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    exits["_TradeId"] = pd.to_numeric(exits.get("TradeId"), errors="coerce")
    exits = exits.drop(columns=["EntryDate", "EntryPrice"], errors="ignore")
    entry_lookup = _build_entry_lookup(df)
    exits = exits.merge(entry_lookup, left_on="_TradeId", right_on="TradeId", how="left")

    summary = pd.DataFrame(
        {
            "Ticker": ticker,
            "TradeId": exits["_TradeId"],
            "EntryDate": exits["EntryDate"],
            "ExitDate": exits.get("Date"),
            "EntryPrice": exits["EntryPrice"],
            "ExitPrice": exits.get("ExitPrice"),
            "HoldingDays": exits.get("HoldingDays"),
            "PnL_pct": exits.get("PnL_pct"),
        }
    )
    return summary[SUMMARY_COLUMNS]


def _sort_trades_by_exit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "ExitDate" not in trades.columns:
        return trades
    temp = trades.copy()
    temp["_ExitDate"] = pd.to_datetime(temp["ExitDate"], errors="coerce")
    if temp["_ExitDate"].notna().any():
        temp = temp.sort_values("_ExitDate")
    return temp.drop(columns=["_ExitDate"])


def _equity_curve(trades: pd.DataFrame) -> Tuple[List[int], List[float]]:
    if trades.empty:
        return [], []
    pnl = pd.to_numeric(trades["PnL_pct"], errors="coerce").fillna(0.0)
    equity = 1.0
    curve: List[float] = []
    for value in pnl:
        equity *= 1.0 + (value / 100.0)
        curve.append(equity)
    x = list(range(1, len(curve) + 1))
    return x, curve


def _compute_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
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

    pnl = pd.to_numeric(trades["PnL_pct"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = float(len(wins)) / float(total_trades)
    win_rate_pct = win_rate * 100.0
    avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses.mean()) if not losses.empty else 0.0
    expectancy_pct = (win_rate * avg_win_pct) + ((1.0 - win_rate) * avg_loss_pct)

    _, equity = _equity_curve(trades)
    cumulative_return_pct = (equity[-1] - 1.0) * 100.0 if equity else 0.0

    if equity:
        peaks = []
        peak = equity[0]
        for value in equity:
            peak = max(peak, value)
            peaks.append(peak)
        drawdowns = [(e - p) / p for e, p in zip(equity, peaks)]
        max_drawdown_pct = min(drawdowns) * 100.0
    else:
        max_drawdown_pct = 0.0

    return {
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "expectancy_pct": expectancy_pct,
        "cumulative_return_pct": cumulative_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }


def _plot_equity_curve(
    ticker: str, x: List[int], equity: List[float], output_path: Path
) -> None:
    if not equity:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(x, equity, marker="o", linewidth=1.5)
    plt.title(f"Equity Curve - {ticker}")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run(output_dir: Path) -> None:
    summary_dir = output_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    metrics_payload_rows: List[Dict[str, Any]] = []

    for ticker_dir in sorted(output_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        if ticker_dir.name in {"_summary", "status_reports"}:
            continue

        final_path = ticker_dir / "final.csv"
        if not final_path.exists():
            continue

        df = _load_final_csv(final_path)
        trades = _extract_trades(df, ticker_dir.name)
        trades = _sort_trades_by_exit(trades)

        trades_summary_path = ticker_dir / "trades_summary.csv"
        trades.to_csv(trades_summary_path, index=False)

        metrics = _compute_metrics(trades)
        metrics_payload = {"ticker": ticker_dir.name, **metrics}
        metrics_path = ticker_dir / "performance_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        x, equity = _equity_curve(trades)
        _plot_equity_curve(ticker_dir.name, x, equity, ticker_dir / "equity_curve.png")

        metrics_rows.append(
            {
                "Ticker": ticker_dir.name,
                "total_trades": metrics["total_trades"],
                "win_rate_pct": metrics["win_rate_pct"],
                "cumulative_return_pct": metrics["cumulative_return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "expectancy_pct": metrics["expectancy_pct"],
            }
        )
        metrics_payload_rows.append(metrics_payload)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = summary_dir / "metrics_by_ticker.csv"
        metrics_xlsx = summary_dir / "metrics_by_ticker.xlsx"
        metrics_df.to_csv(metrics_csv, index=False)

        metrics_payload_df = pd.DataFrame(metrics_payload_rows)
        with pd.ExcelWriter(metrics_xlsx, engine="openpyxl") as writer:
            metrics_df.to_excel(writer, sheet_name="summary", index=False)
            metrics_payload_df.to_excel(writer, sheet_name="metrics_json", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-ticker performance metrics.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base output directory containing per-ticker folders.",
    )
    args = parser.parse_args()
    run(Path(args.output_dir))


if __name__ == "__main__":
    main()
