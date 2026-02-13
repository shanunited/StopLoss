from __future__ import annotations

from typing import Any

import pandas as pd


def compute_trade_state(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing stop, exits, and trade lifecycle columns."""
    result = df.copy()

    candidate_stop = (
        0.935 * result["Close"]
        + (result["Close"] - 2 * result["ATR14"])
        + result["DonchianLow15"]
    ) / 3.0
    result["CandidateStop"] = candidate_stop

    n = len(result)
    trailing_stop: list[Any] = [pd.NA] * n
    in_position: list[int] = [0] * n
    entry_date: list[Any] = [pd.NA] * n
    exit_signal: list[int] = [0] * n
    exit_price: list[Any] = [pd.NA] * n
    exit_reason: list[Any] = [pd.NA] * n
    trade_id: list[Any] = [pd.NA] * n
    holding_days: list[Any] = [pd.NA] * n
    pnl_pct: list[Any] = [pd.NA] * n

    in_pos = False
    current_trade_id = 0
    current_entry_date = None
    entry_idx = None
    entry_price = None
    prev_trailing_stop = None

    for i, (idx, row) in enumerate(result.iterrows()):
        buy_call = row.get("BuyCall") == 1
        low = row.get("Low")
        sma150 = row.get("SMA150")
        candidate = row.get("CandidateStop")
        stop_entry = row.get("StopLoss_Entry")

        if in_pos:
            trailing_hit = (
                prev_trailing_stop is not None
                and pd.notna(prev_trailing_stop)
                and pd.notna(low)
                and low <= prev_trailing_stop
            )
            sma_hit = pd.notna(sma150) and pd.notna(low) and low <= sma150
            if trailing_hit or sma_hit:
                exit_signal[i] = 1
                if trailing_hit and sma_hit:
                    exit_price[i] = max(prev_trailing_stop, sma150)
                    exit_reason[i] = "TRAILING_STOP|SMA150_BREAK"
                elif trailing_hit:
                    exit_price[i] = prev_trailing_stop
                    exit_reason[i] = "TRAILING_STOP"
                else:
                    exit_price[i] = sma150
                    exit_reason[i] = "SMA150_BREAK"
                in_position[i] = 1
                entry_date[i] = current_entry_date
                trade_id[i] = current_trade_id
                trailing_stop[i] = prev_trailing_stop

                if entry_idx is not None and entry_price is not None and pd.notna(entry_price):
                    holding_days[i] = i - entry_idx + 1
                    pnl_pct[i] = (exit_price[i] - entry_price) / entry_price * 100

                in_pos = False
                current_entry_date = None
                entry_idx = None
                entry_price = None
                prev_trailing_stop = None
                continue

            new_stop = candidate
            if prev_trailing_stop is not None and pd.notna(prev_trailing_stop):
                if pd.notna(new_stop):
                    new_stop = max(prev_trailing_stop, new_stop)
                else:
                    new_stop = prev_trailing_stop

            in_position[i] = 1
            entry_date[i] = current_entry_date
            trade_id[i] = current_trade_id
            trailing_stop[i] = new_stop
            prev_trailing_stop = new_stop
        else:
            if buy_call and pd.notna(stop_entry):
                in_pos = True
                current_trade_id += 1
                current_entry_date = idx
                entry_idx = i
                entry_price = row.get("EntryPrice")
                trailing_stop_val = stop_entry if pd.notna(stop_entry) else candidate

                in_position[i] = 1
                entry_date[i] = current_entry_date
                trade_id[i] = current_trade_id
                trailing_stop[i] = trailing_stop_val
                prev_trailing_stop = trailing_stop_val

    result["TrailingStop"] = pd.Series(trailing_stop, index=result.index)
    result["InPosition"] = pd.Series(in_position, index=result.index)
    result["EntryDate"] = pd.Series(entry_date, index=result.index)
    result["ExitSignal"] = pd.Series(exit_signal, index=result.index)
    result["ExitPrice"] = pd.Series(exit_price, index=result.index)
    result["ExitReason"] = pd.Series(exit_reason, index=result.index)
    result["TradeId"] = pd.Series(trade_id, index=result.index, dtype="Int64")
    result["HoldingDays"] = pd.Series(holding_days, index=result.index, dtype="Int64")
    result["PnL_pct"] = pd.Series(pnl_pct, index=result.index)

    return result
