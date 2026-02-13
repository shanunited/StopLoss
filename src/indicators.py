from __future__ import annotations

import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators and signals used by the StopLoss pipeline."""
    result = df.copy()

    close = result["Close"]
    high = result["High"]
    low = result["Low"]

    result["SMA150"] = close.rolling(window=150, min_periods=150).mean()
    result["MA30"] = close.rolling(window=30, min_periods=30).mean()
    result["DonchianLow15"] = low.rolling(window=15, min_periods=15).min()
    result["DonchianHigh20"] = high.rolling(window=20, min_periods=20).max()
    result["DonchianHigh20_prev"] = result["DonchianHigh20"].shift(1)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    result["RSI14"] = rsi

    donch_breakout = (close > result["DonchianHigh20_prev"]) & (
        close.shift(1) <= result["DonchianHigh20_prev"].shift(1)
    )
    ma_filter = close > result["MA30"]
    sma_filter = close > result["SMA150"]
    rsi_filter = result["RSI14"] >= 60
    buy_call = (donch_breakout & ma_filter & sma_filter & rsi_filter).fillna(False)
    result["BuyCall"] = buy_call.astype(int)

    result["EntryPrice"] = close.where(buy_call)
    result["StopLoss"] = (
        0.935 * close + (close - 2 * result["ATR14"]) + result["DonchianLow15"]
    ) / 3.0
    result["StopLoss_Entry"] = (
        0.935 * result["EntryPrice"]
        + (result["EntryPrice"] - 2 * result["ATR14"])
        + result["DonchianLow15"]
    ) / 3.0

    return result
