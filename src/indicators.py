from __future__ import annotations

import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA150, Donchian(15), and ATR(14) using Wilder smoothing."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    sma_150 = close.rolling(window=150, min_periods=150).mean()

    donchian_upper_15 = high.rolling(15, min_periods=15).max()
    donchian_lower_15 = low.rolling(15, min_periods=15).min()
    donchian_mid_15 = (donchian_upper_15 + donchian_lower_15) / 2.0

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    return pd.DataFrame(
        {
            "SMA150": sma_150,
            "SMA150_prev": sma_150.shift(1),
            "DonchianUpper15": donchian_upper_15,
            "DonchianLower15": donchian_lower_15,
            "DonchianMiddle15": donchian_mid_15,
            "DonchianUpper15_prev": donchian_upper_15.shift(1),
            "DonchianLower15_prev": donchian_lower_15.shift(1),
            "DonchianMiddle15_prev": donchian_mid_15.shift(1),
            "ATR14": atr_14,
        },
        index=df.index,
    )
