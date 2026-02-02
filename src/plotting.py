from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _plot_series(
    ax: plt.Axes,
    series: Optional[pd.Series],
    label: str,
    color: str,
    linewidth: float = 1.0,
    linestyle: str = "-",
) -> None:
    if series is None:
        return
    ax.plot(series.index, series.values, label=label, color=color, linewidth=linewidth, linestyle=linestyle)


def plot_instrument(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot price, moving averages, Donchian bands, and BuyCall markers."""
    fig, (ax_price, ax_atr) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    _plot_series(ax_price, df.get("Close"), "Close", "#222222", linewidth=1.2)
    _plot_series(ax_price, df.get("SMA150"), "SMA150", "#1f77b4", linewidth=0.9, linestyle="--")
    _plot_series(ax_price, df.get("MA30"), "MA30", "#ff7f0e", linewidth=1.0)
    _plot_series(
        ax_price, df.get("DonchianLow15"), "Donchian Low (15)", "#d62728", linewidth=0.9
    )
    _plot_series(
        ax_price,
        df.get("DonchianHigh20_prev"),
        "Donchian High (20) Prev",
        "#2ca02c",
        linewidth=0.9,
    )

    buy_mask = df.get("BuyCall") == 1
    if buy_mask is not None and buy_mask.any():
        buy_dates = df.index[buy_mask]
        buy_prices = df.loc[buy_mask, "Close"]
        ax_price.scatter(
            buy_dates,
            buy_prices,
            marker="^",
            color="#111111",
            s=35,
            label="BuyCall",
            zorder=5,
        )
        if "StopLoss_Entry" in df.columns:
            ax_price.scatter(
                buy_dates,
                df.loc[buy_mask, "StopLoss_Entry"],
                marker="x",
                color="#8c564b",
                s=25,
                label="StopLoss_Entry",
                zorder=5,
            )

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    _plot_series(ax_atr, df.get("ATR14"), "ATR14", "#9467bd", linewidth=1.0)
    if buy_mask is not None and buy_mask.any():
        ax_atr.scatter(
            df.index[buy_mask],
            df.loc[buy_mask, "ATR14"],
            marker="^",
            color="#111111",
            s=20,
            label="BuyCall",
            zorder=5,
        )
    ax_atr.set_ylabel("ATR")
    ax_atr.grid(True, alpha=0.3)
    ax_atr.legend(loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
