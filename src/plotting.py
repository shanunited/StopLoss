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
    """Plot price with SMA150 and Donchian(15), plus ATR(14) in a second panel."""
    fig, (ax_price, ax_atr) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    _plot_series(ax_price, df.get("Close"), "Close", "#222222", linewidth=1.2)
    _plot_series(ax_price, df.get("SMA150"), "SMA150", "#1f77b4", linewidth=1.1)
    _plot_series(ax_price, df.get("DonchianUpper15"), "Donchian Upper (15)", "#2ca02c", linewidth=0.9)
    _plot_series(ax_price, df.get("DonchianLower15"), "Donchian Lower (15)", "#d62728", linewidth=0.9)

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    _plot_series(ax_atr, df.get("ATR14"), "ATR14", "#9467bd", linewidth=1.0)
    ax_atr.set_ylabel("ATR")
    ax_atr.grid(True, alpha=0.3)
    ax_atr.legend(loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
