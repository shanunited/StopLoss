# Stop Loss Compute - Daily OHLCV + Indicators

This project downloads 10 years of daily OHLCV data from Yahoo Finance via `yfinance`, rounds OHLC values to exactly 2 decimals immediately, computes SMA150, Donchian(15), and ATR14, and saves per-instrument outputs plus charts.

## Setup and run

### Windows (PowerShell)
```
.\run.ps1
```

### macOS/Linux
```
./run.sh
```

Both scripts:
- Create a `.venv` virtual environment
- Install dependencies from `requirements.txt`
- Run `python -m src.main`

## Folder structure
```
project_root/
  README.md
  requirements.txt
  run.sh
  run.ps1
  src/
    main.py
    indicators.py
    plotting.py
    io_utils.py
    config.py
  output/
    <SAFE_FOLDER_NAME>/
      ohlc_2dp.csv
      indicators.csv
      merged_with_indicators.csv
      chart_last10y.png
      metadata.json
    _summary/
      run_log.txt
      status_report.csv
```

`SAFE_FOLDER_NAME` is derived from instrument name + ticker (uppercase, non-alphanumerics replaced with `_`).

## Indicator formulas (daily)

All indicators are computed after rounding `Open/High/Low/Close` to 2 decimals.

- SMA150 (Close):
  - `SMA150 = Close.rolling(window=150, min_periods=150).mean()`
  - `SMA150_prev = SMA150.shift(1)`

- Donchian 15 (High/Low):
  - `Upper15 = High.rolling(15, min_periods=15).max()`
  - `Lower15 = Low.rolling(15, min_periods=15).min()`
  - `Middle15 = (Upper15 + Lower15) / 2`
  - Previous-day versions:
    - `Upper15_prev = Upper15.shift(1)`
    - `Lower15_prev = Lower15.shift(1)`
    - `Middle15_prev = Middle15.shift(1)`

- ATR14 (Wilder smoothing):
  - `prev_close = Close.shift(1)`
  - `TR = max(High - Low, abs(High - prev_close), abs(Low - prev_close))`
  - `ATR14 = TR.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`

All indicator outputs are rounded to 2 decimals in the saved CSVs.

## Output details

For each instrument, you will find:
- `ohlc_2dp.csv`: OHLCV with OHLC rounded to 2 decimals
- `indicators.csv`: SMA150, Donchian(15) (+ _prev variants), ATR14
- `merged_with_indicators.csv`: OHLCV + indicators
- `chart_last10y.png`: price panel + ATR panel
- `metadata.json`: parameters and row counts

A summary is written to:
- `output/_summary/run_log.txt`
- `output/_summary/status_report.csv`

## Config override (optional)

If `config.yaml` exists in the project root, it can override instruments and date ranges.

Example:
```
output_dir: output
start: "2014-01-01"
end: "2024-01-01"
instruments:
  - name: Reliance
    ticker: RELIANCE.NS
  - name: Nifty 50 Index
    ticker: ^NSEI
```

## Notes for future stop-loss system

- Use the `*_prev` columns (e.g., `SMA150_prev`, `DonchianUpper15_prev`) to avoid lookahead bias when generating signals.
- The data is downloaded with `auto_adjust=True` and daily interval to avoid false signals from corporate actions.
- Missing OHLC rows are dropped prior to indicator calculation; check `metadata.json` for dropped row counts.
