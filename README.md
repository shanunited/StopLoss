# Stop Loss Compute - Daily OHLCV + Indicators

This project downloads daily OHLCV data from Yahoo Finance via `yfinance`, rounds OHLC values to exactly 2 decimals immediately, computes SMA150, MA30, DonchianLow15, DonchianHigh20, ATR14, StopLoss, and BuyCall signals, and saves per-instrument outputs plus charts.

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
    <TICKER>/
      raw_ohlc.json
      final.csv
      chart_last10y.png
      metadata.json
    _summary/
      run_log.txt
```

Per-instrument folders are named by ticker symbol (e.g., `RELIANCE.NS`).

## Indicator formulas (daily)

All indicators are computed after rounding `Open/High/Low/Close` to 2 decimals.

- SMA150 (Close):
  - `SMA150 = Close.rolling(window=150, min_periods=150).mean()`

- MA30 (Close):
  - `MA30 = Close.rolling(window=30, min_periods=30).mean()`

- Donchian Low 15 (Low):
  - `DonchianLow15 = Low.rolling(15, min_periods=15).min()`

- Donchian High 20 (High):
  - `DonchianHigh20 = High.rolling(20, min_periods=20).max()`
  - `DonchianHigh20_prev = DonchianHigh20.shift(1)`

- ATR14 (Wilder smoothing):
  - `prev_close = Close.shift(1)`
  - `TR = max(High - Low, abs(High - prev_close), abs(Low - prev_close))`
  - `ATR14 = TR.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`

- RSI14 (Wilder):
  - `delta = Close.diff()`
  - `gain = delta.clip(lower=0)`
  - `loss = (-delta).clip(lower=0)`
  - `avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`
  - `avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()`
  - `RSI14 = 100 - (100 / (1 + avg_gain/avg_loss))` (with divide-by-zero guards)

All indicator outputs are rounded to 2 decimals in the saved CSVs.

- StopLoss:
  - `StopLoss = (0.935 * Close + (Close - 2 * ATR14) + DonchianLow15) / 3`

- BuyCall (breakout + filter):
  - `donch_breakout = (Close[t] > DonchianHigh20_prev[t]) AND (Close[t-1] <= DonchianHigh20_prev[t-1])`
  - `ma_filter = Close[t] > MA30[t]`
  - `rsi_filter = RSI14[t] >= 60`
  - `BuyCall = donch_breakout AND ma_filter AND rsi_filter`
  - `EntryPrice = Close when BuyCall is True`
  - `StopLoss_Entry = (0.935 * EntryPrice + (EntryPrice - 2 * ATR14) + DonchianLow15) / 3`

## Output details

For each instrument, you will find:
- `raw_ohlc.json`: NDJSON raw OHLCV with OHLC rounded to 2 decimals
- `final.csv`: Date, OHLCV, SMA150, MA30, DonchianLow15, DonchianHigh20, DonchianHigh20_prev, ATR14, RSI14, StopLoss, BuyCall, EntryPrice, StopLoss_Entry
- `chart_last10y.png`: price panel + ATR panel
- `metadata.json`: parameters, run IDs, and row counts

A summary log is written to:
- `output/_summary/run_log.txt`

When running in Airflow, a daily status report is written to:
- `{STOPLOSS_OUTPUT_DIR}/status_reports/{ds}.json`

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

- The data is downloaded with `auto_adjust=True` and daily interval to avoid false signals from corporate actions.
- Missing OHLC rows are dropped prior to indicator calculation; check `metadata.json` for dropped row counts.

## Airflow usage

This repo includes an Airflow DAG: `dags/stoploss_daily_dag.py`.

Key behavior:
- Runs Mon-Fri at 18:00 Asia/Kolkata
- Catchup enabled from 2026-01-22
- Uses incremental downloads (only missing dates)
- Handles "no new data" gracefully and still produces a status report

### Placement

Place the DAG file in your Airflow DAGs folder. Ensure the repo is either:
- Installed as a package, or
- Mounted so Airflow can import `src.stoploss.run_pipeline`

If mounted, you can also set `STOPLOSS_REPO_PATH` to the repo root so the DAG can add it to `sys.path`.

### Docker Compose (LocalExecutor)

Minimal example:
```
version: "3.8"

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - airflow_postgres:/var/lib/postgresql/data

  airflow:
    image: apache/airflow:2.9.3-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__FERNET_KEY: ""
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
      STOPLOSS_REPO_PATH: /opt/airflow/stoploss_repo
      STOPLOSS_OUTPUT_DIR: /opt/airflow/data/stoploss_output
    volumes:
      - ./dags:/opt/airflow/dags
      - ./:/opt/airflow/stoploss_repo
      - ./data:/opt/airflow/data
    depends_on:
      - postgres
    command: >
      bash -c "airflow db migrate &&
               airflow users create --username admin --password admin --firstname admin --lastname admin --role Admin --email admin@example.com &&
               airflow webserver & airflow scheduler"

volumes:
  airflow_postgres:
```

### Airflow Variables

- `STOPLOSS_INSTRUMENTS`: comma-separated list. Each entry can be `Name|TICKER` or just `TICKER`.
- `STOPLOSS_OUTPUT_DIR`: base output directory (default: `/opt/airflow/data/stoploss_output`)
- `STOPLOSS_CONFIG_PATH`: optional path to `config.yaml`

### Backfill for 2026-01-22 and 2026-01-23

```
airflow dags backfill -s 2026-01-22 -e 2026-01-23 stoploss_daily_dag
```
