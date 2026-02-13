"""Airflow DAG for daily stoploss pipeline run.

Backfill example:
    airflow dags backfill -s 2026-01-22 -e 2026-01-23 stoploss_daily_dag
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import get_current_context


def _ensure_repo_on_path() -> Optional[str]:
    """Add repo root to sys.path so `src.*` imports resolve."""
    candidates = [
        os.getenv("STOPLOSS_REPO_PATH"),
        "/opt/airflow/stoploss_repo",
        "/opt/airflow/StopLoss",
        "/opt/airflow/stoploss",
        "/opt/airflow/dags/StopLoss",
        "/opt/airflow/dags/stoploss",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return None


def _load_run_pipeline():
    """Import run_pipeline lazily to avoid DAG parse failures."""
    _ensure_repo_on_path()
    try:
        from src.stoploss.run_pipeline import run_pipeline  # type: ignore

        return run_pipeline
    except Exception:
        from stoploss.run_pipeline import run_pipeline  # type: ignore

        return run_pipeline


def _parse_instruments_var(raw: str) -> list[dict]:
    if not raw:
        return []
    instruments: list[dict] = []
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "|" in entry:
            name, ticker = [part.strip() for part in entry.split("|", 1)]
        elif ":" in entry:
            name, ticker = [part.strip() for part in entry.split(":", 1)]
        else:
            name, ticker = entry, entry
        if name and ticker:
            instruments.append({"name": name, "ticker": ticker})
    return instruments


@dag(
    schedule="30 16 * * 1-5",
    start_date=pendulum.datetime(2026, 1, 22, tz="Asia/Kolkata"),
    catchup=True,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=60),
    },
    tags=["stoploss", "yfinance"],
)
def stoploss_daily_dag():
    @task
    def run_stoploss_pipeline() -> dict:
        context = get_current_context()
        logical_date = context["logical_date"].date()
        run_id = context.get("run_id")
        ds = context.get("ds")
        logging.info("StopLoss DAG run_id=%s logical_date=%s", run_id, logical_date)

        output_dir = Variable.get(
            "STOPLOSS_OUTPUT_DIR", default_var="/opt/airflow/data/stoploss_output"
        )
        config_path = Variable.get("STOPLOSS_CONFIG_PATH", default_var="")
        config_path = config_path.strip() or None

        instruments_raw = Variable.get("STOPLOSS_INSTRUMENTS", default_var="")
        instruments = _parse_instruments_var(instruments_raw)

        run_pipeline = _load_run_pipeline()

        status = run_pipeline(
            instruments=instruments or None,
            start_date=None,
            end_date=str(logical_date),
            output_dir=output_dir,
            config_path=config_path,
            enable_charts=True,
        )

        status_dir = Path(output_dir) / "status_reports"
        status_dir.mkdir(parents=True, exist_ok=True)
        report_path = status_dir / f"{ds}.json"
        payload = {
            "run_id": run_id,
            "logical_date": str(logical_date),
            "status": status,
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("Wrote status report to %s", report_path)

        return status

    run_stoploss_pipeline()


stoploss_daily_dag()
