from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_INSTRUMENTS = [
    {"name": "Fusion Finance", "ticker": "FUSION.NS"},
    {"name": "SGS", "ticker": "SYRMA.NS"},
    {"name": "Reliance", "ticker": "RELIANCE.NS"},
    {"name": "Tatva Chintan", "ticker": "TATVA.NS"},
    {"name": "Zomato (Eternal)", "ticker": "ETERNAL.NS"},
    {"name": "Gabriel", "ticker": "GABRIEL.NS"},
    {"name": "Narayan Rudra", "ticker": "NH.NS"},
    {"name": "Laurus Labs", "ticker": "LAURUSLABS.NS"},
    {"name": "BHEL", "ticker": "BHEL.NS"},
    {"name": "Bank of Maharashtra", "ticker": "MAHABANK.NS"},
    {"name": "Shriram Finance", "ticker": "SHRIRAMFIN.NS"},
    {"name": "Shriram Pistons", "ticker": "SHRIPISTON.NS"},
    {"name": "Nifty 50 Index", "ticker": "^NSEI"},
    {"name": "Blue Jet Healthcare", "ticker": "BLUEJET.NS"},
    {"name": "Windlas Biotech", "ticker": "WINDLAS.NS"},
    {"name": "Nifty MidSmall 400", "ticker": "NIFTYMIDSML400.NS"},
    {"name": "Nifty Smallcap 100", "ticker": "^CNXSC"},
    {"name": "ITC", "ticker": "ITC.NS"},
    {"name": "State Bank of India", "ticker": "SBIN.NS"},
    {"name": "ICICI Bank", "ticker": "ICICIBANK.NS"},
    {"name": "Larsen & Toubro", "ticker": "LT.NS"},
    {"name": "TMCV", "ticker": "TMCV.NS"},
    {"name": "Hindustan Unilever", "ticker": "HINDUNILVR.NS"},
    {"name": "NTPC", "ticker": "NTPC.NS"},
    {"name": "ONGC", "ticker": "ONGC.NS"},
    {"name": "Axis Bank", "ticker": "AXISBANK.NS"},
    {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS"},
]


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "instruments": DEFAULT_INSTRUMENTS,
        "output_dir": "output",
        "start": None,
        "end": None,
        "full_backfill_from": None,
    }


def _merge_config(
    base: Dict[str, Any], override: Dict[str, Any], warnings: List[str]
) -> Dict[str, Any]:
    result = dict(base)
    for key in ["output_dir", "start", "end", "full_backfill_from"]:
        if key in override and override[key] is not None:
            result[key] = override[key]

    if "instruments" in override:
        if isinstance(override["instruments"], list):
            cleaned: List[Dict[str, str]] = []
            for item in override["instruments"]:
                if isinstance(item, dict) and "name" in item and "ticker" in item:
                    cleaned.append(
                        {"name": str(item["name"]), "ticker": str(item["ticker"])}
                    )
                else:
                    warnings.append(
                        "Skipping invalid instrument entry in config.yaml; expected name and ticker"
                    )
            if cleaned:
                result["instruments"] = cleaned
        else:
            warnings.append("config.yaml instruments must be a list; using defaults")

    return result


def load_config(config_path: Path | None = None) -> Tuple[Dict[str, Any], List[str]]:
    """Load config.yaml if present, otherwise return defaults."""
    cfg = default_config()
    warnings: List[str] = []
    yaml_path = config_path or (get_project_root() / "config.yaml")
    if yaml_path.exists():
        try:
            import yaml  # type: ignore
        except ImportError:
            warnings.append(
                "config.yaml found but PyYAML is not installed; ignoring config.yaml"
            )
        else:
            try:
                user_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                warnings.append(f"Failed to read config.yaml: {exc}")
            else:
                if isinstance(user_cfg, dict):
                    cfg = _merge_config(cfg, user_cfg, warnings)
                else:
                    warnings.append(
                        "config.yaml must contain a mapping at top level; ignoring"
                    )

    return cfg, warnings
