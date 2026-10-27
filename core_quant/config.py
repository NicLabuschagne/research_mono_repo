"""
config.py
---------
YAML -> frozen dataclasses, and a run store so every result can be reproduced.

    cfg = load_config(STRATEGY_DIR / "config.yaml", CrushConfig)
    save_run(STRATEGY_DIR, "validation", cfg, metrics, tables, figures)

The YAML holds the values; the dataclass defines which fields exist and their
types. An unknown key in the YAML raises, so a typo cannot silently fall back
to a default.
"""
from __future__ import annotations

import json
import subprocess
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ── Sections every strategy shares ─────────────────────────────────────────────

@dataclass(frozen=True)
class SplitConfig:
    development_start: str
    holdout_start: str                     # nothing on or after this date is used until the holdout run
    holdout_end: str | None = None


@dataclass(frozen=True)
class ValidationConfig:
    n_folds: int = 6
    periods_per_year: int = 252
    grid: dict[str, list] = field(default_factory=dict)          # param name -> values to try
    heatmaps: list[list[str]] = field(default_factory=list)      # [[x_param, y_param], ...]


@dataclass(frozen=True)
class GateConfig:
    """Every gate must pass on the development folds before the holdout may run."""
    min_mean_sharpe: float = 0.5           # mean fold Sharpe of the selected params
    min_pct_folds_positive: float = 0.6    # share of folds with Sharpe > 0
    min_trades: int = 100                  # over the whole development period
    min_wf_oos_sharpe: float = 0.0         # Sharpe of the walk-forward selection, out of sample


# ── Loading ────────────────────────────────────────────────────────────────────

def from_dict(cls, data: dict):
    """Build a (nested) dataclass from a dict, rejecting unknown keys."""
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        kind = hints[name]
        kwargs[name] = from_dict(kind, value) if is_dataclass(kind) and isinstance(value, dict) else value
    return cls(**kwargs)


def load_config(path: Path, cls):
    return from_dict(cls, yaml.safe_load(Path(path).read_text()))


# ── Run store ──────────────────────────────────────────────────────────────────

def git_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def save_run(strategy_dir: Path, kind: str, cfg, metrics: dict,
             tables: dict[str, pd.DataFrame] | None = None, figures: dict | None = None) -> Path:
    """Write runs/<timestamp>_<kind>/ with config.yaml, metrics.json, tables (csv) and figures (png)."""
    git = git_state()
    run_dir = Path(strategy_dir) / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (run_dir / "metrics.json").write_text(json.dumps({"kind": kind, "git": git, **metrics}, indent=2, default=str))
    for name, table in (tables or {}).items():
        table.to_csv(run_dir / f"{name}.csv")
    for name, figure in (figures or {}).items():
        figure.savefig(run_dir / f"{name}.png", dpi=130)
    return run_dir


def latest_run(strategy_dir: Path, kind: str) -> tuple[Path, dict[str, Any]] | None:
    """Most recent run of this kind and its metrics, or None."""
    runs = sorted((Path(strategy_dir) / "runs").glob(f"*_{kind}"))
    if not runs:
        return None
    return runs[-1], json.loads((runs[-1] / "metrics.json").read_text())
