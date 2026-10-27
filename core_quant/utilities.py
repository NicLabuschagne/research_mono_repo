"""
utilities.py
------------
Everyday helpers: data paths, loading / saving processed data, a frame summary,
an optional LSE price feed, and a few common transforms.
"""
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from lse import LSE           # optional data feed; everything else works without it
except ImportError:
    LSE = None


# ── Workspace paths ────────────────────────────────────────────────────────────

from core_quant.config import ROOT

logger = logging.getLogger("research")


def data_path(layer: str = "processed") -> dict[str, Any]:
    """
    Resolve <repo>/data/<layer> without hardcoding paths.

    Parameters
    ----------
    layer : 'raw' | 'processed'

    Returns
    -------
    dict with keys: path (Path), exists (bool)
    """
    path = ROOT / "data" / layer
    return {"path": path, "exists": path.exists()}


def save_processed(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """
    Save a DataFrame to data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: path (Path), rows (int), size_mb (float)
    """
    directory = data_path("processed")["path"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    df.to_parquet(path)
    return {
        "path": path,
        "rows": len(df),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def load_processed(name: str) -> dict[str, Any]:
    """
    Load data/processed/<name>.parquet.

    Returns
    -------
    dict with keys: df (DataFrame), path (Path), rows (int)
    """
    path = data_path("processed")["path"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed file: {path}")
    df = pd.read_parquet(path)
    return {"df": df, "path": path, "rows": len(df)}


# ── Frame inspection ───────────────────────────────────────────────────────────

def frame_summary(df: pd.DataFrame, n: int = 5) -> dict[str, Any]:
    """
    Structured snapshot of a DataFrame: shape, per-column diagnostics, index
    health, and head/tail samples.

    Parameters
    ----------
    df : DataFrame to inspect
    n  : rows to keep in the head/tail samples

    Returns
    -------
    dict with keys: shape (tuple), memory_mb (float), columns (DataFrame),
                    index (dict), head (DataFrame), tail (DataFrame)
    """
    columns = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        columns = columns.join(pd.DataFrame({
            "min": numeric.min(),
            "mean": numeric.mean(),
            "max": numeric.max(),
        }))

    index: dict[str, Any] = {
        "type": type(df.index).__name__,
        "monotonic": bool(df.index.is_monotonic_increasing),
        "duplicates": int(df.index.duplicated().sum()),
    }
    if len(df.index):
        index["start"] = df.index[0]
        index["end"] = df.index[-1]
    if isinstance(df.index, pd.DatetimeIndex):
        index["tz"] = str(df.index.tz)
        steps = df.index.to_series().diff().dropna()
        if not steps.empty:
            # modal step is the bar size; anything longer is a gap in the series
            step = steps.mode().iloc[0]
            index["step"] = step
            index["gaps"] = int((steps > step).sum())

    return {
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "columns": columns,
        "index": index,
        "head": df.head(n),
        "tail": df.tail(n),
    }


### data series from LSE
# api key
api_key = os.environ.get("LSE_API_KEY", "")
client = LSE(api_key=api_key) if (LSE is not None and api_key) else None


# load data series from LSE feed
def load_lse_data(client, asset: str, timeframe: str, limit):
    # load
    rows = client.candles(asset, timeframe, limit=limit, order="desc")
    # build df
    df = (pd.DataFrame(rows)
          .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"]))
          .set_index("timestamp")
          .sort_index())
    return df

# ── Common transforms ──────────────────────────────────────────────────────────
def log_transform(df, cols=None):
    if cols is None:
        cols = []
    for col in cols:
        df[f"{col}_log_return"] = np.log(df[col]).diff()
    return df

# lags of series
def add_lags(
    df: pd.DataFrame,
    cols: str | list[str],
    n_lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add `<col>_lag_<n>` columns for every (column, lag) pair.

    Returns a new frame; `df` is not modified.
    """
    if isinstance(cols, str):
        cols = [cols]
    n_lags = n_lags or []
    new_columns = {}
    for col_name in cols:
        for lag_value in n_lags:
            new_col_name = f"{col_name}_lag_{lag_value}"
            # bind loop vars as defaults so each lambda keeps its own pair
            new_columns[new_col_name] = (
                lambda df_inner, column=col_name, lag=lag_value: df_inner[column].shift(lag)
            )
    return df.assign(**new_columns)

# calculate sharpe ratio
def calculate_sharpe(
    df: pd.DataFrame,
    returns: str,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """
    Annualised Sharpe ratio for a column of per-period returns.

    Sharpe = mean(excess) / std(excess) * sqrt(periods_per_year), where excess
    subtracts the risk-free rate scaled down to the bar frequency.
    """
    series = df[returns].dropna()
    if series.empty:
        raise ValueError(f"Column '{returns}' has no non-null values")

    excess = series - (risk_free / periods_per_year)
    mean = float(excess.mean())
    std = float(excess.std())
    scale = np.sqrt(periods_per_year)

    return {
        "sharpe": float(mean / std * scale) if std > 0 else float("nan"),
        "mean_period": mean,
        "std_period": std,
        "ann_return": mean * periods_per_year,
        "ann_vol": std * scale,
        "n": int(series.size),
    }

# Overview of df
def df_report(df: pd.DataFrame) -> pd.DataFrame:
    print(df.info())
    print(df.index.min(), df.index.max())
    print(df.isnull().sum())
    print(df.describe())

# Resample price frequency
def resample(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    df = df.resample(horizon).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df
