"""
data_ingest.py
--------------
Turn raw 1-minute ZS / ZM / ZL data into clean, session-aligned bars.

Steps
  1. Load the 1-minute OHLCV file (one stitched front-month series per leg).
  2. Tag sessions: a gap longer than 30 minutes starts a new session
     (evening opens 19:00 CT, day opens 08:30 CT).
  3. Switch-free returns: inside a session, close-to-close log returns.
     On a session's first bar we use close/open of that bar, so the opening
     gap (where contract switches happen) is never inside a return.
  4. Detect contract switches: a switch moves ONE leg at a session open while
     the other two legs agree. We flag large single-leg gaps.
  5. Back-adjust prices: remove flagged switch gaps by scaling all earlier prices,
     so price levels are continuous across switches.
  6. Resample to N-minute bars that restart at every session open, so no bar
     ever spans a session break.

Concept: contract switches only happen at session opens, so returns measured
inside a session are switch-free by construction. Price LEVELS need the
back-adjustment, which is approximate (small switches can slip through).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT

LEGS = ["ZS", "ZM", "ZL"]
SESSION_BREAK_MINUTES = 30
ROLL_Z_THRESHOLD = 8          # a leg's opening gap must be an 8-sigma outlier vs the other legs
OTHERS_AGREE_Z = 4            # ...while the other two legs agree with each other
SESSION_OPEN_TIME = {"evening": pd.Timedelta(hours=19), "day": pd.Timedelta(hours=8, minutes=30)}


def load_minute_data(path: Path) -> pd.DataFrame:
    """Keep only OHLCV columns for the three legs (index: UTC timestamps)."""
    columns = [f"{leg}_{field}" for leg in LEGS for field in ["open", "high", "low", "close", "volume"]]
    return pd.read_parquet(path, columns=columns)


def add_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """Session id, session type (evening/day) and the minutes gap before each row."""
    data = data.copy()
    minutes_since_previous = data.index.to_series().diff().dt.total_seconds() / 60
    data["session_open"] = (minutes_since_previous > SESSION_BREAK_MINUTES).fillna(True).values
    data["session_id"] = data["session_open"].cumsum()
    chicago_hour = data.index.tz_convert("America/Chicago").hour
    data["session_type"] = np.where(chicago_hour >= 17, "evening", "day")
    # evening session continues past midnight, so take the type at the session's first bar
    data["session_type"] = data.groupby("session_id")["session_type"].transform("first")
    return data


def add_switch_free_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Per-leg 1-minute log return that never crosses a session open, plus the opening gap."""
    data = data.copy()
    for leg in LEGS:
        close, open_ = data[f"{leg}_close"], data[f"{leg}_open"]
        inside_session = np.log(close / close.shift(1))
        data[f"{leg}_ret"] = np.where(data["session_open"], np.log(close / open_), inside_session)
        data[f"{leg}_gap_ret"] = np.where(data["session_open"], np.log(open_ / close.shift(1)), np.nan)
        data.iloc[0, data.columns.get_loc(f"{leg}_gap_ret")] = np.nan
    return data


def robust_residual_z(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    """Regress target on predictors, refit without the worst 5% of residuals, return residual / MAD scale."""
    design = np.column_stack([predictors, np.ones(len(predictors))])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ beta
    keep = np.abs(residual) < np.quantile(np.abs(residual), 0.95)
    beta = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
    residual = target - design @ beta
    return residual / (np.median(np.abs(residual)) * 1.4826)


def flag_contract_switches(data: pd.DataFrame) -> pd.DataFrame:
    """Flag session opens where one leg gaps alone (a contract switch) while the other two agree."""
    data = data.copy()
    gaps = data.loc[data["session_open"], [f"{leg}_gap_ret" for leg in LEGS]].dropna()
    for leg in LEGS:
        others = [other for other in LEGS if other != leg]
        leg_z = robust_residual_z(gaps[f"{leg}_gap_ret"].values, gaps[[f"{o}_gap_ret" for o in others]].values)
        others_z = robust_residual_z(gaps[f"{others[0]}_gap_ret"].values, gaps[[f"{others[1]}_gap_ret"]].values)
        is_switch = (np.abs(leg_z) > ROLL_Z_THRESHOLD) & (np.abs(others_z) < OTHERS_AGREE_Z)
        data[f"{leg}_switch"] = pd.Series(is_switch, index=gaps.index).reindex(data.index).fillna(False).astype(bool)
    return data


def back_adjust(data: pd.DataFrame) -> pd.DataFrame:
    """Ratio back-adjustment anchored to the last bar: scale every price before a switch by exp(switch gap)."""
    data = data.copy()
    for leg in LEGS:
        switch_gap = data[f"{leg}_gap_ret"].where(data[f"{leg}_switch"], 0.0).fillna(0.0)
        gaps_after_each_bar = switch_gap[::-1].cumsum()[::-1] - switch_gap
        factor = np.exp(gaps_after_each_bar)
        for field in ["open", "close"]:
            data[f"{leg}_adj_{field}"] = data[f"{leg}_{field}"] * factor
    return data


def resample_to_bars(data: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Session-aligned bars: bar number = minutes since the session's nominal open // bar size.
    Each bar is stamped with its LAST minute, the moment its values become known."""
    chicago = pd.Series(data.index.tz_convert("America/Chicago"), index=data.index)
    first_minute = chicago.groupby(data["session_id"]).transform("first")
    # anchor built from local date + clock time so daylight-saving changes cannot shift it
    anchor = (first_minute.dt.tz_localize(None).dt.normalize()
              + data["session_type"].map(SESSION_OPEN_TIME)).dt.tz_localize("America/Chicago")
    data = data.assign(bar_number=((chicago - anchor).dt.total_seconds() / 60 // bar_minutes).astype(int),
                       bar_end=data.index)

    rules = {"bar_end": "last", "session_type": "first"}
    for leg in LEGS:
        rules[f"{leg}_adj_open"] = "first"
        rules[f"{leg}_adj_close"] = "last"
        rules[f"{leg}_ret"] = "sum"            # log returns add up across time
        rules[f"{leg}_volume"] = "sum"
    bars = data.groupby(["session_id", "bar_number"]).agg(rules).reset_index().set_index("bar_end")
    return add_trading_date(bars)


def add_trading_date(bars: pd.DataFrame) -> pd.DataFrame:
    """CBOT trading date: the evening session (from 19:00 CT) belongs to the next day's date."""
    chicago = bars.index.tz_convert("America/Chicago")
    bars = bars.copy()
    bars["trading_date"] = (chicago + pd.Timedelta(hours=5)).tz_localize(None).normalize()
    return bars


def build_bars(raw_path: Path, bar_minutes: int = 5) -> pd.DataFrame:
    """Full ingest: raw 1-minute file -> clean session-aligned bars."""
    data = load_minute_data(raw_path)
    data = add_sessions(data)
    data = add_switch_free_returns(data)
    data = flag_contract_switches(data)
    data = back_adjust(data)
    return resample_to_bars(data, bar_minutes)


if __name__ == "__main__":
    bars = build_bars(ROOT / "data" / "raw" / "latest_df_backup.parquet", bar_minutes=5)
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    bars.to_parquet(output / "bars_5m.parquet")
    print(f"{len(bars):,} bars from {bars.index[0]} to {bars.index[-1]}")
