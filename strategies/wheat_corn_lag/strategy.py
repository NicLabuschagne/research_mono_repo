"""
strategy.py
-----------
Wheat-corn lag. When corn moves and wheat follows by less than usual, trade wheat against corn's move.

Spec (agreed before looking at any results)
  At each 15-minute close t, over a lookback L:
    beta  hedge ratio of wheat on corn, from past 15-minute returns only (up to t-1)
    m     = beta * corn return over L          the part of wheat's move corn explains
    e     = wheat return over L - m            wheat's own move
  Core test: forward wheat return = a*m + b*e.  a < 0: the complex reverts.  b > 0: wheat's lag carries information.
  F0 baseline: the trader's literal spreads 1.5W - C and W - 1.5C (in cents).

Steps 1-3 (this file so far): load the two legs, join them, cut off the holdout, roll-safe returns.

Timestamp convention (check this first when auditing for look-ahead)
  The vendor stamps each bar with its OPEN time. We re-label every bar by its END time,
  so the close in row t is a price that existed at time t, never later.
  Example: the vendor's "14:00" bar holds trades 14:00-14:15; here it is the row labelled 14:15.

Rolls and sessions (the second thing to check)
  Both files are the UNADJUSTED front month. On a roll day the price jumps by the calendar spread
  (corn N1->Z1 in June 2021: -131.5 c in one bar). The provider only switches contract between sessions:
  for corn, all 32 switches 2020-2025 fall between one session's close and the next session's open
  (checked against exchange contract data in data_checks.py).
  So a roll can only sit at a session boundary, and only at a boundary near a roll date:
    corn   the exact switch sessions are known from Databento contract-level bars
    wheat  no contract data. Corn's switches fall 10 sessions before to 3 after first notice day (FND),
           so every boundary from FND-10 to FND+3 business days of each wheat contract (H K N U Z) is
           treated as a possible roll. These windows come from the contract calendar, known years in advance.
  Rule: the series is cut into SEGMENTS at every possible-roll boundary and stitched everywhere else.
  Returns, lookbacks and forward windows never cross a segment edge, so a roll cannot enter a feature or
  a target, while windows can still run across days (about 72% of session boundaries are kept).
  A CBOT session (19:00 -> 13:20 CT) always falls inside one UTC date here, so session = UTC date.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT, GateConfig, SplitConfig, ValidationConfig, load_config

STRATEGY_DIR = Path(__file__).parent


@dataclass(frozen=True)
class StrategyConfig:
    """The v2 rule (backtest.py). Every grid name in config.yaml must be a field here."""
    lookback: int = 32                     # L, bars: the window the divergence is measured over
    scaling: str = "empirical"             # "empirical": own_ze / corn_ze (actual L-bar sd); "sqrt": own_z / corn_z
    corn_sd: float = 1.0                   # |corn_z_L| needed, same direction as wheat's own move; 0 = no corn condition
    own_sd: float = 1.0                    # |own_z_L| needed
    target_share: float = 0.5              # take profit after this many units in our favour
    stop_share: float = 2.0                # safety stop after this many units against us
    level_unit: str = "divergence"         # unit for target and stop: "divergence" = D, how far wheat overshot
                                           # corn; "atr" = wheat's typical move over the lookback, sigma_w x sqrt(L)
    max_hold_bars: int = 71                # time stop
    instrument: str = "outright"           # "outright" = wheat only; "spread" = wheat hedged with beta x corn
    reentry: bool = False                  # after a TARGET exit, re-enter at once while the condition still holds
                                           # (tested 2026-10-30: 31 trades, -13 bp; out of the grid)
    entry_delay_bars: int = 1              # fill at the OPEN of bar t + this; 1 = the bar right after the signal
    cost_ticks: float = 1.4                # round trip, a stress: keep it out of the grid


@dataclass(frozen=True)
class KalmanRuleConfig:
    """The adopted base (2026-11-02, kalman_rule.py): wheat outright on the Kalman level gap, RMS scale."""
    memory_sessions: int                   # which Kalman speed (features kzr_{M}s)
    tail: float                            # trade the top / bottom `tail` share of the signal (past-year quantiles)
    per_session: bool                      # also re-arm at every new session (False: once per excursion)
    rearm_bars: int                        # also re-arm this many bars after the last trigger while s stays in
                                           # the tail (0 = off)
    max_open: int                          # positions open at once
    target_gap: float                      # take profit at this multiple of the Kalman gap at entry (0 = off)
    stop_gap: float                        # safety stop at this multiple of the Kalman gap at entry (0 = off)
    min_gap_cents: float                   # skip a trigger whose Kalman gap at entry is smaller, cents (0 = off)
    hold_bars: int                         # time exit: 71 = one session
    quantile_window_bars: int              # thresholds from this many past bars of the signal
    quantile_min_bars: int                 # no thresholds before this many
    cost_ticks: float                      # round trip, a stress: keep it out of the grid


@dataclass(frozen=True)
class FeatureConfig:
    lookbacks_bars: list[int]              # L, in 15-minute bars
    horizons_bars: list[int]               # h, in 15-minute bars
    stats_window_bars: int                 # rolling window for sigmas and beta
    kalman_beta_sd_per_session: float      # how far the Kalman beta may drift in one session
    kalman_warmup_bars: int                # bars used to start the Kalman filter (no output there)
    f0_ratio: float                        # the trader's 1.5
    f0_flat_sd: float                      # a spread is "flat" when its move is within this many sd
    level_windows_bars: list[int]          # model C: regression windows of the level fair value
    factor_sets: dict[str, list[str]]      # multi-factor fair values: name -> factor returns used, e.g. ["c", "s"]
    kalman_memory_sessions: list[int]      # model K: Kalman level fair value speeds (memory in sessions)


@dataclass(frozen=True)
class Config:
    wheat_file: str
    corn_file: str
    corn_contracts_file: str
    soy_file: str                          # data/processed: per-bar soybean returns (ZS_ret, roll-free) from the crush build
    usd_file: str                          # data/raw: EURUSD 15m (vendor), used as an inverted USD proxy
    bar_minutes: int
    roll_business_days_before_fnd: int
    roll_business_days_after_fnd: int
    split: SplitConfig
    features: FeatureConfig
    strategy: StrategyConfig
    kalman_rule: KalmanRuleConfig
    kalman_grid: dict[str, list]       # walk-forward grid for kalman_rule (validation.grid is model A's)
    validation: ValidationConfig
    gates: GateConfig


def load(path: Path = STRATEGY_DIR / "config.yaml") -> Config:
    return load_config(path, Config)


def load_leg(file_name: str, bar_minutes: int) -> pd.DataFrame:
    """One leg's bars: close in cents/bu and volume, indexed by bar END time (UTC)."""
    frame = pd.read_parquet(ROOT / "data" / "raw" / file_name)
    bar_open = pd.to_datetime(frame["time"], utc=True)
    frame.index = pd.DatetimeIndex(bar_open + pd.Timedelta(minutes=bar_minutes), name="bar_end")

    # trust boundary: the file comes from a vendor, so refuse anything that cannot be a price
    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or prices.isna().any().any():
        raise ValueError(f"{file_name}: non-positive or missing prices")
    duplicates = frame.index.duplicated(keep="last")
    if duplicates.any():
        print(f"{file_name}: dropping {duplicates.sum()} duplicate timestamps (keeping the last)")
        frame = frame[~duplicates]

    frame = frame.sort_index()
    return frame[["open", "high", "low", "close", "volume"]].copy()                  # already cents/bu


def corn_contract_switches(cfg: Config) -> pd.DataFrame:
    """Every time the corn file changes contract, found by matching it hour by hour to Databento ZC contracts.

    Returns one row per switch: the last hour still on the old contract and the first hour on the new one.
    Uses prices only to tell contracts apart; nothing here feeds a feature.
    """
    import databento

    hourly = databento.DBNStore.from_file(ROOT / "data" / "raw" / cfg.corn_contracts_file).to_df(map_symbols=True)
    hourly = hourly[hourly["symbol"].str.fullmatch(r"ZC[HKNUZ]\d")]              # outrights only, no spreads
    hourly.index = hourly.index + pd.Timedelta(hours=1)                         # Databento stamps the open: label by end
    contracts = hourly.pivot_table(index=hourly.index, columns="symbol", values="close", aggfunc="last")

    corn = load_leg(cfg.corn_file, cfg.bar_minutes)["close"]
    corn = corn[corn.index.minute == 0]
    distance = contracts.reindex(corn.index).sub(corn, axis=0).abs().dropna(how="all")
    sorted_distance = np.sort(distance.values, axis=1)
    # an hour counts only when one contract is within 0.5 c and the next nearest is at least 1.5 c further away
    # (at 1 c, Z1 and H2 sat a tick apart on 2021-11-30 and one hour matched the wrong contract)
    clear = (sorted_distance[:, 0] <= 0.5) & (sorted_distance[:, 1] - sorted_distance[:, 0] >= 1.5)
    tracked = distance.idxmin(axis=1)[clear]

    rows = []
    for time in tracked.index[tracked != tracked.shift()][1:]:
        # 5 clear hours in a row on the new contract: a one-hour coincidence is not a switch, but a
        # one-session flip (Dec -> Sep -> Dec, 2020-07-24) is, so the bar cannot be much higher than 5
        if not (tracked[tracked.index >= time].iloc[:5] == tracked[time]).all():
            continue
        before = tracked[tracked.index < time]
        rows.append({"last_on_old": before.index[-1], "first_on_new": time, "old": before.iloc[-1], "new": tracked[time]})
    return pd.DataFrame(rows)


def wheat_first_notice_days(first_year: int, last_year: int) -> pd.DatetimeIndex:
    """FND of every wheat contract (H K N U Z): the last business day of Feb, Apr, Jun, Aug and Nov."""
    days = [pd.Timestamp(year, delivery_month, 1) - pd.offsets.BDay(1)
            for year in range(first_year, last_year + 1) for delivery_month in [3, 5, 7, 9, 12]]
    return pd.DatetimeIndex(days, tz="UTC")


def possible_roll_sessions(sessions: pd.DatetimeIndex, cfg: Config) -> pd.DatetimeIndex:
    """Sessions whose boundary with the previous session may hide a roll on either leg."""
    flagged = set()
    # wheat: every session from FND-before to FND+after business days. Pure calendar, so the window is the
    # same whether the data ends today or in ten years (counting the data's own sessions would not be).
    for fnd in wheat_first_notice_days(sessions[0].year, sessions[-1].year + 1):
        window_start = fnd - pd.offsets.BDay(cfg.roll_business_days_before_fnd)
        window_end = fnd + pd.offsets.BDay(cfg.roll_business_days_after_fnd)
        flagged.update(sessions[(sessions >= window_start) & (sessions <= window_end)])
    # corn: the exact switch, i.e. the first session after the last hour on the old contract
    for last_on_old in corn_contract_switches(cfg)["last_on_old"]:
        following = sessions.searchsorted(last_on_old.normalize(), side="right")
        if following < len(sessions):
            flagged.add(sessions[following])
    return pd.DatetimeIndex(sorted(flagged))


def load_bars(cfg: Config, include_holdout: bool = False, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Wheat (W) and corn (C) closes in cents, only bars where BOTH traded.

    No forward-filling: a stale price on one leg would look like a divergence that never happened.
    include_holdout=False (the default) drops every bar ending on or after holdout_start.
    end: drop every bar after this time BEFORE anything is computed (leak_checks.py: "as if today were end").
    """
    wheat = load_leg(cfg.wheat_file, cfg.bar_minutes)
    corn = load_leg(cfg.corn_file, cfg.bar_minutes)
    # opens are only used to FILL an entry (signal at a bar's close, fill at the next bar's open);
    # wheat highs and lows only to check whether an exit level was touched inside a bar (exits.py)
    bars = pd.DataFrame({"W": wheat["close"], "C": corn["close"], "W_open": wheat["open"], "C_open": corn["open"],
                         "W_high": wheat["high"], "W_low": wheat["low"],
                         "W_volume": wheat["volume"], "C_volume": corn["volume"]}).dropna()

    bars = bars[bars.index >= pd.Timestamp(cfg.split.development_start, tz="UTC")]
    if not include_holdout:
        bars = bars[bars.index < pd.Timestamp(cfg.split.holdout_start, tz="UTC")]
    if end is not None:
        bars = bars[bars.index <= end]
    bars["log_w"] = np.log(bars["W"])
    bars["log_c"] = np.log(bars["C"])

    # segments: a new one starts at the first bar of every session that may follow a roll
    bars["session"] = bars.index.normalize()
    possible_roll = possible_roll_sessions(pd.DatetimeIndex(bars["session"].unique()), cfg)
    starts_session = bars["session"] != bars["session"].shift()
    bars["segment"] = (starts_session & bars["session"].isin(possible_roll)).cumsum()

    # log returns inside each segment only: the return across a possible-roll boundary is NaN
    bars["ret_w"] = bars.groupby("segment")["log_w"].diff()
    bars["ret_c"] = bars.groupby("segment")["log_c"].diff()

    # extra fair-value factors, as log "indices" sampled at OUR bar ends, then differenced inside our segments,
    # so each factor return covers exactly the same interval as wheat's. A missing value stays missing.
    #   s = soybeans: running total of the crush build's per-bar ZS returns (roll-free; its adjusted CLOSE is not:
    #       it jumps 11% at the 2022-06-22 roll). Its bars are labelled at their last minute (00:14 -> our 00:15).
    #   u = USD strength: minus log EURUSD (vendor bars stamped at their open -> + bar_minutes).
    soy = pd.read_parquet(ROOT / "data" / "processed" / cfg.soy_file)["ZS_ret"]
    soy_index = soy.cumsum()
    soy_index.index = soy_index.index + pd.Timedelta(minutes=1)
    usd = pd.read_csv(ROOT / "data" / "raw" / cfg.usd_file)
    usd_index = -np.log(usd["close"].values)
    usd_index = pd.Series(usd_index, index=pd.to_datetime(usd["timestamp"], utc=True) + pd.Timedelta(minutes=cfg.bar_minutes))
    usd_index = usd_index[~usd_index.index.duplicated(keep="last")]
    bars["index_s"] = soy_index.reindex(bars.index)
    bars["index_u"] = usd_index.reindex(bars.index)
    bars["ret_s"] = bars.groupby("segment")["index_s"].diff()
    bars["ret_u"] = bars.groupby("segment")["index_u"].diff()
    return bars
