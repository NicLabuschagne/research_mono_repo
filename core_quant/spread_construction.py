import pandas as pd
from pathlib import Path

"""
Phase 1, Step 1: resample 1-minute data into session-aligned bars.

Concept
-------
Each bar belongs to exactly one session. Within a session, a minute goes into
bar number floor(minutes since session open / bar size). Because bins restart
at every session open, no bar spans a contract switch.

Each bar is stamped with the timestamp of its LAST minute (bar_end): that is
when its values become known, which keeps later steps free of look-ahead.
"""


from core_quant.config import ROOT
PATH = ROOT / "data" / "processed"
BAR_SIZES_MINUTES = [15, 30, 60]
LEGS = ["ZS", "ZM", "ZL"]

# Nominal session opens in Chicago time
SESSION_OPEN_TIME = {"evening": pd.Timedelta(hours=19), "day": pd.Timedelta(hours=8, minutes=30)}


def resample_session_bars(minute_data: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute rows into bars that restart at every session open."""
    data = minute_data.copy()
    chicago_time = data.index.tz_convert("America/Chicago")

    # Anchor each session to its nominal open (19:00 or 08:30 CT on the session's first date).
    # Built from the local date + clock time, so daylight-saving changes cannot shift it by an hour.
    first_bar_time = pd.Series(chicago_time, index=data.index).groupby(data["session_id"]).transform("first")
    local_date = first_bar_time.dt.tz_localize(None).dt.normalize()
    session_anchor = (local_date + data["session_type"].map(SESSION_OPEN_TIME)).dt.tz_localize("America/Chicago")

    minutes_since_open = (pd.Series(chicago_time, index=data.index) - session_anchor).dt.total_seconds() / 60
    data["bar_number"] = (minutes_since_open // bar_minutes).astype(int)
    data["bar_end"] = data.index

    # How each column combines inside a bar
    aggregation = {
        "bar_end": "last",               # timestamp of the last minute in the bar
        "session_type": "first",
        "crush_level": "last",           # levels: take the value at the bar close
        "crush_chg": "sum",              # changes and returns: add up within the bar
        "ZS_adj_high": "max",
        "ZS_adj_low": "min",
        "crush_intraday": "last",
    }
    for leg in LEGS:
        aggregation[f"{leg}_adj_close"] = "last"
        aggregation[f"{leg}_ret"] = "sum"      # log returns add across time
        aggregation[f"{leg}_volume"] = "sum"
        aggregation[f"{leg}_segment"] = "last"

    bars = data.groupby(["session_id", "bar_number"]).agg(aggregation)
    bars["n_minutes"] = data.groupby(["session_id", "bar_number"]).size()
    bars = bars.reset_index()

    # Flag the last bar of each session (often a short stub, e.g. 13:00-13:20 CT)
    bars["is_last_in_session"] = bars["session_id"] != bars["session_id"].shift(-1)

    return bars.set_index("bar_end")


if __name__ == "__main__":
    minute_data = pd.read_parquet(PATH / "feature_df.parquet")

    bars_by_size = {}
    for bar_minutes in BAR_SIZES_MINUTES:
        bars = resample_session_bars(minute_data, bar_minutes)
        bars.to_parquet(PATH / f"bars_{bar_minutes}m.parquet")
        bars_by_size[bar_minutes] = bars
        print(f"{bar_minutes}m: {len(bars):,} bars, median minutes per bar {bars['n_minutes'].median():.0f}")

    # Sanity check: summed returns inside bars must equal the 1-minute total
    for leg in LEGS:
        total_minute = minute_data[f"{leg}_ret"].sum()
        total_bar = bars_by_size[60][f"{leg}_ret"].sum()
        print(f"{leg} return total, 1m vs 60m: {total_minute:.6f} vs {total_bar:.6f}")