import pandas as pd
import numpy as np
from pathlib import Path


# Clean data and organize into feature dataset
# Set Paths
from core_quant.config import ROOT
PATH = ROOT / "data" / "processed"
data = PATH / "crush_legs_processed.parquet"


# Board crush weights, $/bu: 44 lb meal = 0.022 short tons; 11 lb oil = 0.11 (ZL in cents/lb -> $);
# ZS is quoted in cents/bu, so divide by 100 to get $/bu.
MEAL_WEIGHT = 0.022
OIL_WEIGHT = 0.11
BEAN_DIVISOR = 100

full = pd.read_parquet(data)  # the processed file, for raw closes and session opens

# --- crush_chg: per-bar change in board crush ($/bu), switch-free ---
# Raw price differences are exact $ moves inside a session. At a session open
# the difference spans the gap (which may be a contract switch), so we set it to 0.
raw_crush = (MEAL_WEIGHT * full["ZM_close"]
             + OIL_WEIGHT * full["ZL_close"]
             - full["ZS_close"] / BEAN_DIVISOR)
crush_change = raw_crush.diff()
crush_change[full["session_open"]] = 0.0
full["crush_chg"] = crush_change.fillna(0.0)

# --- crush_level: board crush from adjusted closes ($/bu) ---
# Only meaningful over short windows: missed small switches make the long-run level drift.
full["crush_level"] = (MEAL_WEIGHT * full["ZM_adj_close"]
                     + OIL_WEIGHT * full["ZL_adj_close"]
                     - full["ZS_adj_close"] / BEAN_DIVISOR)

full["crush_intraday"] = full.groupby("session_id")["crush_chg"].cumsum()

cols_to_keep = ["crush_intraday", "crush_level", "crush_chg", "ZS_ret", "ZL_ret", "ZM_ret", "ZS_adj_close", "ZL_adj_close", "ZM_adj_close", "ZS_adj_high", "ZS_adj_low", "ZS_volume", "ZL_volume", "ZM_volume", "session_id", "session_type", "bar_in_session", "ZS_segment", "ZM_segment", "ZL_segment"]

ts = full[cols_to_keep]
print(ts.tail())

ts.to_parquet(PATH / "feature_df.parquet")

print(f"Cleaned df and saved {PATH / 'feature_df.parquet'}")
print(f"Length of df: {len(ts)} rows")