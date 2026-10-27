"""
features.py
-----------
z-score of the Kalman residual and the evening-session signal.

z-score (per session type)
  z = (residual - mean) / std, with mean and std taken from the previous 20 trading
  days of the same session type. Today is excluded, so nothing leaks into today's z.
  Per session because the day session's residual moves ~1.8x more per bar; a pooled
  z would flag mostly day-session events. 20 days was chosen on calibration
  (out-of-sample z std closest to 1), not on trading results.

Signal
  First bar where |z| >= threshold. The detector then disarms until |z| falls back
  below 0.5, so one dislocation counts once. Direction is against the dislocation:
  z > 0 (beans rich vs products) -> short beans / long products, and vice versa.
  Only evening-session signals are kept (the day session showed no edge), and the
  first bar of a session is skipped (the hedge has just switched there).
"""
import numpy as np
import pandas as pd

Z_WINDOW_DAYS = 20
REARM_LEVEL = 0.5


def add_session_zscore(bars: pd.DataFrame, window_days: int = Z_WINDOW_DAYS) -> pd.DataFrame:
    """Add z_mean, z_std and z, each from the previous `window_days` days of the same session type."""
    bars = bars.copy()
    bars["z_mean"], bars["z_std"] = np.nan, np.nan
    for session_type in bars["session_type"].unique():
        in_session = bars["session_type"] == session_type
        residual = bars.loc[in_session, "residual"]
        daily = pd.DataFrame({"total": residual, "total_sq": residual ** 2, "count": 1,
                              "date": bars.loc[in_session, "trading_date"]}).groupby("date").sum()
        window = daily.rolling(window_days).sum().shift(1)          # previous days only
        mean = window["total"] / window["count"]
        std = np.sqrt(window["total_sq"] / window["count"] - mean ** 2)
        dates = bars.loc[in_session, "trading_date"]
        bars.loc[in_session, "z_mean"] = dates.map(mean).values
        bars.loc[in_session, "z_std"] = dates.map(std).values
    bars["z"] = (bars["residual"] - bars["z_mean"]) / bars["z_std"]
    return bars


def detect_signals(bars: pd.DataFrame, threshold: float, session: str = "evening") -> pd.DataFrame:
    """First crossing of |z| >= threshold, re-armed below REARM_LEVEL.

    The arm/disarm state runs through every bar (both sessions), then we keep the
    chosen session and drop signals on a session's first bar."""
    rows, armed = [], True
    for position, z_value in enumerate(bars["z"].values):
        if np.isnan(z_value):
            continue
        if armed and abs(z_value) >= threshold:
            rows.append(dict(signal_position=position, entry_z=z_value, direction=-np.sign(z_value)))
            armed = False
        elif not armed and abs(z_value) < REARM_LEVEL:
            armed = True
    signals = pd.DataFrame(rows)
    keep = ((bars["session_type"].values[signals["signal_position"]] == session)
            & (bars["bar_number"].values[signals["signal_position"]] != 0))
    signals = signals[keep].reset_index(drop=True)
    signals["signal_time"] = bars.index[signals["signal_position"]]
    return signals
