"""Sensitivity only: the session-anchored spread rule (spread_session.py) at other z levels. Re-arm stays at 0.41 x the level.
Not a search for the best level: the rule is sound only if neighbouring levels give a similar answer."""
import contextlib, io, runpy
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_session_levels.py", "spread_session.py"))
bars, in_dev, half, weeks, find_triggers, session = (g[k] for k in ["bars", "in_dev", "half", "weeks", "find_triggers", "session"])
gap, z, session_id, dates, log_open, log_close = (g[k] for k in ["gap", "z", "session_id", "dates", "log_open", "log_close"])
beta_meal, beta_oil, price_ticks = g["beta_meal"], g["beta_oil"], g["price_ticks"]
pd.set_option("display.width", 250)
rows = []
for level in [1.25, 1.5, 1.645, 1.8, 2.0]:
    trades, busy_until = [], 0
    for t, rich in find_triggers(z, session_id, level, 0.41 * level):
        entry = t + 1
        if not in_dev[t] or entry < busy_until or entry >= len(bars) or session_id[entry] != session_id[t]:
            continue
        held = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]
        last = entry
        while rich * gap[last] > 0 and last + 1 < len(bars) and session_id[last + 1] == session_id[entry]:
            last += 1
        reached = rich * gap[last] <= 0 and last + 1 < len(bars) and session_id[last + 1] == session_id[entry]
        exit_level = held(log_open, last + 1) if reached else held(log_close, last)
        busy_until = last + 1
        trades.append({"date": dates[t], "half": half[t], "overnight": session[t] != "us", "stretch": abs(gap[t]),
                       "net": -rich * (exit_level - held(log_open, entry)) * price_ticks[t] - 1.1})
    trades = pd.DataFrame(trades)
    def mean_t(part):
        daily = part.groupby("date")["net"].sum()
        return f"{part['net'].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"
    by_half = trades.groupby("half")["net"].mean()
    night = trades[trades["overnight"]]
    night_half = night.groupby("half")["net"].mean()
    rows.append({"z level": level, "bars outside": f"{(np.abs(z[in_dev]) >= level).mean():.0%}", "trades a week": round(len(trades) / weeks, 1),
                 "stretch at entry": round(trades["stretch"].mean(), 1), "net at 1.1 (t)": mean_t(trades), "won": f"{(trades['net'] > 0).mean():.0%}",
                 "winner / loser": f"{trades.loc[trades.net > 0, 'net'].mean():+.1f} / {trades.loc[trades.net <= 0, 'net'].mean():+.1f}",
                 "2021-22 / 2023-24": f"{by_half['2021-22']:+.2f} / {by_half['2023-Nov24']:+.2f}",
                 "overnight only: a week": round(len(night) / weeks, 1), "overnight net (t)": mean_t(night),
                 "overnight halves": f"{night_half['2021-22']:+.2f} / {night_half['2023-Nov24']:+.2f}"})
print(pd.DataFrame(rows).to_string(index=False))
