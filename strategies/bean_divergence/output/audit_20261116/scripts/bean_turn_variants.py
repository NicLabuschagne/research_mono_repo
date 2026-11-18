"""Which 'turn' is needed after a BOTH stretch (live and session gap stretched the same way)? Same data, entry and
outcome columns as bean_signal_stats.py; only the confirmation changes. Wait at most 30 minutes, as before."""
import contextlib, io, runpy
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):                      # reuse its data and functions, not its printout
    g = runpy.run_path(__file__.replace("bean_turn_variants.py", "bean_signal_stats.py"))
live, sess, fair_move, session_id = g["live_gap"].values, g["session_gap"].values, g["fair_move"], g["session_id"]
outcomes, mean_t, weeks, both = g["outcomes"], g["mean_t"], g["weeks"], g["both"]

CHECKS = {
    "no wait (both stretch)": lambda bar, side: True,
    "live gap stopped widening": lambda bar, side: abs(live[bar]) <= abs(live[bar - 1]),
    "session gap stopped widening": lambda bar, side: abs(sess[bar]) <= abs(sess[bar - 1]),
    "either gap stopped widening": lambda bar, side: abs(live[bar]) <= abs(live[bar - 1]) or abs(sess[bar]) <= abs(sess[bar - 1]),
    "fair value flat or turned (15 min)": lambda bar, side: side * fair_move[bar] <= 0,
    "gap stopped OR fair value turned": lambda bar, side: abs(live[bar]) <= abs(live[bar - 1]) or side * fair_move[bar] <= 0,
    "live gap stopped AND fair value turned (drawn)": lambda bar, side: abs(live[bar]) <= abs(live[bar - 1]) and side * fair_move[bar] <= 0,
}
rows = []
for name, check in CHECKS.items():
    events = []
    for t, side in both:
        for bar in range(t, min(t + 7, len(live))):
            if session_id[bar] != session_id[t] or side * live[bar] < 5:
                break
            if check(bar, side):
                events.append((bar, side))
                break
    frame = outcomes(events)
    row = {"confirmation": name, "a week": round(len(frame) / weeks, 1)}
    row.update({c: mean_t(frame, c) for c in ["ahead 30m", "ahead 60m", "exit session"]})
    row["best / worst 60m"] = f"{frame['best 60m'].median():.1f} / {frame['worst 60m'].median():.1f}"
    row["win rate (exit session)"] = round((frame["exit session"] > 0).mean(), 2)
    for key, part in frame.groupby("half"):
        row[f"exit session {key}"] = mean_t(part, "exit session")
    rows.append(row)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).set_index("confirmation").T.to_string())
