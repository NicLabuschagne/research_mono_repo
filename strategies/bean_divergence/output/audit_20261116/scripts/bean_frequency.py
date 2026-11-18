"""How many alerts a week do looser levels give, and what do they cost in quality? (development data)
Levers: how often the live gap counts as stretched (4% / 6% / 8% of bars) and how far the session gap must be
stretched the same way (not required / 0.5 / 0.75 / 1.0 x the expected 1-hour move). The turn condition is unchanged.
Quality: beans move in the fade direction from the next open, ticks BEFORE costs; t from daily sums."""
import contextlib, io, runpy
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("bean_frequency.py", "bean_limit.py"))
bars, live_gaps, yardstick, in_dev = g["bars"], g["live_gaps"], g["yardstick"], g["in_dev"]
open_s, close_s, session_id, dates, half, weeks, TICK = (g[k] for k in ["open_s", "close_s", "session_id", "dates", "half", "weeks", "TICK_CENTS"])
turn_signals = g["turn_signals"]


def quality(events) -> dict:
    rows = []
    for t, rich in events:
        entry = t + 1
        if not in_dev[t] or entry >= len(bars) or session_id[entry] != session_id[t]:
            continue
        end = entry
        while end + 1 < len(bars) and session_id[end + 1] == session_id[entry] and end < entry + 23:
            end += 1
        path = -rich * (close_s[entry:end + 1] - open_s[entry]) / TICK
        rows.append({"date": dates[t], "half": half[t], "30m": path[min(5, len(path) - 1)], "60m": path[min(11, len(path) - 1)],
                     "120m": path[-1], "best": max(path[:12].max(), 0), "worst": min(path[:12].min(), 0)})
    frame = pd.DataFrame(rows)

    def mean_t(part, column):
        daily = part.groupby("date")[column].sum()
        return f"{part[column].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"
    by_half = frame.groupby("half")["60m"].mean()
    return {"a week": round(len(frame) / weeks, 1), "ahead 30m": mean_t(frame, "30m"), "ahead 60m": mean_t(frame, "60m"),
            "ahead 120m": mean_t(frame, "120m"), "best / worst 60m": f"{frame['best'].median():.1f} / {frame['worst'].median():.1f}",
            "moved our way at 60m": round((frame["60m"] > 0).mean(), 2),
            "60m 2021-22 / 2023-24": f"{by_half['2021-22']:+.2f} / {by_half['2023-Nov24']:+.2f}"}


pd.set_option("display.width", 250)
for drift, label in [(0.1, "live speed as now"), (0.01, "live 10x slower")]:
    gap = live_gaps[drift]
    rows = []
    for share in [0.04, 0.06, 0.08]:
        level = float((gap.abs() / yardstick)[in_dev].quantile(1 - share))
        for agree in [None, 0.5, 0.75, 1.0]:
            turn_signals.__globals__["SESSION_AGREE"] = -1e9 if agree is None else agree
            events = turn_signals(gap, level)["both turn"]
            rows.append({"live stretched": f"{share:.0%} of bars", "session agrees": "not required" if agree is None else f">= {agree}",
                         **quality(events)})
    print(f"\n== {label}")
    print(pd.DataFrame(rows).to_string(index=False))
