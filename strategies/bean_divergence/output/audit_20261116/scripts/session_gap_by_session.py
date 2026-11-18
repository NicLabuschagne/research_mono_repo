"""The pull toward the session fair value, split by session (development data only). Every 5-minute bar; beans move
TOWARD fair value from the next open, inside the same session, ticks before costs; mean by day (t).
Two ways to call a gap large: fixed ticks, and each session's OWN distribution (its largest 10% and 5% of gaps),
because 20 ticks is common in the US session and rare in Asia. Descriptive: no pass mark."""
import contextlib, io, runpy
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("session_gap_by_session.py", "session_gap20.py"))
gap, size, ok, session, half, dates, session_last = (g[k] for k in ["gap", "size", "ok", "session", "half", "dates", "session_last"])
open_s, close_s, TICK, n = g["open_s"], g["close_s"], g["TICK_CENTS"], g["n"]
t_index = np.arange(n)
pd.set_option("display.width", 250)


def cell(chosen: np.ndarray, bars_ahead: int) -> str:
    stop = np.minimum(t_index + bars_ahead, session_last)
    move = -np.sign(gap) * (close_s[stop] - open_s[np.minimum(t_index + 1, n - 1)]) / TICK
    by_day = pd.Series(move[chosen]).groupby(dates[chosen]).mean()
    if len(by_day) < 30:
        return f"n={len(by_day)} days"
    return f"{by_day.mean():+.1f} ({by_day.mean() / (by_day.std() / np.sqrt(len(by_day))):+.1f})"


rows = []
for name in ["asia", "europe", "us"]:
    in_session = ok & (session == name) & (t_index + 1 <= session_last)
    top10, top5 = np.percentile(size[in_session], 90), np.percentile(size[in_session], 95)
    groups = [("10 to 20 ticks", (size >= 10) & (size < 20)), ("20 to 30", (size >= 20) & (size < 30)), ("30 to 40", (size >= 30) & (size < 40)),
              ("40 or more", size >= 40), (f"session's largest 10% (over {top10:.0f})", size >= top10), (f"session's largest 5% (over {top5:.0f})", size >= top5)]
    for label, mask in groups:
        chosen = in_session & mask
        hours_left = (session_last[chosen] - t_index[chosen]).mean() / 12
        stop = np.minimum(t_index + 48, session_last)
        move = -np.sign(gap) * (close_s[stop] - open_s[np.minimum(t_index + 1, n - 1)]) / TICK
        rows.append({"session": name, "gap": label, "share of session's bars": f"{chosen.sum() / in_session.sum():.1%}", "average gap": round(size[chosen].mean(), 1),
                     "hours left in session": round(hours_left, 1), "1 hour": cell(chosen, 12), "4 hours": cell(chosen, 48), "session end": cell(chosen, 10 ** 6),
                     "our way, 4 hours": f"{(move[chosen] > 0).mean():.0%}",
                     "4 hours 2021-22": cell(chosen & (half == "2021-22"), 48), "4 hours 2023-24": cell(chosen & (half == "2023-Nov24"), 48)})
print(pd.DataFrame(rows).to_string(index=False))
