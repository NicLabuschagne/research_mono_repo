"""Two checks (development data only).
A. Are beans, meal and oil cointegrated? Engle-Granger test on log prices with FIXED ratios, daily closes: the whole
   sample, each year, and back-to-back 60-day windows. p < 0.05 = cointegrated.
B. When does the session-anchored signal stop working? For every session trigger (|z| >= 1.645, as traded), the
   spread's move in the fade direction from the next open, with the signal bar's hedge held, at 30 minutes to 3
   trading days - straight THROUGH the session close. Plus the rank correlation (IC) of the gap with the forward move
   over ALL bars.
   Decision rule, fixed before the run: the session-close exit is too early only if the move keeps rising after
   the session end by more than one standard error; otherwise it stays."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_decay.py", "spread_session.py"))
bars, in_dev, half, session, find_triggers, out_dir = (g[k] for k in ["bars", "in_dev", "half", "session", "find_triggers", "out_dir"])
gap, z, session_id, log_open, log_close = (g[k] for k in ["gap", "z", "session_id", "log_open", "log_close"])
beta_meal, beta_oil, price_ticks = g["beta_meal"], g["beta_oil"], g["price_ticks"]
pd.set_option("display.width", 250)

# ---------------------------------------------------------------- A: cointegration
daily = np.log(bars.loc[in_dev, ["ZS_adj_close", "ZM_adj_close", "ZL_adj_close"]].groupby(bars.loc[in_dev, "trading_date"]).last())
def p_value(part: pd.DataFrame) -> float:
    return coint(part["ZS_adj_close"], part[["ZM_adj_close", "ZL_adj_close"]])[1]
print("== A. Engle-Granger cointegration of beans with meal and oil, fixed ratios, daily closes (p < 0.05 = cointegrated)")
print(f"whole development sample ({len(daily)} days): p = {p_value(daily):.2f}")
print("by year: " + ", ".join(f"{year} p = {p_value(part):.2f}" for year, part in daily.groupby(daily.index.year)))
windows = [p_value(daily.iloc[start:start + 60]) for start in range(0, len(daily) - 59, 60)]
print(f"60-day windows: {sum(p < 0.05 for p in windows)} of {len(windows)} cointegrated (chance alone gives about {0.05 * len(windows):.0f})")

# ---------------------------------------------------------------- B: how long does the signal last
HORIZONS = {"30 min": 6, "1 hour": 12, "2 hours": 24, "4 hours": 48, "8 hours": 96, "1 day": 211, "2 days": 422, "3 days": 633}
n = len(bars)


def ahead(bars_ahead: int) -> np.ndarray:
    """Spread move from the open of bar t+1 to the close bars_ahead later, hedge of bar t held; ZS ticks."""
    move = np.full(n, np.nan)
    t = np.arange(n - bars_ahead - 1)
    leg = lambda name: log_close[name][t + bars_ahead] - log_open[name][t + 1]
    move[t] = (leg("ZS") - beta_meal[t] * leg("ZM") - beta_oil[t] * leg("ZL")) * price_ticks[t]
    return move


events = [(t, rich) for t, rich in find_triggers(z, session_id, 1.645, 0.6745) if in_dev[t] and t + 635 < n]
bar_at, rich_at = np.array([t for t, _ in events]), np.array([rich for _, rich in events])
to_session_end = np.array([np.searchsorted(session_id, session_id[t], side="right") - 1 - t for t in bar_at])   # bars left
week = bars.index[bar_at].tz_convert("America/Chicago").strftime("%G-%V")
print(f"\n== B. {len(events)} session triggers. Median time left in the session at the signal: {np.median(to_session_end) / 12:.1f} hours "
      f"(asia {np.median(to_session_end[session[bar_at] == 'asia']) / 12:.1f}, europe {np.median(to_session_end[session[bar_at] == 'europe']) / 12:.1f}, "
      f"us {np.median(to_session_end[session[bar_at] == 'us']) / 12:.1f})")
rows, curves = [], {}
usable = in_dev & ~np.isnan(z)
for name, bars_ahead in HORIZONS.items():
    move = ahead(bars_ahead)
    fade = pd.Series(-rich_at * move[bar_at])
    weekly = fade.groupby(week).sum()
    error = weekly.std() / np.sqrt(len(weekly)) * len(weekly) / len(fade)
    ok = usable & ~np.isnan(move)
    row = {"held": name, "move back, all (ticks)": fade.mean(), "std error": error, "t": fade.mean() / error, "won": (fade > 0).mean(),
           "winner": fade[fade > 0].mean(), "loser": fade[fade <= 0].mean(),
           "IC, all bars": pd.Series(-gap[ok]).corr(pd.Series(move[ok]), method="spearman")}
    for key in ["asia", "europe", "us"]:
        row[key] = fade[session[bar_at] == key].mean()
    for key in ["2021-22", "2023-Nov24"]:
        row[key] = fade[half[bar_at] == key].mean()
    rows.append(row)
table = pd.DataFrame(rows).set_index("held")
print("move in the fade direction BEFORE costs; t from weekly sums")
print(table.round(2).to_string())

figure, (left, right) = plt.subplots(1, 2, figsize=(22, 8))
x = np.arange(len(HORIZONS))
left.errorbar(x, table["move back, all (ticks)"], yerr=table["std error"], color="black", marker="o", lw=2, capsize=4, label="all sessions (bar = one standard error)")
for key, colour in [("asia", "#1f77b4"), ("europe", "#2ca02c"), ("us", "#d62728")]:
    left.plot(x, table[key], marker="o", color=colour, label=f"signal in {key}")
left.axhline(0, color="grey", lw=0.8)
for cost, style in [(1.1, "--"), (2.1, ":")]:
    left.axhline(cost, color="grey", ls=style, label=f"cost {cost} ticks")
left.set_xticks(x, list(HORIZONS))
left.set_xlabel("held for")
left.set_ylabel("spread move back toward fair value, ticks before costs")
left.set_title("After a session trigger: how far the spread has come back, by holding time", loc="left")
left.legend()
right.plot(x, table["IC, all bars"], marker="o", color="#9467bd", lw=2)
right.axhline(0, color="grey", lw=0.8)
right.set_xticks(x, list(HORIZONS))
right.set_xlabel("horizon")
right.set_ylabel("rank correlation of the gap with the spread's forward move")
right.set_title("IC of the session gap, all bars", loc="left")
figure.tight_layout()
path = out_dir / "spread_session_decay.png"
figure.savefig(path, dpi=100)
print(path)
