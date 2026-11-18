"""CHARTS ONLY: the outright beans alert, computed on 5-minute bars exactly as before (bean_spread.py: live fair
value 10x slower, live gap stretched 8% of bars, session gap agreeing, full turn), DISPLAYED on 15-minute candles
so each chart holds nine trading days. Three 5-minute bars make one candle, counted from each session's start;
the fair-value lines are their values at the candle's close; an alert sits on the candle that contains it."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("bean_alert_15min.py", "bean_spread.py"))
bars, session_id, TICK, out_dir = g["bars"], g["session_id"], g["TICK"], g["out_dir"]
open_s, close_s, high_s, low_s = g["open_s"], g["close_s"], g["high_s"], g["low_s"]
live_fair = close_s - g["gap_v"] * TICK
session_fair = close_s - g["session_gap"].values * TICK
alerts = g["found"]["beans yardstick, gap + fair value turn (as drawn)"][0]
DAYS_SHOWN, BARS_PER_CANDLE = 9, 3

position_in_session = pd.Series(session_id).groupby(session_id).cumcount().values
candle_id = pd.factorize(pd.Series(session_id).astype(str) + "_" + pd.Series(position_in_session // BARS_PER_CANDLE).astype(str))[0]
frame = pd.DataFrame({"candle": candle_id, "open": open_s, "high": high_s, "low": low_s, "close": close_s,
                      "live": live_fair, "session": session_fair, "session_id": session_id, "bar": np.arange(len(bars))})
candles = frame.groupby("candle").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                      live=("live", "last"), session=("session", "last"), session_id=("session_id", "first"),
                                      last_bar=("bar", "last"))
assert (candles["high"] >= candles[["open", "close"]].max(axis=1) - 1e-6).all()
candle_date = bars["trading_date"].values[candles["last_bar"].values]
candle_time = bars.index[candles["last_bar"].values].tz_convert("Europe/London")
trading_dates = np.array(sorted(bars["trading_date"].unique()))

for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-08"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:DAYS_SHOWN]
    shown = np.flatnonzero(pd.Series(candle_date).isin(days).values)
    first, last = shown[0], shown[-1] + 1
    part = candles.iloc[first:last]
    x = np.arange(len(part))
    colours = np.where(part["close"] >= part["open"], "#2ca02c", "#d62728")
    figure, axis = plt.subplots(figsize=(26, 12))
    axis.vlines(x, part["low"], part["high"], color=colours, lw=0.8, alpha=0.7)
    axis.bar(x, np.maximum((part["close"] - part["open"]).abs(), 0.05), bottom=np.minimum(part["open"], part["close"]), width=0.7, color=colours, alpha=0.7)
    axis.plot(x, part["live"], color="black", lw=1.2, label="live fair value (10x slower)")
    axis.plot(x, part["session"], color="darkorange", lw=1.3, label="session fair value")
    reach = (part["high"].max() - part["low"].min()) * 0.025
    chosen = [(candle_id[t] - first, side) for t, side in alerts if first <= candle_id[t] < last]
    for sign, marker, label in [(1, "v", "sell alert (beans rich)"), (-1, "^", "buy alert (beans cheap)")]:
        xs = np.array([i for i, side in chosen if side == sign], dtype=int)
        ys = part["high"].values[xs] + reach if sign == 1 else part["low"].values[xs] - reach
        axis.scatter(xs, ys, marker=marker, s=260, color="#1f77b4", edgecolor="black", linewidth=1.5, zorder=5, label=f"{label}: {len(xs)}")
    ids = part["session_id"].values
    for session_start in np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]):
        axis.axvline(session_start - 0.5, color="grey", lw=0.6, alpha=0.6)
    dates_shown = candle_date[first:last]
    day_starts = np.flatnonzero(np.r_[True, dates_shown[1:] != dates_shown[:-1]])
    axis.set_xticks(day_starts, [f"{candle_time[first + i]:%d %b\n%H:%M}" for i in day_starts])
    axis.set_xlim(-1, len(x) + 1)
    axis.set_ylabel("soybeans, cents per bushel (roll-adjusted)")
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 15-minute candles. Alerts are computed on 5-minute bars "
                   f"and drawn on the candle that contains them. Grey lines = session starts", loc="left")
    axis.legend(loc="upper left", ncol=4)
    figure.tight_layout()
    path = out_dir / f"alert_15min_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path, len(chosen), "alerts")
