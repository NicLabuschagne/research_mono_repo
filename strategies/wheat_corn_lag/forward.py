"""
forward.py
----------
Forward (paper) tracking of the LOCKED rule. It changes no setting: config.yaml and its hash stay as locked.

    python -m strategies.wheat_corn_lag.forward                      # pull new bars from LSE, re-run, report
    python -m strategies.wheat_corn_lag.forward --no-fetch           # re-run on the bars already on disk
    python -m strategies.wheat_corn_lag.forward --start 2026-07-01   # report from another date (see below)

Data     the locked files (config.yaml) are never modified. Each leg gets a "live" copy in data/raw that starts
         as the locked file and grows by the new 15-minute bars from London Strategic Edge (WHEAT.F, CORN.F).
         Needs LSE_API_KEY in the environment. A bar still forming is never stored.
Rule     kalman_rule.py, untouched, run over all bars (so the filter and the one-year thresholds are warm).
Report   only trades that EXIT on or after --start are shown. The default is the day after the lock
         (2026-11-03): holdout B (Jul-Oct 2026) feeds the filter but its results stay unseen.
Output   runs/forward/: trades.csv, daily.csv, status.json, equity.png (the inputs for a dashboard).
"""
import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, cents_row, kalman_signal, make_kalman_evaluator
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

RAW = ROOT / "data" / "raw"
LEGS = {"wheat_file": ("WHEAT.F", "WHEAT_futures_15m_live.parquet"),
        "corn_file": ("CORN.F", "CORN_futures_15m_live.parquet")}
DEFAULT_START = "2026-11-03"          # the day after the rule was locked
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
ROW_CAP = 5000                        # the vendor's rows per call


def update_leg(client, symbol: str, locked_file: str, live_file: str, bar_minutes: int) -> pd.DataFrame:
    """Append the vendor's new bars to the live copy of one leg and save it."""
    live_path = RAW / live_file
    stored = pd.read_parquet(live_path if live_path.exists() else RAW / locked_file)
    stored["time"] = pd.to_datetime(stored["time"], utc=True)

    # re-read the last 15 days as well: the vendor REVISES recent bars by 1-3 ticks for up to ~8 sessions
    # (seen 2026-11-16: the 30 Oct download vs the API on 19-30 Oct). The vendor's newest version wins.
    start = stored["time"].max() - pd.Timedelta(days=15)
    rows = []
    while True:
        page = client.candles(symbol, f"{bar_minutes}m", start=start.strftime("%Y-%m-%dT%H:%M:%S"),
                              limit=ROW_CAP, order="asc")
        rows += page
        if len(page) < ROW_CAP:
            break
        start = pd.Timestamp(page[-1]["timestamp"], tz="UTC") + pd.Timedelta(minutes=bar_minutes)
    if not rows:
        raise ValueError(f"{symbol}: the vendor returned no bars from {start}")
    new = pd.DataFrame(rows)
    new["time"] = pd.to_datetime(new["timestamp"], utc=True)
    for column in PRICE_COLUMNS:
        new[column] = pd.to_numeric(new[column])
    finished = new["time"] + pd.Timedelta(minutes=bar_minutes) <= pd.Timestamp.now(tz="UTC")
    new = new.loc[finished, ["time"] + PRICE_COLUMNS]

    # trust boundary: the API must be the same series as the file. Revisions move a close by a tick or two;
    # another contract or a shifted clock moves it by much more.
    both = stored.merge(new, on="time", suffixes=("_old", "_new"))
    difference = (both["close_old"] - both["close_new"]).abs()
    print(f"{symbol}: {len(new)} bars fetched, {len(both)} overlap the file, {(difference > 1e-9).sum()} closes "
          f"revised (average {difference.mean():.2f} c, largest {difference.max():.2f} c)")
    if len(both) == 0 or difference.mean() > 0.5:
        raise ValueError(f"{symbol}: API bars do not match {live_file}; not saved")

    merged = pd.concat([stored[["time"] + PRICE_COLUMNS], new]).drop_duplicates("time", keep="last")
    merged = merged.sort_values("time").reset_index(drop=True)
    merged.to_parquet(live_path)
    print(f"{symbol}: {live_file} now ends {merged['time'].max()} (+{len(merged) - len(stored)} bars)")
    return merged


def fetch(cfg) -> None:
    from lse import LSE                    # imported here so --no-fetch works without the SDK or a key
    from core_quant.config import load_env
    load_env()                             # the key lives in the project's .env (ignored by git)
    key = os.environ.get("LSE_API_KEY")
    if not key:
        raise SystemExit("LSE_API_KEY is not set (or use --no-fetch)")
    client = LSE(api_key=key)
    for field, (symbol, live_file) in LEGS.items():
        update_leg(client, symbol, getattr(cfg, field), live_file, cfg.bar_minutes)


def price_chart(bars: pd.DataFrame, gap: pd.Series, trades: pd.DataFrame, start: pd.Timestamp, path) -> None:
    """Wheat 15-minute candles, the Kalman fair value and every trade, from the day before start.
    Bars are drawn side by side (no empty space for the hours the market is closed)."""
    first = int(np.searchsorted(bars.index, pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)))
    x = np.arange(len(bars) - first)
    open_, high, low, close = (bars[name].values[first:] for name in ["W_open", "W_high", "W_low", "W"])
    fair = close * np.exp(-gap.values[first:])               # gap = log wheat - log fair value, known BEFORE the bar
    colours = np.where(close >= open_, "#2ca02c", "#d62728")

    figure, axis = plt.subplots(figsize=(22, 9))
    axis.vlines(x, low, high, color=colours, lw=0.7)
    axis.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours)
    # drawing only: slide fair value by its average distance from price in this window, so the two shapes can be
    # compared. The rule itself trades on the UNSHIFTED gap.
    shift = np.nanmean(fair - close)
    axis.plot(x, fair - shift, color="#ff7f0e", lw=1.6,
              label=f"fair value (Kalman, wheat on corn, 20 sessions), moved {-shift:+.1f} c to sit on price")
    for trade in trades.itertuples():
        entry_x, exit_x = trade.entry_bar - first, trade.exit_bar - first
        exit_price = trade.entry_price + trade.side * trade.move_c
        colour = "#1f77b4" if trade.net_c > 0 else "#7f0000"
        axis.plot([entry_x, exit_x], [trade.entry_price, exit_price], color=colour, lw=1.3, ls="--", zorder=4)
        axis.scatter(entry_x, trade.entry_price, marker="^" if trade.side > 0 else "v", s=130, color=colour,
                     edgecolor="black", zorder=5)
        axis.scatter(exit_x, exit_price, marker="X" if trade.exit_reason == "time" else "*", s=110, color=colour,
                     edgecolor="black", zorder=5)
        axis.annotate(f"{trade.net_c:+.1f}", (exit_x, exit_price), textcoords="offset points", xytext=(5, 6), fontsize=8)
    sessions = bars["session"].values[first:]
    session_starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    axis.set_xticks(session_starts, [f"{day:%d %b}" for day in bars.index[first + session_starts]])
    for session_start in session_starts:
        axis.axvline(session_start - 0.5, color="grey", lw=0.4, alpha=0.5)
    axis.set_xlim(-1, len(x) + 1)
    axis.set_ylabel("wheat, cents per bushel")
    axis.set_title("Wheat 15-minute candles, fair value and the locked rule's trades. Triangle = entry (up = long), "
                   "X = time exit, star = target or stop, number = net cents. Blue = winner, dark red = loser",
                   loc="left", fontsize=10)
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def run(cfg, start: pd.Timestamp) -> None:
    live_cfg = replace(cfg, **{field: live_file for field, (_, live_file) in LEGS.items()})
    evaluate, _, bars, features = make_kalman_evaluator(live_cfg, include_holdout=True)
    rule = cfg.kalman_rule
    cost = rule.cost_ticks * TICK_CENTS

    # self-check: on the live files the rule must still give holdout A exactly as it was run on 2026-11-02
    holdout_a = evaluate({}, pd.Timestamp(cfg.split.holdout_start), pd.Timestamp(cfg.split.holdout_end))
    a_trades = holdout_a.attrs["trades"]
    a_net = (a_trades["move_c"] - cost).mean()
    assert len(a_trades) == 52 and abs(a_net - 0.60) < 0.01, f"holdout A changed: {len(a_trades)} trades, {a_net:.2f} c"

    daily = evaluate({}, start, pd.Timestamp("2100-01-01"))
    trades = daily.attrs.get("trades", pd.DataFrame())
    out_dir = STRATEGY_DIR / "runs" / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)

    # where the signal stands at the last finished bar
    s = kalman_signal(features, rule)
    window = s.iloc[-rule.quantile_window_bars - 1:-1]              # the past year, before the last bar
    gap_cents = features[f"kgap_{rule.memory_sessions}s"].iloc[-1] * bars["W"].iloc[-1]
    status = {"last_bar_end_utc": str(bars.index[-1]), "wheat_close": float(bars["W"].iloc[-1]),
              "corn_close": float(bars["C"].iloc[-1]), "signal_s": float(s.iloc[-1]),
              "long_above": float(window.quantile(1 - rule.tail)), "short_below": float(window.quantile(rule.tail)),
              "gap_cents_wheat_minus_fair": float(gap_cents), "report_start": str(start.date()), "trades": 0}

    print(f"\n== FORWARD, locked rule, trades exiting from {start:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d %H:%M} UTC")
    if len(trades):
        last_bar = len(bars) - 1
        trades = trades.copy()
        # a trade whose "exit" is only the end of the data is still open: its result is marked to the last close
        trades["still_open"] = ((trades["exit_bar"] == last_bar) & (trades["exit_reason"] == "time")
                                & (trades["exit_bar"] - trades["entry_bar"] < rule.hold_bars))
        trades["entry_time"] = bars.index[trades["entry_bar"].values]
        trades["entry_price"] = bars["W_open"].values[trades["entry_bar"].values]
        trades["net_c"] = trades["move_c"] - cost
        closed = trades[~trades["still_open"]]
        weeks = max((bars.index[-1].tz_localize(None) - start).days / 7, 1e-9)
        summary = {**cents_row(closed, cost), "per_week": len(closed) / weeks,
                   "net_cents_total": closed["net_c"].sum(), "dollars_per_contract": closed["net_c"].sum() * 50,
                   "share_target": (closed["exit_reason"] == "target").mean(),
                   "share_stop": (closed["exit_reason"] == "stop").mean(), "open_now": int(trades["still_open"].sum())}
        print("closed trades, cents (net per trade = all trades after costs; average winner / loser = winners / losers only)")
        print(pd.Series(summary).round(2).to_string())
        print("\nlocked expectations: net per trade -0.7..+3.7 c, win rate 60-67%, ~2.4 trades a week, ~40% at target")
        columns = ["entry_time", "side", "entry_price", "exit_time", "exit_reason", "net_c", "still_open"]
        print("\n" + trades[columns].round(2).to_string(index=False))
        trades[columns + ["entry_bar", "exit_bar", "move_c"]].to_csv(out_dir / "trades.csv", index=False)
        status.update({name: float(value) for name, value in summary.items()})

        figure, axis = plt.subplots(figsize=(11, 4.4))
        axis.plot(closed["exit_time"], closed["net_c"].cumsum().values, marker="o", ms=4, lw=1.5)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_ylabel("cumulative net cents, 1 contract per trade")
        axis.set_title(f"Forward, locked rule from {start:%Y-%m-%d}: {len(closed)} closed trades, "
                       f"net per trade {summary['net_per_trade_c']:+.2f} c", loc="left")
        figure.tight_layout()
        figure.savefig(out_dir / "equity.png", dpi=130)
        price_chart(bars, features[f"kgap_{rule.memory_sessions}s"], trades, start, out_dir / "price_chart.png")
    else:
        print("no trades yet")
    daily.to_csv(out_dir / "daily.csv")
    status["tradable_now"] = bool(not np.isnan(status["signal_s"]))       # False inside / just after a roll window
    (out_dir / "status.json").write_text(pd.Series(status).to_json(indent=2))   # pandas writes NaN as null
    print(f"\nsignal now: s {status['signal_s']:+.2f} (long above {status['long_above']:+.2f}, short below "
          f"{status['short_below']:+.2f}), wheat minus fair value {gap_cents:+.1f} c")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    cfg = load()
    if "--no-fetch" not in sys.argv:
        fetch(cfg)
    start_text = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else DEFAULT_START
    run(cfg, pd.Timestamp(start_text))
