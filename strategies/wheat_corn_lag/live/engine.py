"""
engine.py
---------
The live engine behind the dashboard: pull new bars, re-run the rules, write the state the screen shows.

    python -m strategies.wheat_corn_lag.live.engine              # pull from LSE, recompute, write state.json
    python -m strategies.wheat_corn_lag.live.engine --no-fetch   # recompute on the bars already on disk
    python -m strategies.wheat_corn_lag.live.engine --check      # self-check: live files give the validated trades

Each update
  1. pulls new 15-minute bars into data/raw/*_live.parquet (forward.fetch: the API must match the file, and the
     last 15 days are re-read because the vendor revises recent bars by 1-3 ticks for up to ~8 sessions);
  2. re-runs BOTH rules from scratch on all bars (never appends signals: a revised bar can change them);
  3. writes runs/live/state.json (what the screen shows) and appends what changed to runs/live/events.jsonl.
Paper only: nothing here sends an order.

Rules shown
  anchored  the working version (config anchored_rule: memory 1.5, z 1.0 / 1.5 / 2.0 entries, max 3 open, exit at
            fair value or after 2 sessions). Forward results count from FORWARD_START.
  locked    the locked Kalman rule (config kalman_rule), shown alongside like forward.py.
"""
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.wheat_corn_lag.anchored_rule import (anchored_book, anchored_gap, anchored_signal,
                                                     make_anchored_evaluator, slow_signal)
from strategies.wheat_corn_lag.features import BARS_PER_SESSION
from strategies.wheat_corn_lag.forward import LEGS, fetch
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, cents_row, kalman_signal, make_kalman_evaluator
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, wheat_first_notice_days

LIVE_DIR = STRATEGY_DIR / "runs" / "live"
STATE_FILE, EVENTS_FILE, ACTIONS_FILE = LIVE_DIR / "state.json", LIVE_DIR / "events.jsonl", LIVE_DIR / "my_actions.jsonl"
FORWARD_START = pd.Timestamp("2026-11-03", tz="UTC")       # the day after the locked rule's lock
# what the walk-forward led us to expect (runs/20261117-114551_validation, walk-forward picks 2021-2025)
EXPECTED = {"net_per_trade_c": 3.13, "hit_rate": 0.68, "per_week": 1.64, "share_fair_value": 0.74,
            "avg_winner_c": 9.35, "avg_loser_c": -9.90}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def anchored_state(cfg, bars, features) -> dict:
    """Market, signal, open positions and forward results of the anchored rule at the last finished bar."""
    rule = cfg.anchored_rule
    cost = rule.cost_ticks * TICK_CENTS
    tradable = kalman_signal(features, cfg.kalman_rule).notna()
    gap = anchored_gap(bars, rule)
    z = anchored_signal(gap, rule, tradable)
    slow = slow_signal(cfg, features)
    trades = anchored_book(bars, z, rule, slow)
    last = len(bars) - 1
    price = bars["W"].values
    fair = price * np.exp(-gap.values)

    # a trade whose "exit" is only the end of the data is still open ("roll" = the loop ran out of bars);
    # "fair value" on the last bar = price closed back at fair value: the exit fills at the next bar's open
    at_end = trades["exit_bar"] == last
    trades["status"] = np.where(at_end & (trades["exit_reason"] == "roll"), "open",
                                np.where(at_end & (trades["exit_reason"] == "fair value"), "exit at next open", "closed"))
    trades["entry_time"] = bars.index[trades["entry_bar"].values]
    trades["entry_price"] = bars["W_open"].values[trades["entry_bar"].values]
    trades["z_at_signal"] = z.values[trades["entry_bar"].values - 1]
    trades["net_c"] = trades["move_c"] - cost

    positions = []
    for trade in trades[trades["status"] != "closed"].itertuples():
        bars_left = rule.stop_sessions * BARS_PER_SESSION - (last - trade.entry_bar)
        positions.append({"entry_time": str(trade.entry_time), "side": "long" if trade.side > 0 else "short",
                          "entry_number": int(trade.rung), "z_at_signal": float(trade.z_at_signal),
                          "entry_price": float(trade.entry_price), "now_c": float(trade.side * (price[last] - trade.entry_price)),
                          "to_fair_value_c": float(trade.side * (fair[last] - price[last])),
                          "time_stop_in_sessions": float(bars_left / BARS_PER_SESSION),
                          "slow_agrees": not bool(trade.disagree), "status": trade.status})

    # the next add for each side that has open trades: the deepest z entered + add_step
    next_adds = {}
    for side, name in [(1, "long"), (-1, "short")]:
        open_side = trades[(trades["status"] != "closed") & (trades["side"] == side)]
        if len(open_side) and len(open_side) < rule.max_open:
            next_adds[name] = float(side * (np.abs(open_side["z_at_signal"]).max() + rule.add_step))

    forward = trades[(trades["status"] == "closed") & (trades["exit_time"] >= FORWARD_START)].sort_values("exit_time")
    weeks = max((now_utc() - FORWARD_START).days / 7, 1 / 7)
    net = forward["net_c"]
    equity = net.cumsum()
    performance = {"trades": int(len(forward)), "per_week": len(forward) / weeks,
                   "net_c_total": float(net.sum()), "max_dd_c": float((equity - np.maximum(equity.cummax(), 0)).min()) if len(net) else 0.0,
                   "share_fair_value": float((forward["exit_reason"] == "fair value").mean()) if len(net) else np.nan,
                   "equity": [float(value) for value in equity.values]}
    if len(net):
        performance.update({key: float(value) for key, value in cents_row(forward, cost).items() if key != "trades"})

    history = trades[trades["exit_time"] >= FORWARD_START].sort_values("entry_time", ascending=False)
    history_rows = [{"entry_time": str(t.entry_time), "side": "long" if t.side > 0 else "short", "entry_number": int(t.rung),
                     "z_at_signal": float(t.z_at_signal), "entry_price": float(t.entry_price),
                     "exit_time": "" if t.status != "closed" else str(t.exit_time), "exit_reason": t.exit_reason if t.status == "closed" else t.status,
                     "net_c": float(t.net_c), "slow_agrees": not bool(t.disagree)} for t in history.itertuples()]

    fnds = wheat_first_notice_days(bars.index[-1].year, bars.index[-1].year + 1)
    next_fnd = fnds[fnds >= bars.index[-1].normalize()][0]
    slow_now = slow.iloc[-1]
    return {"wheat": float(price[last]), "corn": float(bars["C"].iloc[-1]), "fair_value": float(fair[last]),
            "gap_c": float(price[last] - fair[last]), "z": float(z.iloc[-1]), "z_before": float(z.iloc[-2]),
            "tradable": bool(tradable.iloc[-1]), "entry_z": rule.entry_z, "add_step": rule.add_step, "max_open": rule.max_open,
            "slow_gap_c": float(-slow_now * price[last]) if pd.notna(slow_now) else np.nan,
            "slow_agrees": bool(np.sign(slow_now) == np.sign(z.iloc[-1])) if pd.notna(slow_now) and pd.notna(z.iloc[-1]) else None,
            "next_roll_window_from": str((next_fnd - pd.offsets.BDay(cfg.roll_business_days_before_fnd)).date()),
            "positions": positions, "next_adds": next_adds, "performance": performance, "history": history_rows,
            "expected": EXPECTED}


def locked_state(cfg, evaluate, bars, features) -> dict:
    """The locked Kalman rule: its signal now and its forward trades (as forward.py reports them)."""
    rule = cfg.kalman_rule
    cost = rule.cost_ticks * TICK_CENTS
    trades = evaluate({}, FORWARD_START.tz_localize(None), pd.Timestamp("2100-01-01")).attrs.get("trades", pd.DataFrame())
    s = kalman_signal(features, rule)
    window = s.iloc[-rule.quantile_window_bars - 1:-1]
    state = {"signal_s": float(s.iloc[-1]), "long_above": float(window.quantile(1 - rule.tail)),
             "short_below": float(window.quantile(rule.tail)),
             "gap_c": float(features[f"kgap_{rule.memory_sessions}s"].iloc[-1] * bars["W"].iloc[-1]),
             "open": 0, "trades": 0, "net_per_trade_c": np.nan, "net_c_total": 0.0}
    if len(trades):
        last = len(bars) - 1
        still_open = ((trades["exit_bar"] == last) & (trades["exit_reason"] == "time")
                      & (trades["exit_bar"] - trades["entry_bar"] < rule.hold_bars))
        closed = trades[~still_open]
        net = closed["move_c"] - cost
        state.update({"open": int(still_open.sum()), "trades": int(len(closed)),
                      "net_per_trade_c": float(net.mean()) if len(net) else np.nan, "net_c_total": float(net.sum())})
    return state


def events_between(old: dict | None, new: dict) -> list[str]:
    """What changed since the last update, in words."""
    if old is None:
        return ["engine started"]
    events = []
    a_old, a_new = old["anchored"], new["anchored"]
    old_open = {(p["entry_time"], p["side"]) for p in a_old["positions"]}
    new_open = {(p["entry_time"], p["side"]): p for p in a_new["positions"]}
    for key, p in new_open.items():
        if key not in old_open:
            events.append(f"ANCHORED {p['side'].upper()} entry {p['entry_number']} at z {p['z_at_signal']:+.2f}, "
                          f"{p['entry_price']:.2f} c")
        elif p["status"] == "exit at next open":
            events.append(f"ANCHORED {p['side']} from {key[0][5:16]} closed back at fair value: exit at the next open")
    for row in a_new["history"]:
        if (row["entry_time"], row["side"]) in old_open and row["exit_time"]:
            events.append(f"ANCHORED {row['side']} from {row['entry_time'][5:16]} closed ({row['exit_reason']}): "
                          f"{row['net_c']:+.1f} c net")
    for level in [1.0, 1.5, 2.0]:
        for sign, name in [(1, "long"), (-1, "short")]:
            was, now = (sign * (value if value is not None else np.nan) for value in (a_old["z"], a_new["z"]))
            if np.isfinite(was) and np.isfinite(now) and was < level <= now:
                events.append(f"z crossed {sign * level:+.1f} ({name} side)")
    if a_old["tradable"] != a_new["tradable"]:
        events.append("TRADABLE again" if a_new["tradable"] else "NOT tradable: roll window")
    if old["locked"]["open"] != new["locked"]["open"]:
        events.append(f"LOCKED rule open positions {old['locked']['open']} -> {new['locked']['open']}")
    return events


def append_events(lines: list[str]) -> None:
    with EVENTS_FILE.open("a") as file:
        for line in lines:
            file.write(json.dumps({"time": str(now_utc().floor("s")), "event": line}) + "\n")


def equity_chart(state: dict) -> None:
    equity = state["anchored"]["performance"]["equity"]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(range(1, len(equity) + 1), equity, marker="o", ms=4, lw=1.6, color="#2a78d6")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xlabel("closed trade number")
    axis.set_ylabel("cumulative net cents, 1 contract per trade")
    axis.set_title(f"Anchored rule, forward from {FORWARD_START:%Y-%m-%d}: {len(equity)} closed trades", loc="left")
    figure.tight_layout()
    figure.savefig(LIVE_DIR / "equity.png", dpi=120)
    plt.close(figure)


def update(fetch_new: bool = True) -> dict:
    """One poll: fetch, recompute, write state and events. Returns the new state."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load()
    log = io.StringIO()
    with redirect_stdout(log):                     # forward.fetch prints; keep the screen clean, keep the lines
        if fetch_new:
            fetch(cfg)
        live_cfg = replace(cfg, **{field: live_file for field, (_, live_file) in LEGS.items()})
        evaluate, _, bars, features = make_kalman_evaluator(live_cfg, include_holdout=True)
    state = {"updated_utc": str(now_utc().floor("s")), "last_bar_end_utc": str(bars.index[-1]),
             "fetched": fetch_new, "fetch_log": [line for line in log.getvalue().splitlines() if line.strip()],
             "anchored": anchored_state(cfg, bars, features), "locked": locked_state(cfg, evaluate, bars, features)}
    old = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None
    revised = [line for line in state["fetch_log"] if "revised" in line and " 0 closes revised" not in line]
    append_events(events_between(old, state) + revised)
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str).replace("NaN", "null"))
    equity_chart(state)
    return state


def check() -> None:
    """The live files must give the same anchored trades as the validated development files. Compared on trades
    exiting before 2025-12-15: a trade still open when the development data ends (31 Dec) is cut there, but runs
    on in the live files, so the last days cannot match by design."""
    cfg = load()
    start, end = pd.Timestamp("2021-01-21"), pd.Timestamp("2025-12-15")
    results = []
    for files_cfg, with_holdout in [(cfg, False), (replace(cfg, **{f: live for f, (_, live) in LEGS.items()}), True)]:
        _, _, bars, features = make_kalman_evaluator(files_cfg, include_holdout=with_holdout)
        tradable = kalman_signal(features, cfg.kalman_rule).notna()
        evaluate = make_anchored_evaluator(cfg, bars, tradable, slow_signal(cfg, features))
        trades = evaluate({}, start, end).attrs["trades"]
        results.append(trades.assign(entry_time=bars.index[trades["entry_bar"].values])[["entry_time", "side", "move_c"]]
                       .reset_index(drop=True))
    development, live = results
    assert len(development) == len(live) and (development["entry_time"] == live["entry_time"]).all() \
        and np.allclose(development["move_c"], live["move_c"]), "live pipeline differs from the validated rule"
    print(f"OK: live files give the validated anchored trades ({len(live)} trades exiting 2021-01-21 to 2025-12-14, identical)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit()
    result = update(fetch_new="--no-fetch" not in sys.argv)
    a = result["anchored"]
    print(f"last bar {result['last_bar_end_utc']}: wheat {a['wheat']:.2f}, fair value {a['fair_value']:.2f}, z {a['z']:+.2f}, "
          f"{len(a['positions'])} open; forward {a['performance']['trades']} closed trades. State: {STATE_FILE}")
