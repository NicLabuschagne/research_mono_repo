"""
cards.py
--------
Wheat-corn as signal cards (and History / Tear sheet / My log data) for the terminal preview. Uses the engine's
state.json (written by the dashboard / engine) for signals and positions, and recomputes only the gap history from the
live bar files for the context. Fetches nothing; the only write is log_action (the user's own log, the same file and
record as the old dashboard's "l" key).

  fast  = the anchored fair value (memory 1.5 sessions)       context: |gap| vs the previous 20 sessions
  slow  = the locked rule's K20 Kalman fair value              context: |gap| vs the previous 20 sessions
  beans = price only (LSE has no meal / oil, so no crush fair value live)
"""
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace

import numpy as np

from core_quant.terminal.cards import SignalCard, StrategyDetail, gap_context
from strategies.wheat_corn_lag.anchored_rule import anchored_gap
from strategies.wheat_corn_lag.features import BARS_PER_SESSION
from strategies.wheat_corn_lag.forward import LEGS
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, make_kalman_evaluator
from strategies.wheat_corn_lag.live.engine import ACTIONS_FILE, STATE_FILE, now_utc
from strategies.wheat_corn_lag.strategy import load

CONTEXT_BARS = 20 * BARS_PER_SESSION
ANCHORED, LOCKED = "Wheat-corn anchored", "Wheat-corn locked K20"


def gap_histories() -> dict:
    """The fast and slow gap in cents on every live bar (the same functions the engine uses)."""
    cfg = load()
    live_cfg = replace(cfg, **{field: live_file for field, (_, live_file) in LEGS.items()})
    with redirect_stdout(io.StringIO()):
        _, _, bars, features = make_kalman_evaluator(live_cfg, include_holdout=True)
    price = bars["W"]
    fast = price - price * np.exp(-anchored_gap(bars, cfg.anchored_rule))
    slow = features[f"kgap_{cfg.kalman_rule.memory_sessions}s"] * price
    return {"fast": fast, "slow": slow}


def anchored_signal_word(a: dict) -> str:
    if not a["tradable"] or a["z"] is None:
        return "not tradable"
    if abs(a["z"]) >= a["entry_z"]:
        return "LONG" if a["z"] > 0 else "SHORT"                # z + = wheat cheap -> long
    return "flat"


def locked_signal_word(locked: dict) -> str:
    s = locked["signal_s"]
    if s is None:
        return "not tradable"
    return "LONG" if s >= locked["long_above"] else "SHORT" if s <= locked["short_below"] else "flat"


def wheat_cards(state: dict | None = None, histories: dict | None = None) -> list[SignalCard]:
    state = state or json.loads(STATE_FILE.read_text())
    histories = histories if histories is not None else gap_histories()
    a, locked, beans = state["anchored"], state["locked"], state.get("beans", {})
    as_of = state["last_bar_end_utc"][:16]
    perf = a["performance"]
    fast = SignalCard(
        "Wheat", ANCHORED, "fast", a["wheat"], a["fair_value"], a["gap_c"], TICK_CENTS,
        gap_context(histories["fast"], CONTEXT_BARS), anchored_signal_word(a), as_of, instrument="ZW front",
        note=f"z {a['z']:+.2f}" if a["z"] is not None else "z - (roll window / warm-up)",
        extra={"open positions": len(a["positions"]),
               "forward": f"{perf['trades']} closed, net {perf.get('net_per_trade_c', float('nan')):+.2f} c a trade "
                          f"(expected {a['expected']['net_per_trade_c']:+.2f})"})
    slow_gap = locked["gap_c"]
    slow = SignalCard(
        "Wheat", LOCKED, "slow", a["wheat"], a["wheat"] - slow_gap if slow_gap is not None else None, slow_gap,
        TICK_CENTS, gap_context(histories["slow"], CONTEXT_BARS), locked_signal_word(locked), as_of, instrument="ZW front",
        note=f"s {locked['signal_s']:+.2f}" if locked["signal_s"] is not None else "s - (roll window / warm-up)",
        extra={"open positions": locked["open"],
               "forward": f"{locked['trades']} closed, net {(locked['net_per_trade_c'] or float('nan')):+.2f} c a trade"})
    bean = SignalCard("Beans", "Crush fair value", "fast", beans.get("price"), None, None, 0.25, None, "no model",
                      (beans.get("bar_utc") or "")[:16].replace("T", " "), instrument="ZS front",
                      note="price only (no live meal / oil feed)")
    return [fast, slow, bean]


def wheat_details(state: dict | None = None) -> list[StrategyDetail]:
    """History, tear sheet and equity of both rules, as the old dashboard shows them."""
    state = state or json.loads(STATE_FILE.read_text())
    a, locked = state["anchored"], state["locked"]
    perf, expected = a["performance"], a["expected"]
    anchored = StrategyDetail(
        ANCHORED,
        ["entry (UTC)", "side", "entry #", "z", "price", "exit (UTC)", "exit", "net c", "slow"],
        [[r["entry_time"][:16], r["side"], r["entry_number"], r["z_at_signal"], r["entry_price"], r["exit_time"][:16],
          r["exit_reason"], r["net_c"], "agree" if r["slow_agrees"] else "disagree"] for r in a["history"]],
        [("closed trades", perf["trades"], "-"), ("trades a week", perf["per_week"], expected["per_week"]),
         ("net per trade c", perf.get("net_per_trade_c"), expected["net_per_trade_c"]),
         ("win rate", perf.get("hit_rate"), expected["hit_rate"]),
         ("average winner c", perf.get("avg_winner_c"), expected["avg_winner_c"]),
         ("average loser c", perf.get("avg_loser_c"), expected["avg_loser_c"]),
         ("share exiting at fair value", perf.get("share_fair_value"), expected["share_fair_value"]),
         ("net total c", perf["net_c_total"], "-"), ("max drawdown c", perf["max_dd_c"], "dev. worst -170 over 5 years")],
        perf["equity"], str(STATE_FILE.parent / "equity.png"))
    closed = sorted((r for r in locked["history"] if r["exit_time"]), key=lambda r: r["exit_time"])
    locked_detail = StrategyDetail(
        LOCKED,
        ["entry (UTC)", "side", "price", "exit (UTC)", "exit", "net c"],
        [[r["entry_time"][:16], r["side"], r["entry_price"], r["exit_time"][:16], r["exit_reason"], r["net_c"]]
         for r in sorted(locked["history"], key=lambda r: r["entry_time"], reverse=True)],
        [("closed trades", locked["trades"], "-"), ("open", locked["open"], "-"),
         ("net per trade c", locked["net_per_trade_c"], "-"), ("net total c", locked["net_c_total"], "-")],
        list(np.cumsum([r["net_c"] for r in closed])))
    return [anchored, locked_detail]


# ---- my log: the same file and record as the old dashboard's "l" key, so both screens share one log
def my_log() -> list[list]:
    if not ACTIONS_FILE.exists():
        return []
    records = [json.loads(line) for line in ACTIONS_FILE.read_text().splitlines() if line.strip()]
    return [[r["time"][:16], r["wheat"], r["z"], r["open_positions"], r["text"]] for r in reversed(records)]


def log_action(text: str) -> None:
    a = json.loads(STATE_FILE.read_text())["anchored"]
    record = {"time": str(now_utc().floor("s")), "wheat": a["wheat"], "z": a["z"],
              "open_positions": len(a["positions"]), "text": text}
    with ACTIONS_FILE.open("a") as file:
        file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    for card in wheat_cards():
        print(card.market, card.strategy, card.line, card.fair_value, card.gap, card.gap_ticks, card.context, card.signal)
