"""
dashboard.py
------------
Terminal dashboard for live (paper) monitoring of the wheat-corn rules.

    python -m strategies.wheat_corn_lag.live.dashboard

It shows runs/live/state.json at once, then polls LSE through engine.update() in a background thread: at start
and every POLL_MINUTES (bars are 15 minutes; a short delay is fine for a monitoring screen). Closing the screen
loses nothing: the engine writes everything to runs/live/.

Keys:  r refresh now · l log what I would do · d my disc entry · x my disc exit · t open the equity chart · 1-5 tabs · q quit
"""
import json
import os

import numpy as np
import pandas as pd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Input, Label, RichLog, Sparkline, Static, TabbedContent, TabPane

from strategies.wheat_corn_lag.live import engine

POLL_MINUTES = 15
GREEN, RED, AMBER, MUTED = "#3fb950", "#f85149", "#d29922", "#8b949e"


def money(value, digits: int = 2) -> str:
    """Signed cents, green / red."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"[{MUTED}]-[/]"
    colour = GREEN if value > 0 else RED if value < 0 else MUTED
    return f"[{colour}]{value:+.{digits}f}[/]"


def number(value, fmt: str = "{:.2f}") -> str:
    return f"[{MUTED}]-[/]" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value)


class LogAction(ModalScreen[str]):
    """Type what you would do now; it is saved with the market at this moment."""

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("What would you do now? (Enter to save, Esc to cancel)")
            yield Input(placeholder="e.g. would exit long #1 here, price touched fair value")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def key_escape(self) -> None:
        self.dismiss("")


class Form(ModalScreen):
    """A few labelled fields. check(values) returns the record to save, or an error message to show."""

    def __init__(self, heading: str, fields: list[tuple[str, str]], check):
        super().__init__()
        self.heading, self.fields, self.check = heading, fields, check

    def compose(self) -> ComposeResult:
        with Vertical(id="form"):
            yield Label(f"{self.heading}   (Tab between fields, Enter to save, Esc to cancel)")
            for name, hint in self.fields:
                yield Input(placeholder=f"{name}: {hint}", id=f"field_{name}")
            yield Label("", id="form_error")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        values = {name: self.query_one(f"#field_{name}", Input).value.strip() for name, _ in self.fields}
        result = self.check(values)
        if isinstance(result, str):
            self.query_one("#form_error", Label).update(f"[{RED}]{result}[/]")
        else:
            self.dismiss(result)

    def key_escape(self) -> None:
        self.dismiss(None)


def parse_time(text: str) -> pd.Timestamp:
    """Blank = now; '14:30' = today 14:30; or '2026-11-17 14:30'. Always UTC."""
    return engine.now_utc().floor("s") if not text else pd.Timestamp(text, tz="UTC")


def parse_price(text: str, near: float | None) -> float:
    price = float(text)
    if near and abs(price / near - 1) > 0.1:          # catches typos like 61.25 for 612.5
        raise ValueError(f"price {price} is more than 10% from the market ({near:.2f})")
    return price


class Dashboard(App):
    TITLE = "wheat-corn · live paper dashboard"
    CSS = """
    Screen { background: #0d1117; color: #c9d1d9; }
    .panel { border: round #30363d; border-title-color: #58a6ff; padding: 0 1; height: auto; }
    #banner { border: heavy #1f6feb; height: 4; padding: 0 1; }
    #verdict { border: heavy #d29922; height: 3; padding: 0 1; }
    #top, #bottom { height: auto; }
    #market, #signal, #performance, #locked { width: 1fr; }
    #positions { height: 9; }
    #events { height: 12; width: 1fr; }
    #equity_box { width: 1fr; height: 12; }
    Sparkline { height: 6; margin: 1 0; }
    DataTable { height: auto; max-height: 30; }
    #dialog { width: 80; height: 7; border: heavy #58a6ff; background: #161b22; padding: 1; }
    #form { width: 90; height: auto; border: heavy #58a6ff; background: #161b22; padding: 1; }
    LogAction, Form { align: center middle; }
    """
    BINDINGS = [Binding("r", "refresh", "refresh now"), Binding("l", "log_action", "log my action"),
                Binding("d", "disc_entry", "disc entry"), Binding("x", "disc_exit", "disc exit"),
                Binding("t", "open_chart", "equity chart"), Binding("1", "tab('live')", "live"),
                Binding("2", "tab('history')", "history"), Binding("3", "tab('tearsheet')", "tear sheet"),
                Binding("4", "tab('mylog')", "my log"), Binding("5", "tab('disc')", "disc entries"),
                Binding("q", "quit", "quit")]

    def __init__(self):
        super().__init__()
        self.state = None
        self.polling = False

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="live"):
            with TabPane("Live [1]", id="live"):
                yield Static(id="banner")
                with Horizontal(id="top"):
                    yield Static(id="market", classes="panel")
                    yield Static(id="signal", classes="panel")
                yield DataTable(id="positions", classes="panel", zebra_stripes=True)
                with Horizontal(id="bottom"):
                    yield Static(id="performance", classes="panel")
                    yield Static(id="locked", classes="panel")
                with Horizontal():
                    with Vertical(id="equity_box", classes="panel"):
                        yield Static(id="equity_label")
                        yield Sparkline([], id="equity")
                    yield RichLog(id="events", classes="panel", markup=True, wrap=True)
                yield Static(id="verdict")
            with TabPane("History [2]", id="history"):
                yield DataTable(id="history_table", zebra_stripes=True)
            with TabPane("Tear sheet [3]", id="tearsheet"):
                yield DataTable(id="tearsheet_table")
                yield Static(id="tearsheet_note")
            with TabPane("My log [4]", id="mylog"):
                yield DataTable(id="mylog_table", zebra_stripes=True)
            with TabPane("Disc entries [5]", id="disc"):
                yield DataTable(id="disc_table", zebra_stripes=True)
                yield Static(id="disc_note")
        yield Footer()

    def on_mount(self) -> None:
        for panel_id, title in [("market", "MARKET"), ("signal", "SIGNAL · anchored z"), ("positions", "OPEN POSITIONS · anchored"),
                                ("performance", "FORWARD · anchored"), ("locked", "LOCKED KALMAN RULE"),
                                ("equity_box", "EQUITY · closed forward trades"), ("events", "EVENTS (UTC)")]:
            self.query_one(f"#{panel_id}").border_title = title
        self.query_one("#positions", DataTable).add_columns("#", "side", "entry (UTC)", "entry", "z", "price", "now c",
                                                             "to fair value c", "time stop in", "slow", "status")
        self.query_one("#history_table", DataTable).add_columns("entry (UTC)", "side", "entry", "z", "price", "exit (UTC)",
                                                                "exit", "net c", "slow")
        self.query_one("#tearsheet_table", DataTable).add_columns("measure", "forward", "walk-forward expectation")
        self.query_one("#mylog_table", DataTable).add_columns("time (UTC)", "wheat", "z", "open", "what I would do")
        self.query_one("#disc_table", DataTable).add_columns(
            "id", "market", "side", "my entry (UTC)", "my price", "model entry", "model price", "entry edge c",
            "my exit", "model exit", "exit edge c", "my c", "model c", "note")
        self.query_one("#disc_note", Static).update(
            "\nd = log a disc entry, x = log its exit. Saved append-only in forward_ledger/disc_trades.jsonl.\n"
            "Wheat is matched to the anchored rule's trade on the same side entered within 2 h of yours.\n"
            "Edge = cents better than the model (+ = you did better). my c / model c = before costs, open trades "
            "marked to the last price.\nBeans: price only, no model yet (LSE has no soybean meal or oil for the crush fair value).")
        if engine.STATE_FILE.exists():
            self.show(json.loads(engine.STATE_FILE.read_text()))
        self.load_events()
        self.load_my_log()
        self.action_refresh()
        self.set_interval(POLL_MINUTES * 60, self.action_refresh)
        self.set_interval(30, self.update_banner)

    # ---- polling (in a thread, so the screen stays responsive)
    def action_refresh(self) -> None:
        if not self.polling:
            self.polling = True
            self.update_banner()
            self.run_worker(self.poll, thread=True, exclusive=True)

    def poll(self) -> None:
        try:
            state = engine.update(fetch_new=True)
            self.call_from_thread(self.show, state)
        except Exception as error:                       # keep the screen alive; say what failed
            engine.append_events([f"POLL FAILED: {error}"])
        finally:
            self.polling = False
            self.call_from_thread(self.load_events)
            self.call_from_thread(self.update_banner)

    # ---- drawing
    def update_banner(self) -> None:
        state = self.state
        if state is None:
            self.query_one("#banner", Static).update("[b]WHEAT-CORN[/]  waiting for the first update ...")
            return
        last_bar = pd.Timestamp(state["last_bar_end_utc"])
        age = (pd.Timestamp.now(tz="UTC") - last_bar).total_seconds() / 60
        fresh = GREEN if age < 40 else AMBER if age < 24 * 60 else RED
        status = f"[{AMBER}]● polling LSE ...[/]" if self.polling else f"[{GREEN}]● idle[/] next poll in ≤ {POLL_MINUTES} min"
        self.query_one("#banner", Static).update(
            f"[b reverse #1f6feb] ◆ WHEAT-CORN · ANCHORED (memory 1.5, max 3) [/]  [b]PAPER[/]  {status}\n"
            f"last bar {last_bar:%Y-%m-%d %H:%M} UTC  [{fresh}]({age:,.0f} min old)[/]   updated {state['updated_utc'][:19]}"
            f"   forward from {engine.FORWARD_START:%Y-%m-%d}   no orders are sent")

    def show(self, state: dict) -> None:
        self.state = state
        a, locked = state["anchored"], state["locked"]
        self.update_banner()
        side_word = "wheat CHEAP vs corn → long side" if (a["z"] or 0) > 0 else "wheat RICH vs corn → short side"
        self.query_one("#market", Static).update(
            f"Wheat      {a['wheat']:>9.2f} c\nCorn       {a['corn']:>9.2f} c\n"
            f"Fair value {a['fair_value']:>9.2f} c   wheat {money(a['gap_c'], 1)} c vs fair value\n"
            f"Slow (K20) gap {number(a['slow_gap_c'], '{:+.1f}')} c   next roll window from {a['next_roll_window_from']}\n"
            f"Beans      {number(state.get('beans', {}).get('price'), '{:>9.2f}')} c   (price only, no fair value yet)")
        z = a["z"]
        if not a["tradable"] or z is None:
            zone = f"[{AMBER}]NOT TRADABLE (roll window / warm-up)[/]"
        elif abs(z) >= a["entry_z"]:
            zone = f"[b {GREEN if z > 0 else RED}]IN ZONE · {'LONG' if z > 0 else 'SHORT'}[/]"
        else:
            zone = f"[{MUTED}]flat · entry at |z| ≥ {a['entry_z']:.1f}[/]"
        slow = {True: f"[{GREEN}]AGREES[/]", False: f"[{AMBER}]DISAGREES (adds earn less)[/]", None: f"[{MUTED}]-[/]"}[a["slow_agrees"]]
        adds = "   ".join(f"{side} add at z {level:+.1f}" for side, level in a["next_adds"].items()) or "no open side"
        self.query_one("#signal", Static).update(
            f"z now {number(z, '{:+.2f}')}  (bar before {number(a['z_before'], '{:+.2f}')})   {zone}\n"
            f"{side_word if z is not None and np.isfinite(z) else ''}\n"
            f"slow fair value {slow}\nnext adds: {adds}   (max {a['max_open']} open)")

        table = self.query_one("#positions", DataTable)
        table.clear()
        for number_, p in enumerate(a["positions"], 1):
            table.add_row(str(number_), p["side"], p["entry_time"][5:16], f"#{p['entry_number']}", f"{p['z_at_signal']:+.2f}",
                          f"{p['entry_price']:.2f}", money(p["now_c"]), f"{p['to_fair_value_c']:+.1f}",
                          f"{p['time_stop_in_sessions']:.1f} sessions", "agree" if p["slow_agrees"] else "disagree", p["status"])
        if not a["positions"]:
            table.add_row("", "no open positions", *[""] * 9)

        perf, expected = a["performance"], a["expected"]
        self.query_one("#performance", Static).update(
            f"closed trades {perf['trades']}  ({perf['per_week']:.1f} a week, expected {expected['per_week']:.1f})\n"
            f"net per trade {money(perf.get('net_per_trade_c'))} c   expected {expected['net_per_trade_c']:+.2f}\n"
            f"win rate {number(perf.get('hit_rate'), '{:.0%}')}   winner {money(perf.get('avg_winner_c'), 1)} / "
            f"loser {money(perf.get('avg_loser_c'), 1)} c\n"
            f"net total {money(perf['net_c_total'], 1)} c   max drawdown {money(perf['max_dd_c'], 1)} c")
        self.query_one("#locked", Static).update(
            f"signal s {number(locked['signal_s'], '{:+.2f}')}  long above {locked['long_above']:+.2f} · short below {locked['short_below']:+.2f}\n"
            f"wheat vs K20 fair value {money(locked['gap_c'], 1)} c\n"
            f"open {locked['open']}   closed {locked['trades']}   net per trade {money(locked['net_per_trade_c'])} c\n"
            f"net total {money(locked['net_c_total'], 1)} c")
        self.query_one("#equity_label", Static).update(
            f"{len(perf['equity'])} trades   now {money(perf['equity'][-1] if perf['equity'] else 0.0, 1)} c")
        self.query_one("#equity", Sparkline).data = perf["equity"] or [0.0]
        self.show_verdict(a)
        self.show_history(a)
        self.show_tearsheet(a)
        self.load_disc()

    def show_verdict(self, a: dict) -> None:
        perf, expected = a["performance"], a["expected"]
        if perf["trades"] < 10:
            words = f"too early to judge: {perf['trades']} closed forward trades (needs ~30 to tell +3 c from 0)"
        else:
            gap = perf["net_per_trade_c"] - expected["net_per_trade_c"]
            words = (f"net per trade {perf['net_per_trade_c']:+.2f} c vs expected {expected['net_per_trade_c']:+.2f} "
                     f"({'in line' if abs(gap) < 2 else 'BELOW' if gap < 0 else 'above'})")
        self.query_one("#verdict", Static).update(f"[b reverse {AMBER}] VERDICT SO FAR [/]  {words}  ·  "
                                                  f"{len(a['positions'])} open")

    def show_history(self, a: dict) -> None:
        table = self.query_one("#history_table", DataTable)
        table.clear()
        for row in a["history"]:
            table.add_row(row["entry_time"][:16], row["side"], f"#{row['entry_number']}", f"{row['z_at_signal']:+.2f}",
                          f"{row['entry_price']:.2f}", row["exit_time"][:16], row["exit_reason"], money(row["net_c"], 1),
                          "agree" if row["slow_agrees"] else "disagree")

    def show_tearsheet(self, a: dict) -> None:
        perf, expected = a["performance"], a["expected"]
        rows = [("closed trades", f"{perf['trades']}", "-"),
                ("trades a week", f"{perf['per_week']:.2f}", f"{expected['per_week']:.2f}"),
                ("net per trade (c)", money(perf.get("net_per_trade_c")), f"{expected['net_per_trade_c']:+.2f}"),
                ("win rate", number(perf.get("hit_rate"), "{:.0%}"), f"{expected['hit_rate']:.0%}"),
                ("average winner (c)", money(perf.get("avg_winner_c"), 1), f"{expected['avg_winner_c']:+.1f}"),
                ("average loser (c)", money(perf.get("avg_loser_c"), 1), f"{expected['avg_loser_c']:+.1f}"),
                ("share exiting at fair value", number(perf.get("share_fair_value"), "{:.0%}"), f"{expected['share_fair_value']:.0%}"),
                ("net total (c)", money(perf["net_c_total"], 1), "-"),
                ("max drawdown (c)", money(perf["max_dd_c"], 1), "dev. worst -170 over 5 years")]
        table = self.query_one("#tearsheet_table", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)
        self.query_one("#tearsheet_note", Static).update(
            f"\nForward = closed trades exiting from {engine.FORWARD_START:%Y-%m-%d}, 1 contract each, after costs.\n"
            "Expectation = walk-forward picks 2021-2025 (runs/20261117-114551_validation).\n"
            "Press [b]t[/] for the equity chart (runs/live/equity.png).")

    def load_events(self) -> None:
        log = self.query_one("#events", RichLog)
        log.clear()
        if engine.EVENTS_FILE.exists():
            for line in engine.EVENTS_FILE.read_text().splitlines()[-40:]:
                event = json.loads(line)
                colour = RED if "FAILED" in event["event"] else GREEN if "entry" in event["event"] else "#c9d1d9"
                log.write(f"[{MUTED}]{event['time'][5:16]}[/] [{colour}]{event['event']}[/]")

    # ---- my own decisions, logged next to the rule's
    def action_log_action(self) -> None:
        def save(text: str) -> None:
            if not text or self.state is None:
                return
            a = self.state["anchored"]
            record = {"time": str(engine.now_utc().floor("s")), "wheat": a["wheat"], "z": a["z"],
                      "open_positions": len(a["positions"]), "text": text}
            with engine.ACTIONS_FILE.open("a") as file:
                file.write(json.dumps(record) + "\n")
            self.load_my_log()
        self.push_screen(LogAction(), save)

    def load_my_log(self) -> None:
        table = self.query_one("#mylog_table", DataTable)
        table.clear()
        if engine.ACTIONS_FILE.exists():
            for line in reversed(engine.ACTIONS_FILE.read_text().splitlines()):
                record = json.loads(line)
                table.add_row(record["time"][:16], f"{record['wheat']:.2f}", number(record["z"], "{:+.2f}"),
                              str(record["open_positions"]), record["text"])

    # ---- my discretionary trades, next to the model's
    def action_disc_entry(self) -> None:
        if self.state is None:
            return
        state = self.state
        existing = engine.disc_trades(state)
        next_id = max((t["id"] for t in existing), default=0) + 1

        def check(values: dict):
            market = {"w": "wheat", "b": "beans"}.get(values["market"][:1].lower())
            side = {"l": "long", "s": "short"}.get(values["side"][:1].lower())
            if market is None or side is None:
                return "market must be wheat or beans, side long or short"
            near = state["anchored"]["wheat"] if market == "wheat" else state.get("beans", {}).get("price")
            try:
                price, time = parse_price(values["price"], near), parse_time(values["time"])
            except ValueError as error:
                return f"{error}"
            if time > engine.now_utc():
                return "time is in the future"
            return {"event": "open", "id": next_id, "market": market, "side": side, "entry_time": str(time),
                    "entry_price": price, "note": values["note"]}

        self.push_screen(Form(f"Disc entry #{next_id}", [("market", "wheat / beans"), ("side", "long / short"),
                                                         ("price", "my fill, cents"), ("time", "UTC, blank = now, or 14:30"),
                                                         ("note", "why (optional)")], check), self.save_disc)

    def action_disc_exit(self) -> None:
        if self.state is None:
            return
        state = self.state
        open_trades = {t["id"]: t for t in engine.disc_trades(state) if t["open"]}

        def check(values: dict):
            try:
                trade = open_trades[int(values["id"])]
            except (ValueError, KeyError):
                return f"id must be one of the open disc trades: {sorted(open_trades) or 'none open'}"
            near = state["anchored"]["wheat"] if trade["market"] == "wheat" else state.get("beans", {}).get("price")
            try:
                price, time = parse_price(values["price"], near), parse_time(values["time"])
            except ValueError as error:
                return f"{error}"
            if time > engine.now_utc() or time < pd.Timestamp(trade["entry_time"]):
                return "exit time must be after the entry and not in the future"
            return {"event": "close", "id": trade["id"], "exit_time": str(time), "exit_price": price}

        self.push_screen(Form(f"Disc exit (open: {sorted(open_trades) or 'none'})",
                              [("id", "which disc trade"), ("price", "my exit fill, cents"),
                               ("time", "UTC, blank = now, or 14:30")], check), self.save_disc)

    def save_disc(self, record: dict | None) -> None:
        if record:
            engine.log_disc(record)
            self.load_disc()
            self.action_tab("disc")

    def load_disc(self) -> None:
        table = self.query_one("#disc_table", DataTable)
        table.clear()
        if self.state is None:
            return
        for row in reversed(engine.disc_trades(self.state)):
            model = row["model"] or {}
            model_exit = model.get("exit_price")
            table.add_row(str(row["id"]), row["market"], row["side"], row["entry_time"][5:16], f"{row['entry_price']:.2f}",
                          model.get("entry_time", "")[5:16] or ("no model trade" if row["market"] == "wheat" else "no model"),
                          number(model.get("entry_price")), money(row["entry_edge_c"], 1),
                          "open" if row["open"] else f"{row['exit_price']:.2f}",
                          ("open" if model and model_exit is None else number(model_exit)),
                          money(row["exit_edge_c"], 1), money(row["my_c"], 1), money(row["model_c"], 1), row["note"])

    def action_open_chart(self) -> None:
        path = engine.LIVE_DIR / "equity.png"
        if path.exists():
            os.startfile(path)                                   # Windows: opens in the default image viewer

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab


if __name__ == "__main__":
    Dashboard().run()
