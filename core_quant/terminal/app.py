"""
app.py
------
Terminal PREVIEW: every strategy's signal cards on one screen, grouped two ways.

    python -m core_quant.terminal.app

  Markets [1]     one grid, a header row per market (launchpad): every signal on that market, its gap, context, signal
  Strategies [2]  one panel per strategy: the same cards plus that strategy's own lines (positions, forward results)
  History [3]     each strategy's trades          Tear sheet [4]  forward vs expectation + equity
  My log [5]      what I would do (shared with the old wheat dashboard's log)

It re-reads everything every POLL_MINUTES; the wheat numbers come from the wheat-corn engine's state.json, so keep
the wheat dashboard (or engine) running to keep them fresh. The only write is "l" (my log).
Keys: r refresh · l log my action · t equity chart · 1-5 tabs · q quit
"""
from collections import defaultdict

import pandas as pd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Sparkline, Static, TabbedContent, TabPane

from core_quant.terminal.cards import SignalCard

POLL_MINUTES = 15
GREEN, RED, AMBER, MUTED = "#3fb950", "#f85149", "#d29922", "#8b949e"


def curve_placeholders() -> list[SignalCard]:
    """shortcut: rows only, no model - replace with the curve research's own cards once it has a signal."""
    return [SignalCard(market, "Curve structure", "-", None, None, None, 0.25, None, "not built", "-",
                       instrument=spread, note="placeholder: curve research in progress")
            for market, spread in [("Wheat", "ZW calendar"), ("Beans", "ZS calendar")]]


def producers() -> list:
    """Each entry returns a list of cards. Add a strategy here."""
    from strategies.wheat_corn_lag.live.cards import wheat_cards
    return [wheat_cards, curve_placeholders]


def detail_producers() -> list:
    """Each entry returns a list of StrategyDetail (History / Tear sheet tabs)."""
    from strategies.wheat_corn_lag.live.cards import wheat_details
    return [wheat_details]


def journal():
    """(read rows, write one entry). shortcut: one shared log, the wheat-corn file; give it its own file when a
    second strategy needs logging."""
    from strategies.wheat_corn_lag.live.cards import log_action, my_log
    return my_log, log_action


def cell(column: str, value) -> str:
    """A History / Tear sheet value: '-' if missing, columns ending in ' c' signed and coloured, floats to 2 dp."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return f"[{MUTED}]-[/]"
    if column.endswith(" c") and isinstance(value, (int, float)):
        return signed(value, "{:+.1f}")
    if isinstance(value, float):
        return f"{value:.0%}" if column in ("win rate", "share exiting at fair value") else f"{value:.2f}"
    return str(value)


def fmt(value, pattern: str) -> str:
    return f"[{MUTED}]-[/]" if value is None or pd.isna(value) else pattern.format(value)


def signal_text(card: SignalCard) -> str:
    if card.signal in ("LONG", "SHORT"):                  # fired: a filled cell, so it stands out on the grid
        return f"[b white on {GREEN if card.signal == 'LONG' else RED}] {card.signal} [/]"
    return f"[{MUTED if card.signal == 'flat' else AMBER}]{card.signal}[/]"


def signed(value, pattern: str) -> str:
    """The gap in its direction's colour: green above fair value, red below."""
    if value is None or pd.isna(value):
        return f"[{MUTED}]-[/]"
    return f"[{GREEN if value > 0 else RED if value < 0 else MUTED}]{pattern.format(value)}[/]"


def context_text(card: SignalCard) -> str:
    if card.context is None:
        return f"[{MUTED}]-[/]"
    colour = RED if card.context >= 0.95 else AMBER if card.context >= 0.80 else MUTED
    return f"[{colour}]p{card.context * 100:.0f}[/]"


def row(card: SignalCard, first: str) -> tuple:
    return (first, card.instrument, card.line, fmt(card.price, "{:.2f}"), fmt(card.fair_value, "{:.2f}"),
            signed(card.gap, "{:+.2f}"), signed(card.gap_ticks, "{:+.0f}"), context_text(card), signal_text(card),
            card.as_of[5:16] if len(card.as_of) >= 16 else card.as_of, card.note)


COLUMNS = ("", "instrument", "line", "price", "fv", "gap", "ticks", "ctx", "signal", "as of UTC", "note")


class Terminal(App):
    TITLE = "research terminal · preview"
    CSS = """
    Screen { background: #0d1117; color: #c9d1d9; }
    #status { height: 1; padding: 0 1; color: #8b949e; }
    .group { border: round #30363d; border-title-color: #58a6ff; height: auto; padding: 0 1; margin: 0 0 1 0; }
    DataTable { height: auto; }
    .lines { height: auto; color: #8b949e; }
    Sparkline { height: 4; margin: 0 0 1 0; }
    #dialog { width: 80; height: 7; border: heavy #58a6ff; background: #161b22; padding: 1; }
    LogAction { align: center middle; }
    """
    BINDINGS = [Binding("r", "refresh", "refresh"), Binding("l", "log_action", "log my action"),
                Binding("t", "open_chart", "equity chart"), Binding("1", "tab('markets')", "markets"),
                Binding("2", "tab('strategies')", "strategies"), Binding("3", "tab('history')", "history"),
                Binding("4", "tab('tearsheet')", "tear sheet"), Binding("5", "tab('mylog')", "my log"),
                Binding("q", "quit", "quit")]

    def __init__(self):
        super().__init__()
        self.charts: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("loading cards ...", id="status")
        with TabbedContent(initial="markets"):
            with TabPane("Markets [1]", id="markets"):
                yield VerticalScroll(id="markets_body")
            with TabPane("Strategies [2]", id="strategies"):
                yield VerticalScroll(id="strategies_body")
            with TabPane("History [3]", id="history"):
                yield VerticalScroll(id="history_body")
            with TabPane("Tear sheet [4]", id="tearsheet"):
                yield VerticalScroll(id="tearsheet_body")
            with TabPane("My log [5]", id="mylog"):
                yield DataTable(id="mylog_table", zebra_stripes=True, cursor_type="none")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()
        self.set_interval(POLL_MINUTES * 60, self.action_refresh)

    def action_refresh(self) -> None:
        self.query_one("#status", Static).update(f"[{AMBER}]● reading cards ...[/]")
        self.run_worker(self.load, thread=True, exclusive=True)

    def load(self) -> None:
        cards, details, errors = [], [], []
        for produce, into in [(p, cards) for p in producers()] + [(p, details) for p in detail_producers()]:
            try:
                into += produce()
            except Exception as error:                  # one broken strategy must not blank the screen
                errors.append(f"{produce.__name__}: {error}")
        self.call_from_thread(self.show, cards, errors, details)

    async def show(self, cards: list[SignalCard], errors: list[str], details: list | None = None) -> None:
        await self.show_details(details or [])
        self.show_log()
        now = pd.Timestamp.now(tz="UTC")
        problems = f"   [{RED}]{' · '.join(errors)}[/]" if errors else ""
        self.query_one("#status", Static).update(f"[{GREEN}]●[/] {len(cards)} cards · read {now:%H:%M} UTC · "
                                                 f"next in ≤ {POLL_MINUTES} min{problems}")
        by_market, by_strategy = defaultdict(list), defaultdict(list)
        for card in cards:
            by_market[card.market].append(card)
            by_strategy[card.strategy].append(card)
        await self.launchpad(by_market)
        await self.fill("#strategies_body", by_strategy, first=lambda c: c.market, lines=True)

    async def launchpad(self, by_market: dict) -> None:
        """Markets: ONE dense grid, a header row per market, one row per signal (like a Bloomberg monitor)."""
        body = self.query_one("#markets_body", VerticalScroll)
        await body.remove_children()
        table = DataTable(zebra_stripes=False, cursor_type="none")
        table.add_columns(*COLUMNS)
        for market, group in by_market.items():
            table.add_row(f"[b #58a6ff]── {market.upper()} ──[/]", *[""] * (len(COLUMNS) - 1))
            for card in group:
                table.add_row(*row(card, f"  {card.strategy}"))
        await body.mount(table)

    async def fill(self, body_id: str, groups: dict, first, lines: bool) -> None:
        body = self.query_one(body_id, VerticalScroll)
        await body.remove_children()
        for name, group in groups.items():
            table = DataTable(zebra_stripes=True, cursor_type="none")
            table.add_columns(*COLUMNS)
            for card in group:
                table.add_row(*row(card, first(card)))
            children = [table]
            if lines:
                extra = {key: value for card in group for key, value in card.extra.items()}
                if extra:
                    children.append(Static("\n".join(f"{key}: {value}" for key, value in extra.items()), classes="lines"))
            panel = VerticalScroll(*children, classes="group")
            panel.border_title = name.upper()
            await body.mount(panel)

    async def show_details(self, details: list) -> None:
        """History: each strategy's trades. Tear sheet: forward vs expectation, plus its equity as a sparkline."""
        history, tearsheet = self.query_one("#history_body", VerticalScroll), self.query_one("#tearsheet_body", VerticalScroll)
        await history.remove_children()
        await tearsheet.remove_children()
        self.charts = [d.chart for d in details if d.chart]
        for d in details:
            table = DataTable(zebra_stripes=True, cursor_type="none")
            table.add_columns(*d.history_columns)
            for values in d.history_rows:
                table.add_row(*[cell(column, value) for column, value in zip(d.history_columns, values)])
            if not d.history_rows:
                table.add_row(f"[{MUTED}]no trades yet[/]", *[""] * (len(d.history_columns) - 1))
            panel = VerticalScroll(table, classes="group")
            panel.border_title = f"{d.strategy.upper()} · {len(d.history_rows)} trades"
            await history.mount(panel)

            sheet = DataTable(cursor_type="none")
            sheet.add_columns("measure", "forward", "expected")
            for measure, value, expected in d.tearsheet:
                sheet.add_row(measure, cell(measure, value), cell(measure, expected))
            equity = d.equity or [0.0]
            children = [sheet, Static(f"equity, closed trades: {len(d.equity)} · now {cell('net c', equity[-1])} c"
                                      + ("   (t = open the chart)" if d.chart else ""), classes="lines"),
                        Sparkline(equity)]
            panel = VerticalScroll(*children, classes="group")
            panel.border_title = d.strategy.upper()
            await tearsheet.mount(panel)

    def show_log(self) -> None:
        table = self.query_one("#mylog_table", DataTable)
        if not table.columns:
            table.add_columns("time (UTC)", "wheat", "z", "open", "what I would do")
        table.clear()
        try:
            rows = journal()[0]()
        except Exception as error:
            rows = [[f"[{RED}]log unreadable: {error}[/]", None, None, None, ""]]
        for values in rows:
            table.add_row(*[cell(column, value) for column, value in zip(["t", "wheat", "z", "open", "text"], values)])
        if not rows:
            table.add_row(f"[{MUTED}]nothing logged yet: press l[/]", "", "", "", "")

    def action_log_action(self) -> None:
        from strategies.wheat_corn_lag.live.dashboard import LogAction        # the old dashboard's input box

        def save(text: str) -> None:
            if text:
                journal()[1](text)
                self.show_log()
                self.action_tab("mylog")
        self.push_screen(LogAction(), save)

    def action_open_chart(self) -> None:
        import os
        for path in self.charts:
            if os.path.exists(path):
                os.startfile(path)                       # Windows: the default image viewer
                return

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab


if __name__ == "__main__":
    Terminal().run()
