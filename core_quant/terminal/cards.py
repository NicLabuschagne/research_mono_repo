"""
cards.py
--------
The one record every strategy hands to the terminal: a SIGNAL CARD. The screen only groups cards (by market or by
strategy), so a new strategy = one function returning cards; the screen does not change. The same cards could feed
BQuant or a web page later.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SignalCard:
    market: str                    # "Wheat", "Beans", ...
    strategy: str                  # "Wheat-corn anchored", "Curve structure", ...
    line: str                      # "fast" / "slow": which fair value ("-" if the signal has only one)
    price: float | None            # last value of the INSTRUMENT, market units (cents; a spread for curve signals)
    fair_value: float | None       # None = no fair value (a signal that is not a fair value puts its value in gap)
    gap: float | None              # value - fair value, market units (+ = rich)
    tick: float                    # one tick in market units, so the gap can be shown in ticks
    context: float | None          # how unusual |gap| is: percentile vs the recent window (0..1)
    signal: str                    # "LONG", "SHORT", "flat", "not tradable", "no model"
    as_of: str                     # time of the last bar the card was computed on (UTC)
    instrument: str = ""           # what price / fair value / gap refer to: "ZW front", "ZW H27-K27", ...
    note: str = ""
    extra: dict = field(default_factory=dict)      # strategy-specific lines for the Strategies view

    @property
    def gap_ticks(self) -> float | None:
        return None if self.gap is None else self.gap / self.tick


@dataclass
class StrategyDetail:
    """What a strategy shows on the History / Tear sheet tabs. Raw values; the screen formats them
    (a column whose name ends in ' c' is shown signed and coloured)."""
    strategy: str
    history_columns: list[str]
    history_rows: list[list]
    tearsheet: list[tuple]                          # (measure, forward value, expectation or "-")
    equity: list[float]                             # cumulative net, closed trades in order
    chart: str | None = None                        # a saved equity chart to open with 't'


def gap_context(gaps: pd.Series, window_bars: int) -> float | None:
    """Percentile of the latest |gap| among the |gap| of the window_bars BEFORE it (0 = smallest, 1 = largest)."""
    values = np.abs(gaps.dropna().values)
    if len(values) < 2:
        return None
    history, now = values[-window_bars - 1:-1], values[-1]
    return float((history < now).mean())


if __name__ == "__main__":
    # self-check: the last value is ranked against the earlier ones only
    gaps = pd.Series([1.0, -2.0, 3.0, -4.0, 2.5])
    assert gap_context(gaps, 4) == 0.5                      # |2.5| beats 1 and 2, not 3 and 4
    assert gap_context(pd.Series([5.0, -10.0]), 10) == 1.0
    card = SignalCard("Wheat", "test", "fast", 600.0, 592.5, 7.5, 0.25, 0.9, "SHORT", "2026-11-18")
    assert card.gap_ticks == 30
    print("cards.py self-check OK")
