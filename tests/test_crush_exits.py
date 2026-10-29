import numpy as np
import pandas as pd

from strategies.crush_strategy.strategy import StrategyConfig, simulate_trades


def flat_bars(sessions: int = 3, bars_per_session: int = 40) -> pd.DataFrame:
    """Constant prices (no barrier is ever hit), so only the time stop decides the exit."""
    n = sessions * bars_per_session
    index = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp("2021-01-04 01:05", tz="UTC") + pd.Timedelta(days=s), periods=bars_per_session, freq="5min")
        for s in range(sessions)]))
    bars = pd.DataFrame({"session_id": np.repeat(np.arange(sessions), bars_per_session),
                         "trading_date": index.tz_localize(None).normalize(), "z_std": 0.01,
                         "frozen_alpha": 0.0, "frozen_beta_meal": 0.6, "frozen_beta_oil": 0.35}, index=index)
    for leg, price in [("ZS", 1200.0), ("ZM", 400.0), ("ZL", 50.0)]:
        bars[f"{leg}_adj_open"] = bars[f"{leg}_adj_close"] = price
    bars["ZS_adj_close"] = 1200.0
    return bars


def signals_at(positions) -> pd.DataFrame:
    return pd.DataFrame({"signal_position": positions, "entry_z": 2.0, "direction": -1.0})


def test_session_time_stop_never_leaves_the_session():
    bars = flat_bars()
    trades = simulate_trades(bars, signals_at([10, 50]), StrategyConfig(holding_hours=12, min_hours_left=0.0))
    session = bars["session_id"].values
    assert (session[trades["exit_bar"]] == session[trades["entry_bar"]]).all()
    assert list(trades["exit_bar"]) == [39, 79]                    # last bar of each session


def test_session_time_stop_respects_the_clock_cap():
    trades = simulate_trades(flat_bars(), signals_at([10]), StrategyConfig(holding_hours=1, min_hours_left=0.0))
    assert trades["exit_bar"].iloc[0] == 11 + 12                   # entry bar + 60 minutes of 5-minute bars


def test_min_hours_left_skips_late_signals():
    bars = flat_bars()                                             # sessions of 40 bars = 3h15 from first to last bar
    late, early = 35, 50                                           # 20 min vs 3h05 before their session's last bar
    trades = simulate_trades(bars, signals_at([late, early]),
                             StrategyConfig(holding_hours=12, min_hours_left=1.0))
    assert list(trades["entry_bar"]) == [early + 1]


def ramp_bars() -> pd.DataFrame:
    """Beans rise 1 tick-ish per bar, products flat: the residual climbs steadily."""
    bars = flat_bars(sessions=1, bars_per_session=60)
    bars["ZS_adj_open"] = bars["ZS_adj_close"] = 1200.0 * np.exp(np.arange(60) * 1e-4)
    bars["z_std"] = 1.0                                            # barriers never hit: time exit only
    return bars


def test_entry_delay_shifts_the_fill():
    trades = simulate_trades(ramp_bars(), signals_at([5]), StrategyConfig(min_hours_left=0.0, entry_delay_minutes=15))
    assert trades["entry_bar"].iloc[0] == 6 + 3


def test_entry_window_averages_the_fill_price():
    bars = ramp_bars()
    one = simulate_trades(bars, signals_at([5]), StrategyConfig(min_hours_left=0.0))
    twap = simulate_trades(bars, signals_at([5]), StrategyConfig(min_hours_left=0.0, entry_window_minutes=20))
    assert twap["entry_slices"].iloc[0] == 4 and one["exit_bar"].iloc[0] == twap["exit_bar"].iloc[0]
    # short beans into a rising residual: averaging in at higher prices loses less (log, x bean price at the signal bar)
    expected = -(np.log(bars["ZS_adj_close"].iloc[59]) - np.log(bars["ZS_adj_open"].iloc[6:10]).mean()) * bars["ZS_adj_close"].iloc[5]
    assert np.isclose(twap["gross_cents"].iloc[0], expected)
