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
    trades = simulate_trades(bars, signals_at([10, 50]), StrategyConfig(time_stop="session", holding_hours=12))
    session = bars["session_id"].values
    assert (session[trades["exit_bar"]] == session[trades["entry_bar"]]).all()
    assert list(trades["exit_bar"]) == [39, 79]                    # last bar of each session


def test_session_time_stop_respects_the_clock_cap():
    trades = simulate_trades(flat_bars(), signals_at([10]), StrategyConfig(time_stop="session", holding_hours=1))
    assert trades["exit_bar"].iloc[0] == 11 + 12                   # entry bar + 60 minutes of 5-minute bars


def test_bar_time_stop_still_counts_bars_across_sessions():
    trades = simulate_trades(flat_bars(), signals_at([10]), StrategyConfig(time_stop="bars", holding_hours=4))
    assert trades["exit_bar"].iloc[0] == 11 + 48                   # crosses into the next session, as before


def test_min_hours_left_skips_late_signals():
    bars = flat_bars()                                             # sessions of 40 bars = 3h15 from first to last bar
    late, early = 35, 50                                           # 20 min vs 3h05 before their session's last bar
    trades = simulate_trades(bars, signals_at([late, early]),
                             StrategyConfig(time_stop="session", holding_hours=12, min_hours_left=1.0))
    assert list(trades["entry_bar"]) == [early + 1]
