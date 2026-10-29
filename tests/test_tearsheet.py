import numpy as np
import pandas as pd

from core_quant.tearsheet import monthly_returns, plot_tear_sheet, tear_sheet


def small_case():
    dates = pd.bdate_range("2021-01-04", periods=6)
    daily = pd.DataFrame({"ret": [1.0, -2.0, 0.0, 3.0, -1.0, 1.0], "trades": [1, 1, 0, 1, 1, 1]}, index=dates)
    trades = pd.DataFrame({"exit_date": dates[[0, 1, 3, 4, 5]], "ret": [1.0, -2.0, 3.0, -1.0, 1.0],
                           "pnl": [2.0, -4.0, 6.0, -2.0, 2.0]})
    trades.attrs["pnl_unit"] = "c_per_bu"
    return daily, trades


def test_return_metrics_match_hand_calculation():
    daily, trades = small_case()
    s = tear_sheet(daily, trades, periods_per_year=252)
    r = daily["ret"]
    assert np.isclose(s["sharpe"], r.mean() / r.std() * np.sqrt(252))
    # equity 1, -1, -1, 2, 1, 2 -> worst drawdown -2 (from 1 to -1); underwater for 2 periods, then 1
    assert s["max_drawdown"] == -2.0 and s["longest_drawdown_periods"] == 2
    assert np.isclose(s["calmar"], r.mean() * 252 / 2.0)
    downside = np.sqrt(((np.minimum(r, 0)) ** 2).mean())
    assert np.isclose(s["sortino"], r.mean() / downside * np.sqrt(252))


def test_trade_metrics_match_hand_calculation():
    daily, trades = small_case()
    s = tear_sheet(daily, trades)
    assert s["trades"] == 5 and np.isclose(s["win_rate"], 60.0)
    assert np.isclose(s["expectancy"], 0.4) and np.isclose(s["profit_factor"], 5.0 / 3.0)
    assert np.isclose(s["payoff_ratio"], (5 / 3) / 1.5) and s["max_consecutive_losses"] == 1
    assert np.isclose(s["expectancy_c_per_bu"], 0.8)


def test_monthly_table_and_figure_build():
    daily, trades = small_case()
    assert monthly_returns(daily).loc[2021, 1] == 2.0
    fig = plot_tear_sheet(daily, trades, tear_sheet(daily, trades), title="test")
    assert len(fig.axes) >= 6
