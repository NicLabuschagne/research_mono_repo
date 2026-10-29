import numpy as np
import pandas as pd

from strategies.crush_strategy.features import add_session_zscore
from strategies.crush_strategy.kalman_calibration import CrushKalman
from strategies.crush_strategy.kalman_filter import KalmanConfig, run_kalman_filter


def synthetic_bars(n_days: int = 30, bars_per_session: int = 20, seed: int = 0) -> pd.DataFrame:
    """Two sessions a day (evening, day); beans = 0.6 meal + 0.35 oil + noise, in logs."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for k, session_type in enumerate(["evening", "day"]):
            for b in range(bars_per_session):
                rows.append(dict(trading_date=pd.Timestamp("2021-01-04") + pd.Timedelta(days=day),
                                 session_type=session_type, session_id=2 * day + k, bar_number=b))
    bars = pd.DataFrame(rows)
    n = len(bars)
    log_meal = np.log(400) + np.cumsum(rng.normal(0, 0.002, n))
    log_oil = np.log(50) + np.cumsum(rng.normal(0, 0.003, n))
    log_beans = 2.4 + 0.6 * log_meal + 0.35 * log_oil + rng.normal(0, 0.001, n)
    bars["ZS_adj_close"], bars["ZM_adj_close"], bars["ZL_adj_close"] = np.exp(log_beans), np.exp(log_meal), np.exp(log_oil)
    bars.index = pd.date_range("2021-01-04", periods=n, freq="5min", tz="UTC")
    return bars


def test_mle_filter_matches_statsmodels_reference():
    bars = synthetic_bars()
    kalman = KalmanConfig(method="mle", speed=0.5, q_alpha=1e-8, q_meal=3e-8, q_oil=2e-8,
                          r_evening=1e-6, r_day=4e-6, open_mult=20.0)
    ours = run_kalman_filter(bars, 5, kalman)
    reference = CrushKalman(bars, "mle_diag", burn=0)
    params = np.array([kalman.q_alpha * kalman.speed, kalman.q_meal * kalman.speed, kalman.q_oil * kalman.speed,
                       kalman.r_evening, kalman.r_day, kalman.open_mult])
    states = reference.filter(params).filtered_state.T
    states[:, 0] -= states[:, 1:] @ reference.centre
    np.testing.assert_allclose(ours.values, states, atol=1e-9)


def test_hand_set_is_the_default():
    bars = synthetic_bars(seed=1)
    np.testing.assert_allclose(run_kalman_filter(bars, 5).values,
                               run_kalman_filter(bars, 5, KalmanConfig(method="hand_set")).values)


def test_pct20_z_is_uniform_when_residuals_are_iid():
    bars = synthetic_bars(n_days=60, seed=2)
    bars["residual"] = np.random.default_rng(3).standard_t(4, len(bars)) * 1e-3   # fat tails on purpose
    z = add_session_zscore(bars, method="pct20")
    z = z.loc[z["trading_date"] >= np.sort(z["trading_date"].unique())[20], "z"]
    assert abs((z.abs() >= 2).mean() - 0.0455) < 0.01     # the threshold means its normal tail probability
    roll = add_session_zscore(bars, method="roll20")["z"].loc[z.index]
    assert (roll.abs() >= 3).mean() > (z.abs() >= 3).mean()  # roll20 over-fires on fat tails
