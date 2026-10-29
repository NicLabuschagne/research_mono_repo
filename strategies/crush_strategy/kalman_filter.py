"""
kalman_filter.py
----------------
Model the beans-vs-products relationship with a Kalman filter.

    log ZS_t = alpha_t + beta_meal_t * log ZM_t + beta_oil_t * log ZL_t + noise

The coefficients (the "state") follow a random walk, so the hedge ratios can drift.
Each bar the filter:
  predict : keeps the state, but grows its uncertainty P by Q (how much drift we allow)
  update  : compares predicted log ZS with the actual value and moves the state
            toward it by the Kalman gain  K = P x / (x' P x + R)

Q is set as a drift-to-noise ratio (how far the fitted value may drift per bar,
relative to the observation noise R). Raw "delta" values are scale-dependent:
with log prices even delta = 1e-5 lets the filter re-fit every bar.

Frozen hedge: the residual in each session uses the state at the PREVIOUS
session's close. Refitting inside a session would pull the residual back to zero
and fake mean reversion that a trade cannot capture.

Two noise settings (config.yaml -> kalman.method)
  hand_set : the original. Q = drift-to-noise ratio x R (same for every coefficient),
             R = variance of the warm-up OLS residual.
  mle      : structure estimated by maximum likelihood (kalman_calibration.py): one drift
             variance per coefficient on centred regressors, R per session type, extra
             drift at session opens. `speed` scales every drift variance; it was chosen
             on the reversion of the residual (kalman_speed_sweep.py), not on forecasting.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DRIFT_TO_NOISE_PER_15_MINUTES = 0.1   # adaptation speed, defined per unit of time
WARMUP_DAYS = 20                      # OLS on the first 20 days gives the start state and R


@dataclass(frozen=True)
class KalmanConfig:
    method: str = "hand_set"          # hand_set | mle
    speed: float = 1.0                # mle only: multiplies q_alpha, q_meal, q_oil
    hand_set_speed: float = 1.0       # hand_set only: multiplies the drift-to-noise ratio
    q_alpha: float = 0.0              # mle drift variances per bar (centred regressors)
    q_meal: float = 0.0
    q_oil: float = 0.0
    r_evening: float = 0.0            # mle observation noise per session type
    r_day: float = 0.0
    open_mult: float = 1.0            # drift multiplier on a session's first bar


def design_matrix(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y = log beans, X = [1, log meal, log oil]."""
    y = np.log(bars["ZS_adj_close"].values)
    X = np.column_stack([np.ones(len(bars)), np.log(bars["ZM_adj_close"].values), np.log(bars["ZL_adj_close"].values)])
    return y, X


def run_kalman_filter(bars: pd.DataFrame, bar_minutes: int, kalman: KalmanConfig = KalmanConfig()) -> pd.DataFrame:
    """Filtered state (alpha, beta_meal, beta_oil) after each bar."""
    if kalman.method == "mle":
        return run_mle_filter(bars, kalman)
    if kalman.method != "hand_set":
        raise ValueError(f"unknown kalman.method {kalman.method!r}")
    y, X = design_matrix(bars)

    # warm-up: plain OLS on the first days sets the starting state and the noise level R
    warmup_dates = np.sort(bars["trading_date"].unique())[:WARMUP_DAYS]
    warm = bars["trading_date"].isin(warmup_dates).values
    state = np.linalg.lstsq(X[warm], y[warm], rcond=None)[0]
    observation_noise = np.var(y[warm] - X[warm] @ state)

    # Q chosen so the fitted value drifts by drift_to_noise * R per bar for a typical x
    drift_to_noise = DRIFT_TO_NOISE_PER_15_MINUTES * bar_minutes / 15 * kalman.hand_set_speed   # per unit of time
    typical_xx = np.mean(np.sum(X[warm] ** 2, axis=1))
    process_noise = drift_to_noise * observation_noise / typical_xx * np.eye(3)

    state_covariance = np.eye(3)
    states = np.empty((len(bars), 3))
    for t in range(len(bars)):
        # predict: state unchanged, uncertainty grows
        state_covariance = state_covariance + process_noise
        # update: move toward the observation in proportion to the gain
        x = X[t]
        forecast_error = y[t] - x @ state
        forecast_variance = x @ state_covariance @ x + observation_noise
        gain = state_covariance @ x / forecast_variance
        state = state + gain * forecast_error
        state_covariance = state_covariance - np.outer(gain, x) @ state_covariance
        states[t] = state
    return pd.DataFrame(states, index=bars.index, columns=["alpha", "beta_meal", "beta_oil"])


def run_mle_filter(bars: pd.DataFrame, kalman: KalmanConfig) -> pd.DataFrame:
    """MLE noise structure. Regressors are centred on the warm-up means (the intercept and the
    betas are less collinear that way); the intercept is un-centred at the end, so the
    fitted value and the residual mean the same as in the hand-set filter."""
    y, X = design_matrix(bars)
    warmup_dates = np.sort(bars["trading_date"].unique())[:WARMUP_DAYS]
    warm = bars["trading_date"].isin(warmup_dates).values
    centre = X[warm, 1:].mean(axis=0)
    X = np.column_stack([X[:, 0], X[:, 1:] - centre])
    state = np.linalg.lstsq(X[warm], y[warm], rcond=None)[0]

    process_noise = np.diag([kalman.q_alpha, kalman.q_meal, kalman.q_oil]) * kalman.speed
    observation_noise = np.where(bars["session_type"].values == "evening", kalman.r_evening, kalman.r_day)
    session = bars["session_id"].values
    opens = np.r_[False, session[1:] != session[:-1]]

    state_covariance = np.eye(3)
    states = np.empty((len(bars), 3))
    for t in range(len(bars)):
        if t > 0:
            state_covariance = state_covariance + process_noise * (kalman.open_mult if opens[t] else 1.0)
        x = X[t]
        forecast_error = y[t] - x @ state
        forecast_variance = x @ state_covariance @ x + observation_noise[t]
        gain = state_covariance @ x / forecast_variance
        state = state + gain * forecast_error
        state_covariance = state_covariance - np.outer(gain, x) @ state_covariance
        states[t] = state
    states[:, 0] -= states[:, 1:] @ centre      # back to the un-centred intercept
    return pd.DataFrame(states, index=bars.index, columns=["alpha", "beta_meal", "beta_oil"])


def freeze_at_previous_session_close(bars: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """For every bar, the state as of the last bar of the previous session."""
    session_close_state = states.groupby(bars["session_id"]).last().shift(1)
    frozen = session_close_state.reindex(bars["session_id"])
    frozen.index = bars.index
    return frozen


def add_kalman_residual(bars: pd.DataFrame, bar_minutes: int, kalman: KalmanConfig = KalmanConfig()) -> pd.DataFrame:
    """Add frozen hedge ratios and the residual (log beans minus the hedged products)."""
    states = run_kalman_filter(bars, bar_minutes, kalman)
    frozen = freeze_at_previous_session_close(bars, states)
    y, X = design_matrix(bars)
    residual = y - np.sum(X * frozen.values, axis=1)
    return bars.join(frozen.add_prefix("frozen_")).assign(residual=residual)
