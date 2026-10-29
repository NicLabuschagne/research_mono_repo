"""
kalman_calibration.py
---------------------
Step 1 of the recalibration: estimate the Kalman filter's noise levels by maximum
likelihood (MLE) on the development period, instead of setting them by hand.

Model (one row per 5-minute bar t)
    log ZS_t = alpha_t + beta_meal_t * log ZM_t + beta_oil_t * log ZL_t [+ u_t] + e_t
    coefficients : random walk, drift variance Q per bar
    e_t          : white observation noise, variance R
    u_t          : optional dislocation, u_t+1 = phi * u_t + eta (mean-reverting)

Why the dislocation state matters
    Without u, the only way to explain a residual that persists for hours is to move
    the coefficients, so MLE pushes Q up and the hedge starts chasing the very
    dislocations the strategy wants to trade. With u, the likelihood can separate
    slow drift in the relationship (Q) from transient dislocations (phi, eta).

Specs (nested, each adds to the previous)
    current         : production filter, hand-set Q = q * I and R (reproduced exactly)
    mle_isotropic   : same structure, q and R estimated
    mle_diag        : one drift variance per coefficient, R per session type,
                      extra drift at session opens (the relationship can reset overnight)
    mle_dislocation : mle_diag + u_t, with phi and eta per session type

Regressors are centred on the warm-up means for the MLE specs, so the intercept and
the betas are less collinear; the fitted value (and so the residual) is unaffected.
Scoring is P&L-free: likelihood on development, likelihood on a later window with
params fitted on an earlier one, and hedge / residual diagnostics.

    python -m strategies.crush_strategy.kalman_calibration
"""
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel

from core_quant.config import ROOT, save_run
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import add_session_zscore
from strategies.crush_strategy.kalman_filter import (
    DRIFT_TO_NOISE_PER_15_MINUTES, WARMUP_DAYS, freeze_at_previous_session_close, run_kalman_filter)
from strategies.crush_strategy.strategy import STRATEGY_DIR, load

BAR_MINUTES = 5
FIT_SPLIT = "2023-01-01"          # out-of-sample model check: fit before, score from here to holdout_start
SPECS = ["current", "mle_isotropic", "mle_diag", "mle_dislocation"]


class CrushKalman(MLEModel):
    """Time-varying-coefficient regression of log beans on log meal and log oil."""

    def __init__(self, bars: pd.DataFrame, spec: str, burn: int):
        self.spec = spec
        self.has_u = spec == "mle_dislocation"
        k = 4 if self.has_u else 3
        y = np.log(bars["ZS_adj_close"].values)
        x = np.column_stack([np.log(bars["ZM_adj_close"].values), np.log(bars["ZL_adj_close"].values)])
        warm = bars["trading_date"].isin(np.sort(bars["trading_date"].unique())[:WARMUP_DAYS]).values
        self.centre = np.zeros(2) if spec == "current" else x[warm].mean(axis=0)
        X = np.column_stack([np.ones(len(y)), x - self.centre])
        super().__init__(y, k_states=k, k_posdef=k)

        design = np.zeros((1, k, len(y)))
        design[0, :3] = X.T
        if self.has_u:
            design[0, 3] = 1.0
        self["design"] = design
        self["selection"] = np.eye(k)
        self.evening = (bars["session_type"].values == "evening")
        session = bars["session_id"].values
        # transition t -> t+1 uses the session of bar t+1; True where bar t+1 opens a session
        self.next_opens = np.r_[session[1:] != session[:-1], False]
        self.next_evening = np.r_[self.evening[1:], self.evening[-1]]

        # start state and R from OLS on the warm-up, as production does
        state0 = np.linalg.lstsq(X[warm], y[warm], rcond=None)[0]
        self.r0 = np.var(y[warm] - X[warm] @ state0)
        self.q0 = DRIFT_TO_NOISE_PER_15_MINUTES * BAR_MINUTES / 15 * self.r0 / np.mean(np.sum(X[warm] ** 2, axis=1))
        a1 = np.r_[state0, 0.0] if self.has_u else state0
        p1 = np.eye(k) + (self.q0 * np.eye(k) if spec == "current" else 0)   # production: P0 = I, then + Q
        self.ssm.initialize_known(a1, p1)
        self.ssm.loglikelihood_burn = burn

    # ---- parameters: log variances, logit phi
    @property
    def param_names(self):
        names = {"current": [], "mle_isotropic": ["q", "R"],
                 "mle_diag": ["q_alpha", "q_meal", "q_oil", "R_evening", "R_day", "open_mult"]}
        names["mle_dislocation"] = names["mle_diag"] + ["phi_evening", "phi_day", "eta_evening", "eta_day"]
        return names[self.spec]

    @property
    def start_params(self):
        q, r = self.q0, self.r0
        return {"current": np.array([]), "mle_isotropic": np.array([q, r]),
                "mle_diag": np.array([q, q, q, r, r, 1.0]),
                "mle_dislocation": np.array([q / 10, q / 10, q / 10, r / 2, r / 2, 1.0, 0.95, 0.95, r / 20, r / 20]),
                }[self.spec]

    def transform_params(self, unconstrained):
        p = np.exp(unconstrained)
        if self.has_u:
            p[6:8] = 1 / (1 + np.exp(-unconstrained[6:8]))
        return p

    def untransform_params(self, constrained):
        u = np.log(constrained)
        if self.has_u:
            u[6:8] = np.log(constrained[6:8] / (1 - constrained[6:8]))
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        n, k = self.nobs, self.k_states
        if self.spec == "current":
            q_diag, r_obs, open_mult = np.full(3, self.q0), np.full(n, self.r0), 1.0
        elif self.spec == "mle_isotropic":
            q_diag, r_obs, open_mult = np.full(3, params[0]), np.full(n, params[1]), 1.0
        else:
            q_diag, open_mult = params[:3], params[5]
            r_obs = np.where(self.evening, params[3], params[4])
        self["obs_cov"] = r_obs[None, None, :]

        state_cov = np.zeros((k, k, n))
        mult = np.where(self.next_opens, open_mult, 1.0)
        for i in range(3):
            state_cov[i, i] = q_diag[i] * mult
        transition = np.eye(k)
        if self.has_u:
            transition = np.repeat(np.eye(k)[:, :, None], n, axis=2)
            transition[3, 3] = np.where(self.next_evening, params[6], params[7])
            state_cov[3, 3] = np.where(self.next_evening, params[8], params[9])
        self["state_cov"] = state_cov
        self["transition"] = transition


def fit(bars: pd.DataFrame, spec: str, burn: int, start=None) -> tuple[CrushKalman, np.ndarray]:
    model = CrushKalman(bars, spec, burn)
    if spec == "current":
        return model, np.array([])
    result = model.fit(start_params=start if start is not None else model.start_params,
                       method="lbfgs", maxiter=300, disp=False, approx_complex_step=False, cov_type="none")
    return model, result.params


def frozen_residual(bars: pd.DataFrame, model: CrushKalman, params: np.ndarray) -> pd.DataFrame:
    """Production residual definition: log beans minus the hedge frozen at the previous session close."""
    states = model.filter(params).filtered_state.T[:, :3]
    states = pd.DataFrame(states, index=bars.index, columns=["alpha", "beta_meal", "beta_oil"])
    states["alpha"] -= states["beta_meal"] * model.centre[0] + states["beta_oil"] * model.centre[1]  # un-centre
    frozen = freeze_at_previous_session_close(bars, states)
    y = np.log(bars["ZS_adj_close"].values)
    X = np.column_stack([np.ones(len(bars)), np.log(bars["ZM_adj_close"].values), np.log(bars["ZL_adj_close"].values)])
    out = bars.join(frozen.add_prefix("frozen_")).assign(residual=y - np.sum(X * frozen.values, axis=1))
    return add_session_zscore(out)


def diagnostics(bars: pd.DataFrame, start: str, end: str) -> dict:
    """Hedge stability and residual behaviour on [start, end), P&L-free."""
    window = bars[(bars["trading_date"] >= pd.Timestamp(start)) & (bars["trading_date"] < pd.Timestamp(end))]
    per_session = window.groupby("session_id")[["frozen_beta_meal", "frozen_beta_oil"]].first()
    out = {"beta_meal_mean": per_session["frozen_beta_meal"].mean(), "beta_oil_mean": per_session["frozen_beta_oil"].mean(),
           "beta_meal_session_change_sd": per_session["frozen_beta_meal"].diff().std(),
           "beta_oil_session_change_sd": per_session["frozen_beta_oil"].diff().std()}
    for session in ["evening", "day"]:
        s = window[window["session_type"] == session]
        # residual at session close vs open: how far it wanders inside one session with the hedge fixed
        span = s.groupby("session_id")["residual"].agg(lambda r: r.iloc[-1] - r.iloc[0])
        out[f"{session}_resid_bp_sd"] = s["residual"].std() * 1e4
        out[f"{session}_in_session_move_bp_sd"] = span.std() * 1e4
        out[f"{session}_z_sd"] = s["z"].std()          # production calibration target: close to 1
        out[f"{session}_pct_abs_z_ge_2"] = (s["z"].abs() >= 2).mean() * 100
    return out


if __name__ == "__main__":
    cfg = load()
    holdout = cfg.split.holdout_start
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    dev = bars[bars["trading_date"] < pd.Timestamp(holdout)]
    burn = int((dev["trading_date"] < pd.Timestamp(cfg.split.development_start)).sum())

    # self-check: the "current" spec must reproduce the production filter exactly
    model, _ = fit(dev, "current", burn)
    production = run_kalman_filter(dev, BAR_MINUTES).values
    assert np.allclose(model.filter([]).filtered_state.T, production, atol=1e-10), "current spec != production"

    rows, params_out = {}, {}
    early = dev[dev["trading_date"] < pd.Timestamp(FIT_SPLIT)]
    late_mask = (dev["trading_date"] >= pd.Timestamp(FIT_SPLIT)).values
    for spec in SPECS:
        t0 = time.time()
        model, params = fit(dev, spec, burn)
        result = model.filter(params) if spec != "current" else model.filter([])
        # out-of-sample model check: params fitted on the early window, likelihood scored on the late window
        _, early_params = fit(early, spec, burn)
        late_llf = model.filter(early_params).llf_obs[late_mask].mean() if spec != "current" else result.llf_obs[late_mask].mean()
        k = len(params)
        n = model.nobs - burn
        rows[spec] = {"n_params": k, "loglik_dev": result.llf, "bic": -2 * result.llf + k * np.log(n),
                      "loglik_per_bar_late_oos": late_llf,
                      **diagnostics(frozen_residual(dev, model, params), cfg.split.development_start, holdout),
                      "fit_seconds": time.time() - t0}
        params_out[spec] = dict(zip(model.param_names, params)) | {"early_fit": dict(zip(model.param_names, early_params))}
        print(f"{spec}: done in {time.time() - t0:.0f}s", flush=True)

    table = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    print(table.T.to_string())
    for spec, p in params_out.items():
        print(spec, {k: f"{v:.3e}" for k, v in p.items() if k != "early_fit"})
    run_dir = save_run(STRATEGY_DIR, "kalman_calibration", cfg, {"params": params_out},
                       tables={"spec_comparison": table})
    print("saved", run_dir)
