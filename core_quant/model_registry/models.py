"""
models.py
---------
Model registry: pick an sklearn model by name from config, and get leak-free
walk-forward predictions from it.

sklearn already gives every model the same fit / predict interface, so this does
not wrap it. It only adds:
  - MODELS / SCALERS : names usable in config.yaml
  - build_model      : ModelConfig -> Pipeline (imputer -> scaler -> model), so
                       preprocessing is refit on each training fold, never on test rows
  - walk_forward_predict : out-of-sample predictions, each fold trained on earlier rows only

To tune a model, wrap walk_forward_predict in a strategy's `evaluate(params, start, end)`
and put the hyper-parameters in cfg.validation.grid (see core_quant.walk_forward).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.ar_model import AutoReg

MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "quantile": QuantileRegressor,
    "logit": LogisticRegression,
    "tree": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "rf_classifier": RandomForestClassifier,
    "gbr": GradientBoostingRegressor,
}

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "quantile": QuantileTransformer,
}


@dataclass(frozen=True)
class ModelConfig:
    name: str                                   # key of MODELS
    params: dict = field(default_factory=dict)  # passed to the model, e.g. {C: 0.1}
    scaler: str | None = "standard"             # key of SCALERS, or None
    impute: str | None = None                   # SimpleImputer strategy, e.g. "median", or None


def build_model(cfg: ModelConfig) -> Pipeline:
    steps = []
    if cfg.impute:
        steps.append(SimpleImputer(strategy=cfg.impute))
    if cfg.scaler:
        steps.append(SCALERS[cfg.scaler]())
    steps.append(MODELS[cfg.name](**cfg.params))
    return make_pipeline(*steps)


def walk_forward_predict(cfg: ModelConfig, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = 5, gap: int = 0) -> pd.Series:
    """Out-of-sample prediction for every row after the first training block.

    gap : rows dropped between train and test. Set it to at least the label horizon
          (in rows), or the last training labels overlap the first test rows.
    Classifiers return P(class 1); regressors return the prediction.
    """
    predictions = pd.Series(np.nan, index=X.index)
    for train, test in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        model = build_model(cfg).fit(X.iloc[train], y.iloc[train])
        if hasattr(model, "predict_proba"):
            predictions.iloc[test] = model.predict_proba(X.iloc[test])[:, 1]
        else:
            predictions.iloc[test] = model.predict(X.iloc[test])
    return predictions


def ols_model(df: pd.DataFrame, col: str, lags: int):
    """
    Fit an AutoReg(`lags`) model with a constant to `df[col]`.

    Returns the fitted statsmodels results object (not a dict) so callers keep
    access to `.summary()`, `.params`, `.predict()` and the rest.
    """
    model = AutoReg(df[col].dropna(), lags=lags, trend="c")
    return model.fit()
