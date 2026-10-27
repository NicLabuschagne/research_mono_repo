from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core_quant.config import GateConfig, SplitConfig, from_dict
from core_quant.walk_forward import (evaluate_grid, make_folds, param_grid, score, summarise,
                                     walk_forward_selection)


@dataclass(frozen=True)
class Inner:
    a: int


@dataclass(frozen=True)
class Outer:
    inner: Inner
    split: SplitConfig


def test_from_dict_builds_nested_and_rejects_typos():
    cfg = from_dict(Outer, {"inner": {"a": 1}, "split": {"development_start": "2020", "holdout_start": "2021"}})
    assert cfg.inner.a == 1 and cfg.split.holdout_end is None
    with pytest.raises(ValueError, match="unknown"):
        from_dict(Outer, {"inner": {"a": 1, "typo": 2}, "split": {}})


def test_folds_cover_dates_in_order():
    dates = pd.bdate_range("2020-01-01", periods=100)
    folds = make_folds(dates, 4)
    assert sum(len(f) for f in folds) == 100
    assert all(folds[i][-1] < folds[i + 1][0] for i in range(3))


def test_param_grid_is_cartesian():
    assert param_grid({"a": [1, 2], "b": [3]}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_walk_forward_picks_from_past_folds_only():
    dates = pd.bdate_range("2020-01-01", periods=120)
    folds = make_folds(dates, 4)
    starts = [f[0] for f in folds]
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(dates))

    def evaluate(params, start, end):
        # combo "good" wins in the first half, "late" wins in the second half
        edge = np.where(dates < starts[2], 1.0, -1.0) * (1 if params["kind"] == "good" else -1)
        return pd.DataFrame({"ret": noise + edge, "trades": 1}, index=dates)

    results, dailies = evaluate_grid(evaluate, param_grid({"kind": ["good", "late"]}), dates[0], dates[-1], starts)
    picks, oos = walk_forward_selection(results, dailies, starts)
    # folds 1 and 2 pick "good" (best so far); it then loses in fold 2-3, which the OOS result must show
    assert list(picks["combo"][:2]) == [0, 0]
    assert len(oos) == sum(len(f) for f in folds[1:])
    assert score(oos)["sharpe"] < score(dailies[0][dates < starts[2]])["sharpe"]
    summary = summarise(results, ["kind"])
    assert set(summary.columns) >= {"mean_sharpe", "pct_folds_positive", "trades"}


def test_score_matches_hand_calc():
    daily = pd.DataFrame({"ret": [1.0, -2.0, 3.0], "trades": [1, 0, 2]})
    result = score(daily, periods_per_year=1)
    assert result["total_return"] == 2.0 and result["trades"] == 3
    assert result["max_drawdown"] == -2.0
    assert np.isclose(result["sharpe"], np.mean([1, -2, 3]) / np.std([1, -2, 3], ddof=1))
