"""
walk_forward.py
---------------
The validation stage every strategy goes through, and the gate in front of the holdout.

  1. Folds      : the development period is cut into n blocks in time order.
  2. Grid       : every param combo is run once over development; its returns are
                  then scored block by block ("fold by fold").
  3. Walk-forward selection : at fold k, pick the combo with the best mean Sharpe on
                  folds 0..k-1, and record how it does on fold k. Joined up, that is the
                  out-of-sample result of the TUNING PROCESS itself, not of one hand-picked combo.
  4. Gates      : the selected params (cfg.strategy) and the walk-forward result must
                  pass every gate in cfg.gates.
  5. Holdout    : run_holdout refuses unless step 4 passed on the current config, and runs once.

A strategy plugs in with one function:

    evaluate(params: dict, start: Timestamp, end: Timestamp) -> DataFrame
        indexed by date (or bar time), columns:
          ret    : return for that period (additive, e.g. % of a fixed notional)
          trades : number of trades closed that period

The strategy config must have `strategy`, `split`, `validation` and `gates` sections,
and every grid name must be a field of `strategy`.

Words used below
  combo  : one set of params, e.g. {"threshold": 2.0, "window": 5}
  fold   : one block of dates in the development period
  daily  : the DataFrame evaluate() returns (one row per period, columns ret and trades)
"""
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core_quant import plots
from core_quant.config import latest_run, save_run

Evaluate = Callable[[dict, pd.Timestamp, pd.Timestamp], pd.DataFrame]


# ── Building blocks ────────────────────────────────────────────────────────────

def config_hash(cfg) -> str:
    """A short fingerprint of the whole config: if any value changes, the hash changes."""
    text = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def make_folds(dates: pd.DatetimeIndex, n_folds: int) -> list[pd.DatetimeIndex]:
    """Split the dates into `n_folds` consecutive blocks of (almost) equal length."""
    if n_folds < 3:
        raise ValueError("need at least 3 folds for walk-forward selection")
    blocks = np.array_split(dates, n_folds)
    return [pd.DatetimeIndex(block) for block in blocks]


def param_grid(grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid values.

    {"a": [1, 2], "b": [3]}  ->  [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    """
    names = list(grid.keys())
    combos = []
    for values in itertools.product(*grid.values()):
        combos.append(dict(zip(names, values)))
    return combos


def score(daily: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Sharpe, total return and max drawdown of additive per-period returns, plus the trade count."""
    returns = daily["ret"]
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()

    std = returns.std()
    if std > 0:
        sharpe = returns.mean() / std * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan                       # no variation (e.g. never traded): Sharpe undefined

    return {
        "sharpe": float(sharpe),
        "total_return": float(returns.sum()),
        "max_drawdown": float(drawdown.min()),
        "trades": int(daily["trades"].sum()),
    }


def assign_folds(index: pd.DatetimeIndex, fold_starts: list[pd.Timestamp]) -> pd.Series:
    """Fold number of every date: the last fold that started on or before it."""
    fold_of_date = pd.Series(-1, index=index)
    for fold_number, fold_start in enumerate(fold_starts):
        fold_of_date[index >= fold_start] = fold_number      # later folds overwrite earlier ones
    return fold_of_date


# ── Steps 2 and 3: grid, summary, walk-forward selection ───────────────────────

def evaluate_grid(evaluate: Evaluate, combos: list[dict], start, end, fold_starts,
                  periods_per_year: int = 252) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Run every combo once, score it on every fold.

    Returns
      results        : one row per (combo, fold): the params, the fold number and its score
      daily_by_combo : {combo number: the DataFrame evaluate() returned}
    """
    rows = []
    daily_by_combo = {}
    for combo_number, params in enumerate(combos):
        daily = evaluate(params, start, end)
        daily_by_combo[combo_number] = daily

        fold_of_date = assign_folds(daily.index, fold_starts)
        for fold_number in range(len(fold_starts)):
            fold_daily = daily[fold_of_date == fold_number]
            row = {"combo": combo_number, **params, "fold": fold_number}
            row.update(score(fold_daily, periods_per_year))
            rows.append(row)

    return pd.DataFrame(rows), daily_by_combo


def summarise(results: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per combo: how good it is (mean Sharpe) and how stable (std, worst fold, share positive)."""
    rows = []
    for combo_number, combo_results in results.groupby("combo"):
        sharpe_by_fold = combo_results["sharpe"]
        row = {"combo": combo_number}
        for name in names:
            row[name] = combo_results[name].iloc[0]           # params are the same on every fold
        row["mean_sharpe"] = sharpe_by_fold.mean()
        row["std_sharpe"] = sharpe_by_fold.std()
        row["min_sharpe"] = sharpe_by_fold.min()
        row["pct_folds_positive"] = float((sharpe_by_fold > 0).mean())
        row["total_return"] = combo_results["total_return"].sum()
        row["trades"] = combo_results["trades"].sum()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_sharpe", ascending=False)


def walk_forward_selection(results: pd.DataFrame, daily_by_combo: dict[int, pd.DataFrame],
                           fold_starts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At each fold k >= 1: choose the combo with the best mean Sharpe on folds before k,
    then record what it actually did on fold k (which it has not seen).

    Returns
      picks     : one row per fold: which combo was picked, its past mean Sharpe, its fold-k Sharpe
      oos_daily : the picked combos' returns on their fold k, joined into one out-of-sample series
    """
    # table of Sharpe: one row per fold, one column per combo
    sharpe_table = results.pivot(index="fold", columns="combo", values="sharpe")

    picks = []
    oos_pieces = []
    for fold_number in range(1, len(fold_starts)):
        past_folds = sharpe_table[sharpe_table.index < fold_number]
        past_mean = past_folds.mean()                         # one number per combo
        best_combo = int(past_mean.idxmax())                  # combo with the highest past mean

        picks.append({
            "fold": fold_number,
            "combo": best_combo,
            "past_mean_sharpe": past_mean[best_combo],
            "oos_sharpe": sharpe_table.loc[fold_number, best_combo],
        })

        daily = daily_by_combo[best_combo]
        fold_of_date = assign_folds(daily.index, fold_starts)
        oos_pieces.append(daily[fold_of_date == fold_number])

    return pd.DataFrame(picks), pd.concat(oos_pieces)


# ── Step 4: gates ──────────────────────────────────────────────────────────────

def check_gates(selected: pd.Series, development: dict, walk_forward: dict, gates) -> pd.DataFrame:
    """One row per gate: the value, the minimum, and whether it passed."""
    checks = [
        ("mean fold Sharpe", selected["mean_sharpe"], gates.min_mean_sharpe),
        ("share of folds positive", selected["pct_folds_positive"], gates.min_pct_folds_positive),
        ("development trades", development["trades"], gates.min_trades),
        ("walk-forward OOS Sharpe", walk_forward["sharpe"], gates.min_wf_oos_sharpe),
    ]
    table = pd.DataFrame(checks, columns=["gate", "value", "minimum"])
    table["passed"] = table["value"] >= table["minimum"]          # NaN counts as a fail
    return table


def selected_params(cfg) -> dict:
    """The current values of the grid params in cfg.strategy, e.g. {"threshold": 1.5, ...}."""
    selected = {}
    for name in cfg.validation.grid:
        selected[name] = getattr(cfg.strategy, name)
    return selected


# ── The two entry points a strategy calls ──────────────────────────────────────

def run_validation(cfg, evaluate: Evaluate, trading_dates: pd.DatetimeIndex, strategy_dir: Path) -> dict:
    """Grid x folds on the development period, walk-forward selection, gates, plots; saved as a run."""
    periods_per_year = cfg.validation.periods_per_year
    names = list(cfg.validation.grid)

    # 1. folds: development period only, the holdout is never touched here
    start = pd.Timestamp(cfg.split.development_start)
    end = pd.Timestamp(cfg.split.holdout_start)
    development_dates = trading_dates[(trading_dates >= start) & (trading_dates < end)]
    folds = make_folds(development_dates, cfg.validation.n_folds)
    fold_starts = [fold[0] for fold in folds]

    # 2. grid: every combo, plus the selected params if they are not on the grid
    selected = selected_params(cfg)
    combos = param_grid(cfg.validation.grid)
    if selected not in combos:
        combos.append(selected)
    selected_combo = combos.index(selected)

    results, daily_by_combo = evaluate_grid(evaluate, combos, start, end, fold_starts, periods_per_year)
    summary = summarise(results, names)

    # 3. walk-forward selection: how well does "pick the best so far" do out of sample?
    picks, oos_daily = walk_forward_selection(results, daily_by_combo, fold_starts)

    # 4. gates on the selected params and the walk-forward result
    selected_row = summary.set_index("combo").loc[selected_combo]
    development = score(daily_by_combo[selected_combo], periods_per_year)
    walk_forward = score(oos_daily, periods_per_year)
    gates = check_gates(selected_row, development, walk_forward, cfg.gates)
    gates_passed = bool(gates["passed"].all())

    # plots
    figures = {}
    for x, y in cfg.validation.heatmaps:
        figures[f"heatmap_{y}_x_{x}"] = plots.plot_param_heatmap(summary, x, y, selected)
    if names:                                                 # nothing to plot when no param is tuned
        figures["param_curves"] = plots.plot_param_curves(summary, names, selected)
    figures["fold_sharpe"] = plots.plot_fold_sharpe(results, picks, selected_combo, folds)
    figures["equity"] = plots.plot_cumulative(
        {"selected params": daily_by_combo[selected_combo]["ret"],
         "walk-forward selection (OOS)": oos_daily["ret"]},
        title=f"Development, {len(folds)} folds", boundaries=fold_starts)

    # save everything as one run
    stability_columns = ["mean_sharpe", "std_sharpe", "min_sharpe", "pct_folds_positive"]
    metrics = {
        "config_hash": config_hash(cfg),
        "selected_params": selected,
        "n_combos_tried": len(combos),
        "folds": [[str(fold[0].date()), str(fold[-1].date())] for fold in folds],
        "development": development,
        "selected_across_folds": selected_row[stability_columns].astype(float).to_dict(),
        "walk_forward_oos": walk_forward,
        "gates": gates.to_dict("records"),
        "gates_passed": gates_passed,
    }
    tables = {"grid_results": results, "grid_summary": summary, "walk_forward_picks": picks, "gates": gates}
    run_dir = save_run(strategy_dir, "validation", cfg, metrics, tables=tables, figures=figures)

    print(f"{len(combos)} combos x {len(folds)} folds. Top 10 by mean fold Sharpe:")
    print(summary.head(10).round(3).to_string(index=False))
    print(f"\nWalk-forward picks:\n{picks.round(3).to_string(index=False)}")
    print(f"\nSelected {selected}: development {development}")
    print(f"\nGates:\n{gates.round(3).to_string(index=False)}")
    verdict = "PASSED - holdout allowed" if gates_passed else "FAILED - holdout locked"
    print(f"\n{verdict}. Saved {run_dir}")
    return {"metrics": metrics, "results": results, "summary": summary, "picks": picks,
            "figures": figures, "run_dir": run_dir}


def run_holdout(cfg, evaluate: Evaluate, strategy_dir: Path, force: bool = False) -> dict:
    """The one-time test set. Only for a config whose latest validation passed every gate."""
    # check: there is a validation run, for exactly this config, and it passed
    last = latest_run(strategy_dir, "validation")
    if last is None:
        raise RuntimeError("No validation run yet: run validation first.")
    run_dir, validation = last
    if validation["config_hash"] != config_hash(cfg):
        raise RuntimeError(f"config.yaml changed since {run_dir.name}: re-run validation first.")
    if not validation["gates_passed"]:
        raise RuntimeError(f"{run_dir.name} failed its gates: holdout locked.")

    # check: the holdout has not been used before
    previous = latest_run(strategy_dir, "holdout")
    if previous and not force:
        raise RuntimeError(f"Holdout already used in {previous[0].name}. Re-running it turns it into "
                           "another development set; pass force=True only if you accept that.")

    # run the selected params on the holdout period
    start = pd.Timestamp(cfg.split.holdout_start)
    if cfg.split.holdout_end:
        end = pd.Timestamp(cfg.split.holdout_end)
    else:
        end = pd.Timestamp.max
    params = selected_params(cfg)
    daily = evaluate(params, start, end)
    holdout = score(daily, cfg.validation.periods_per_year)

    figure = plots.plot_cumulative({"holdout": daily["ret"]}, title=f"Holdout {params}")
    metrics = {"config_hash": config_hash(cfg), "selected_params": params, "validation_run": run_dir.name,
               "development": validation["development"], "holdout": holdout, "forced": force}
    saved = save_run(strategy_dir, "holdout", cfg, metrics, figures={"equity": figure})

    print(pd.DataFrame({"development": validation["development"], "holdout": holdout}).round(3))
    print(f"Saved {saved}")
    return {"metrics": metrics, "daily": daily, "figure": figure, "run_dir": saved}
