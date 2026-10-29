# Research Monorepo

**A repeatable pipeline for quantitative trading research.** Every strategy goes through the same process, from hypothesis to a locked holdout. Every number comes from a config file, and every result is saved with the config and git commit that produced it.

## The research cycle

```
 hypothesis ─► data checks ─► features & IC ─► walk-forward validation ─► gates ─► holdout (once)
  README.md    research.ipynb  core_quant.        validate.py               config     holdout.py
                               validation         grid × folds              .yaml
```

1. **Hypothesis first.** Write the thesis, why it should pay, and the kill criteria before looking at results.
2. **Holdout locked up front.** `holdout_start` is chosen before exploring and never moved.
3. **Explore** in the notebook: data integrity, features, information coefficient (IC) and its decay.
4. **Validate** on the development period only. Run the parameter grid across walk-forward folds, plus a walk-forward selection that measures the tuning process itself out of sample.
5. **Gates** (Sharpe, consistency across folds, trade count, walk-forward out-of-sample Sharpe) must all pass before the holdout unlocks.
6. **Holdout runs once**, on the exact config that passed.

## Layout

```
core_quant/              strategy-agnostic engine, written once
  walk_forward.py        folds, grid, walk-forward selection, gates, holdout lock
  config.py              YAML → dataclasses, run store (config + git commit + metrics)
  validation.py          IC, Newey-West t-stats, block permutation, bootstrap, Ljung-Box
  feature_registry.py    reusable features and signals
  model_registry/        sklearn models by name, leak-free walk-forward predictions
  plots.py, theme.py     tuning heatmaps, parameter curves, fold Sharpe, equity
  tearsheet.py           standard tear sheet saved with every validation / holdout run (tearsheet.png + metrics)
strategies/<name>/       one folder per idea
  README.md              hypothesis, data notes, run log
  config.yaml            every parameter, split, grid and gate
  strategy.py            dataclasses + evaluate(params, start, end)
  research.ipynb         scratch exploration
  validate.py, holdout.py
  runs/                  every validation / holdout run, saved
strategies/_template/    copy this to start a new strategy
data/{raw,processed}     market data (not in git)
```

A strategy only supplies **`evaluate(params, start, end)`**, which returns per-period returns and trade counts. Everything else is shared. If it also sets `daily.attrs["trades"]` (one row per trade: `ret`, `exit_date`, optionally `pnl`, `holding_minutes`, `exit_reason`), the tear sheet adds win rate, expectancy, profit factor and losing streaks.

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .
pytest

cp -r strategies/_template strategies/my_idea          # then fix the imports in validate.py / holdout.py
python -m strategies.my_idea.validate                   # tune: read runs/<latest>/
python -m strategies.my_idea.holdout                    # once, after the gates pass
```

## Strategies

| Strategy | Idea | Status |
|---|---|---|
| [`crush_strategy`](strategies/crush_strategy/output/README.md) | Overnight reversion of the soybean crush residual (ZS vs ZM/ZL), Kalman hedge | Recalibrated Oct 2026 (session-close exit): passes all 4 gates on development (Sharpe 2.25, walk-forward OOS 1.10); holdout already seen, needs post-2025 data |
| [`wheat_microstructure`](strategies/wheat_microstructure/README.md) | Order flow in ZW tick data | In progress: data checks |
