# <strategy name>

Copy this folder to `strategies/<name>/`, rename the imports in `validate.py` / `holdout.py`.

## 0. Hypothesis (write BEFORE looking at results)
- **Thesis:** what mispricing / behaviour is being traded
- **Why it should pay:** who is on the other side and why
- **Kill criteria:** what result would make you drop it

## Process
| Gate | Where | Pass when |
|---|---|---|
| 1. Data | `research.ipynb`, `core_quant.utilities.frame_summary` | no gaps / duplicates / look-ahead |
| 2. Features & signal | `core_quant.feature_registry`, `core_quant.validation` (IC, NW t-stat, block permutation) | signal has IC with an honest p-value |
| 3. Walk-forward validation | `python -m strategies.<name>.validate` | every gate in `config.yaml` passes |
| 4. Holdout (once) | `python -m strategies.<name>.holdout` | result close to the walk-forward OOS |

Every validation / holdout run is saved in `runs/` with its config, git commit, metrics and plots.

## Log
| Date | Run | Change | Result |
|---|---|---|---|
