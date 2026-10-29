"""
validate.py
-----------
Walk-forward validation on the DEVELOPMENT period only: every combo in config.yaml's
grid, scored fold by fold, walk-forward selection, gates. Results land in runs/.

    python -m strategies.crush_strategy.validate [path/to/config.yaml]

Then read the heatmaps / param curves, set the selected params in config.yaml
(a plateau, not the best cell), and re-run until the gates pass. A config path
re-runs an alternative (e.g. a baseline) without editing config.yaml.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_validation
from strategies.crush_strategy.strategy import STRATEGY_DIR, load, make_evaluator

if __name__ == "__main__":
    cfg = load(Path(sys.argv[1])) if len(sys.argv) > 1 else load()
    evaluate, trading_dates = make_evaluator(cfg)
    run_validation(cfg, evaluate, trading_dates, STRATEGY_DIR)
