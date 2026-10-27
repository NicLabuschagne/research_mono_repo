"""
validate.py
-----------
Walk-forward validation on the DEVELOPMENT period only: every combo in config.yaml's
grid, scored fold by fold, walk-forward selection, gates. Results land in runs/.

    python -m strategies.crush_strategy.validate

Then read the heatmaps / param curves, set the selected params in config.yaml
(a plateau, not the best cell), and re-run until the gates pass.
"""
import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_validation
from strategies.crush_strategy.strategy import STRATEGY_DIR, load, make_evaluator

if __name__ == "__main__":
    cfg = load()
    evaluate, trading_dates = make_evaluator(cfg)
    run_validation(cfg, evaluate, trading_dates, STRATEGY_DIR)
