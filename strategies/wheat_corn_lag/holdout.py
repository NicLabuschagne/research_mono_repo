"""One-time holdout, locked until validation passes.  python -m strategies.wheat_corn_lag.holdout [--force]"""
import sys

import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_holdout
from strategies.wheat_corn_lag.backtest import make_evaluator
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

if __name__ == "__main__":
    cfg = load()
    evaluate, _ = make_evaluator(cfg, include_holdout=True)
    run_holdout(cfg, evaluate, STRATEGY_DIR, force="--force" in sys.argv)
