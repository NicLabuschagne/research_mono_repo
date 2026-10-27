"""One-time holdout, locked until validation passes.  python -m strategies.<name>.holdout [--force]"""
import sys

import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_holdout
from strategies._template.strategy import STRATEGY_DIR, load, make_evaluator

if __name__ == "__main__":
    cfg = load()
    evaluate, _ = make_evaluator(cfg)
    run_holdout(cfg, evaluate, STRATEGY_DIR, force="--force" in sys.argv)
