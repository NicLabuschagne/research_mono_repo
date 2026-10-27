"""
holdout.py
----------
The one-time test set. Refuses unless the latest validation run passed every gate
on the current config.yaml, and refuses a second run unless --force.

    python -m strategies.crush_strategy.holdout
"""
import sys

import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_holdout
from strategies.crush_strategy.strategy import STRATEGY_DIR, load, make_evaluator

if __name__ == "__main__":
    cfg = load()
    evaluate, _ = make_evaluator(cfg)
    run_holdout(cfg, evaluate, STRATEGY_DIR, force="--force" in sys.argv)
