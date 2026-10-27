"""Walk-forward validation on the development period.  python -m strategies.<name>.validate"""
import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_validation
from strategies._template.strategy import STRATEGY_DIR, load, make_evaluator

if __name__ == "__main__":
    cfg = load()
    evaluate, trading_dates = make_evaluator(cfg)
    run_validation(cfg, evaluate, trading_dates, STRATEGY_DIR)
