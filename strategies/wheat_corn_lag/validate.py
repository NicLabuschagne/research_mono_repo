"""Walk-forward validation on the development period.  python -m strategies.wheat_corn_lag.validate"""
import matplotlib
matplotlib.use("Agg")

from core_quant.walk_forward import run_validation
from strategies.wheat_corn_lag.backtest import make_evaluator
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

if __name__ == "__main__":
    cfg = load()
    evaluate, trading_dates = make_evaluator(cfg)
    run_validation(cfg, evaluate, trading_dates, STRATEGY_DIR)
