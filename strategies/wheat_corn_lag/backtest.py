"""
backtest.py
-----------
Step 6: the v2 rule, bar by bar, and the evaluate() the walk-forward engine calls.

    python -m strategies.wheat_corn_lag.backtest     # one illustrative run of the selected params, with charts
    python -m strategies.wheat_corn_lag.validate     # the walk-forward (Step 6 proper)

The rule (params in StrategyConfig, strategy.py)
  condition(t)  |own_z_L| > own_sd, and |corn_z_L| > corn_sd in the SAME direction (corn_sd = 0: no corn
                condition). Direction: against wheat's own move (wheat overshot up -> sell).
  entry         every bar the condition switches on (a fresh trigger), when flat
  episodes      a trigger in the same direction within one session (71 bars) of the previous trade's exit is a
                "re-trigger": the same divergence traded again ("take a few ticks, wait for it to trigger
                again"). It is labelled into the SAME episode, which is the unit of evidence: trades inside one
                episode are not independent. Labels change the counting, not the P&L.
  re-entry      optional (reentry=True): after a TARGET exit, enter again at once while the condition holds.
  fill          signal at bar t's CLOSE, entry at the OPEN of bar t + entry_delay_bars (default 1: the next
                bar's open). The entry bar earns its open-to-close move; later bars their close-to-close move.
                Exits are checked on closes and fill at that close.
  instrument    outright  wheat only; P&L = side x wheat's log move
                spread    wheat hedged with beta x corn, beta fixed at the signal bar;
                          P&L = side x (wheat's log move - beta x corn's log move)
  size of the divergence   D = |own_z_L| x sigma_e x sqrt(L), in log units, fixed at the signal bar
  level unit    "divergence": target and stop in units of D;  "atr": in units of sigma_w x sqrt(L), wheat's
                typical move over the lookback (an ATR-style level that ignores how big the overshoot was)
  exits         first of: the instrument moves target_share units our way (target), stop_share units against
                (stop), max_hold_bars (time), or the last bar before a possible-roll cut (roll cut). Fills at
                that bar's close, so a gap through a level fills at the gap price, not at the level.
  costs         cost_ticks x 0.25 c per round trip on wheat; the spread pays it on corn too, x beta

P&L is in log return of wheat notional, marked to market bar by bar and summed per session
(one row per trading date), the unit the walk-forward scores.
"""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.divergence import implied_wheat
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, StrategyConfig, load, load_bars

TICK_CENTS = 0.25
RETRIGGER_BARS = 71           # a same-direction trigger within one session of the last exit = same episode


def z_columns(params: StrategyConfig) -> tuple[str, str]:
    """Names of wheat's own and corn's standardised moves for this lookback and scaling."""
    if params.scaling == "empirical":
        return f"own_ze_{params.lookback}", f"corn_ze_{params.lookback}"
    if params.scaling == "sqrt":
        return f"own_z_{params.lookback}", f"corn_z_{params.lookback}"
    raise ValueError(f"scaling must be 'empirical' or 'sqrt', not {params.scaling!r}")


def signal_condition(features: pd.DataFrame, params: StrategyConfig) -> np.ndarray:
    own_column, corn_column = z_columns(params)
    own = features[own_column].values
    corn = features[corn_column].values
    with np.errstate(invalid="ignore"):                      # NaN compares as False: no signal
        condition = np.abs(own) > params.own_sd
        if params.corn_sd > 0:
            condition &= (np.abs(corn) > params.corn_sd) & (np.sign(corn) == np.sign(own))
    return condition


def simulate(bars: pd.DataFrame, features: pd.DataFrame, params: StrategyConfig) -> pd.DataFrame:
    """One row per trade."""
    lookback = params.lookback
    own_column, corn_column = z_columns(params)
    own = features[own_column].values
    # D, the size of the divergence in log units: a variant may supply its own (robustness.py)
    if f"div_{lookback}" in features:
        divergence = features[f"div_{lookback}"].values
    else:
        divergence = np.abs(features[f"own_move_{lookback}"].values)
    if params.level_unit == "divergence":
        unit = divergence
    elif params.level_unit == "atr":
        unit = features["sigma_w"].values * np.sqrt(lookback)            # wheat's typical move over the lookback
    else:
        raise ValueError(f"level_unit must be 'divergence' or 'atr', not {params.level_unit!r}")
    beta = features["beta"].values
    log_w, log_c = bars["log_w"].values, bars["log_c"].values
    open_w, open_c = np.log(bars["W_open"].values), np.log(bars["C_open"].values)
    price, corn_price = bars["W_open"].values, bars["C_open"].values          # entry prices (fills at the open)
    segment = bars["segment"].values
    hedged = params.instrument == "spread"
    if params.instrument not in ("outright", "spread"):
        raise ValueError(f"instrument must be 'outright' or 'spread', not {params.instrument!r}")
    condition = signal_condition(features, params)
    fresh = condition & ~np.concatenate([[False], condition[:-1]])
    count = len(bars)

    trades = []
    episode = 0
    last_side, last_exit = 0, -10 ** 9
    reentry_side = 0              # set after a target exit: the side we may re-enter while the condition holds
    t = 0
    while t < count:
        side = -np.sign(own[t]) if condition[t] else 0
        if reentry_side and (not condition[t] or side != reentry_side):
            reentry_side = 0                                   # the divergence closed or flipped: episode over
        is_first = fresh[t]
        is_reentry = params.reentry and reentry_side != 0 and condition[t] and side == reentry_side and not is_first
        if not (is_first or is_reentry):
            t += 1
            continue

        entry = t + params.entry_delay_bars
        if entry >= count or segment[entry] != segment[t]:
            t += 1
            continue
        retrigger = is_first and side == last_side and t - last_exit <= RETRIGGER_BARS
        if is_first and not retrigger:
            episode += 1
        kind = "re-entry" if not is_first else ("re-trigger" if retrigger else "first")
        target, stop = params.target_share * unit[t], params.stop_share * unit[t]
        hedge = beta[t] if hedged else 0.0                        # corn per unit of wheat, fixed at the signal

        last = min(entry + params.max_hold_bars, count - 1)
        exit_bar, reason = last, "time" if entry + params.max_hold_bars <= count - 1 else "data end"
        for bar in range(entry, last + 1):                        # the entry bar's own close can already exit
            if segment[bar] != segment[entry]:
                exit_bar, reason = bar - 1, "roll cut"            # close before the possible roll
                break
            move = side * ((log_w[bar] - open_w[entry]) - hedge * (log_c[bar] - open_c[entry]))
            if move >= target:
                exit_bar, reason = bar, "target"
                break
            if move <= -stop:
                exit_bar, reason = bar, "stop"
                break

        cost = params.cost_ticks * TICK_CENTS * (1 / price[entry] + abs(hedge) / corn_price[entry])
        trades.append({
            "episode": episode, "kind": kind, "hedge": hedge,
            "signal_time": bars.index[t], "entry_time": bars.index[entry], "exit_time": bars.index[exit_bar],
            "entry_bar": entry, "exit_bar": exit_bar, "side": side, "exit_reason": reason,
            "own_z": own[t], "corn_z": features[corn_column].values[t], "divergence_bp": divergence[t] * 1e4,
            "entry_price": price[entry], "exit_price": bars["W"].values[exit_bar], "bars_held": exit_bar - entry + 1,
            "entry_open_w": open_w[entry], "entry_open_c": open_c[entry],
            "ret": side * ((log_w[exit_bar] - open_w[entry]) - hedge * (log_c[exit_bar] - open_c[entry])) - cost,
            "cost": cost,
        })
        reentry_side = side if reason == "target" else 0
        last_side, last_exit = side, exit_bar
        t = exit_bar + 1
    return pd.DataFrame(trades)


def session_returns(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per trading date: P&L marked to market bar by bar (log return of notional), costs at entry, trades closed."""
    wheat = np.zeros(len(bars))           # wheat held over each bar, summed over open trades (they may overlap)
    corn = np.zeros(len(bars))            # corn held (the spread hedge), summed the same way
    extra = np.zeros(len(bars))           # the entry bar: open-to-close move, minus costs
    log_w, log_c = bars["log_w"].values, bars["log_c"].values
    for trade in trades.itertuples():
        wheat[trade.entry_bar + 1:trade.exit_bar + 1] += trade.side          # full bars: entry+1 .. exit
        corn[trade.entry_bar + 1:trade.exit_bar + 1] -= trade.side * trade.hedge
        entry_bar_move = (log_w[trade.entry_bar] - trade.entry_open_w) - trade.hedge * (log_c[trade.entry_bar] - trade.entry_open_c)
        extra[trade.entry_bar] += trade.side * entry_bar_move - trade.cost
        fill = getattr(trade, "exit_fill_log", np.nan)
        if not np.isnan(fill):                        # exit at a limit / stop price instead of the bar's close
            extra[trade.exit_bar] += trade.side * (fill - log_w[trade.exit_bar])
    bar_pnl = wheat * np.nan_to_num(bars["ret_w"].values) + corn * np.nan_to_num(bars["ret_c"].values) + extra
    dates = bars.index.normalize().tz_localize(None)
    daily = pd.DataFrame({"ret": pd.Series(bar_pnl, index=bars.index).groupby(dates).sum()})
    closed = pd.Series(1, index=trades["exit_time"].dt.normalize().dt.tz_localize(None)) if len(trades) else pd.Series(dtype=int)
    daily["trades"] = closed.groupby(level=0).sum().reindex(daily.index, fill_value=0)
    return daily


def make_evaluator(cfg, include_holdout: bool = False):
    """Load and build features once; return evaluate(params, start, end) and every trading date.
    include_holdout=True only from holdout.py, which the engine lets run once, after the gates pass."""
    bars = load_bars(cfg, include_holdout=include_holdout)
    features = build_features(bars, cfg.features)
    trading_dates = pd.DatetimeIndex(bars.index.normalize().tz_localize(None).unique())

    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        settings = replace(cfg.strategy, **params)
        trades = simulate(bars, features, settings)           # on ALL bars so the features are warm, then cut
        daily = session_returns(bars, trades)
        inside = (daily.index >= start) & (daily.index < end)
        daily = daily[inside]
        if len(trades):
            exit_dates = trades["exit_time"].dt.normalize().dt.tz_localize(None)
            kept = trades[(exit_dates >= start) & (exit_dates < end)].copy()
            kept["exit_date"] = kept["exit_time"].dt.normalize().dt.tz_localize(None)
            daily.attrs["trades"] = kept
        return daily

    return evaluate, trading_dates


# ------------------------------------------------------------------ charts and summary
def trade_summary(trades: pd.DataFrame, weeks: float) -> pd.DataFrame:
    grouped = trades.groupby("kind")["ret"]
    table = pd.DataFrame({"trades": grouped.size(), "per_week": grouped.size() / weeks,
                          "mean_bp": grouped.mean() * 1e4, "median_bp": grouped.median() * 1e4,
                          "hit_rate": grouped.apply(lambda r: (r > 0).mean()),
                          "avg_bars_held": trades.groupby("kind")["bars_held"].mean()})
    per_episode = trades.groupby("episode")["ret"].sum()
    table.loc["per EPISODE (all its trades summed)"] = [len(per_episode), len(per_episode) / weeks,
                                                          per_episode.mean() * 1e4, per_episode.median() * 1e4,
                                                          (per_episode > 0).mean(), np.nan]
    return table


def trades_on_chart(bars, features, trades: pd.DataFrame, params: StrategyConfig, start: str, days: int = 7) -> plt.Figure:
    begin = pd.Timestamp(start, tz="UTC")
    window = (bars.index >= begin) & (bars.index < begin + pd.Timedelta(days=days))
    frame = bars[window]
    position_of = {time: i for i, time in enumerate(frame.index)}
    implied = implied_wheat(bars, features, params.lookback)[window]
    own_column, corn_column = z_columns(params)
    own = features[own_column][window]
    corn = features[corn_column][window]
    x = np.arange(len(frame))

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, height_ratios=[1.7, 1])
    top.plot(x, frame["W"], lw=1.3, label="wheat")
    top.plot(x, implied, lw=1.1, ls="--", label=f"corn-implied wheat ({params.lookback} bars back)")
    colours = {"target": "#2ca02c", "stop": "#d62728", "time": "#7f7f7f", "roll cut": "#9467bd", "data end": "#7f7f7f"}
    shown = trades[trades["entry_time"].isin(frame.index) & trades["exit_time"].isin(frame.index)]
    for trade in shown.itertuples():
        x0, x1 = position_of[trade.entry_time], position_of[trade.exit_time]
        marker = "^" if trade.side > 0 else "v"
        face = colours[trade.exit_reason] if trade.kind == "first" else "white"
        top.scatter(x0, trade.entry_price, marker=marker, s=60, facecolor=face, edgecolor=colours[trade.exit_reason],
                    linewidth=1.5, zorder=6)
        top.plot([x0, x1], [trade.entry_price, trade.exit_price], color=colours[trade.exit_reason], lw=1.2, zorder=5)
        top.scatter(x1, trade.exit_price, marker="o", s=22, color=colours[trade.exit_reason], zorder=6)
    for reason, colour in [("target", "#2ca02c"), ("stop", "#d62728"), ("time", "#7f7f7f"), ("roll cut", "#9467bd")]:
        top.plot([], [], color=colour, lw=2, label=f"exit: {reason}")
    top.scatter([], [], marker="^", facecolor="white", edgecolor="black", label="hollow = re-trigger (same episode)")
    top.set_ylabel("cents/bu")
    top.legend(loc="best", fontsize=8, ncol=2)
    top.set_title(f"Week of {start}: trades on wheat vs corn-implied wheat (up = buy, down = sell; line = trade)", loc="left")

    bottom.plot(x, own, lw=1.2, label=own_column)
    bottom.plot(x, corn, lw=1.0, alpha=0.7, label=corn_column)
    for level in (params.own_sd, -params.own_sd):
        bottom.axhline(level, color="grey", ls=":", lw=0.9)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.legend(loc="best", fontsize=8)
    starts = np.flatnonzero((frame["session"] != frame["session"].shift()).values)
    bottom.set_xticks(starts, [f"{frame.index[i]:%a %d %b}" for i in starts])
    figure.tight_layout()
    return figure


def equity_chart(daily: pd.DataFrame, trades: pd.DataFrame) -> plt.Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 4.4))
    left.plot(daily.index, daily["ret"].cumsum() * 1e4, lw=1.4, label="all trades")
    kinds = [kind for kind in ["first", "re-trigger", "re-entry"] if (trades["kind"] == kind).any()]
    for kind in kinds:
        part = trades[trades["kind"] == kind]
        left.plot(part["exit_time"], part["ret"].cumsum() * 1e4, lw=1.0, label=f"{kind} only (booked at exit)")
    left.axhline(0, color="grey", lw=0.8)
    left.set_ylabel("cumulative bp of notional, after costs")
    left.set_title("Development period, IN-SAMPLE illustration (not the walk-forward)", loc="left")
    left.legend(fontsize=8)
    for kind in kinds:
        right.hist((trades.loc[trades["kind"] == kind, "ret"] * 1e4).clip(-300, 300), bins=40, density=True,
                   histtype="step", lw=1.6, label=kind)
    right.axvline(0, color="grey", lw=0.8)
    right.set_xlabel("bp per trade after costs (clipped at +-300)")
    right.set_title("Trade results by kind", loc="left")
    right.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)
    features = build_features(bars, cfg.features)
    weeks = (bars.index[-1] - bars.index[0]).days / 7
    pd.set_option("display.width", 200)

    figures, tables, metrics = {}, {}, {}
    for instrument in ["outright", "spread"]:
        params = replace(cfg.strategy, instrument=instrument)
        trades = simulate(bars, features, params)
        daily = session_returns(bars, trades)
        # self-check: bar-by-bar marking must add up to the trade-by-trade P&L
        assert abs(daily["ret"].sum() - trades["ret"].sum()) < 1e-9, "daily P&L does not match the trades"

        print(f"\n######## {params}")
        print("== trades by kind (development, IN-SAMPLE illustration)")
        print(trade_summary(trades, weeks).round(2).to_string())
        print("== exit reasons")
        print(pd.crosstab(trades["kind"], trades["exit_reason"]).to_string())
        print(f"trades per episode: mean {trades.groupby('episode').size().mean():.2f}, max {trades.groupby('episode').size().max()}")

        if instrument == "outright":
            entry_week = trades["entry_time"].dt.tz_localize(None).dt.to_period("W").dt.start_time
            busiest = trades.groupby(entry_week)["episode"].size().nlargest(1).index[0]
            for week in [f"{busiest:%Y-%m-%d}", "2025-07-28", "2023-02-06"]:
                figures[f"trades_{week}"] = trades_on_chart(bars, features, trades, params, week)
        figures[f"equity_{instrument}"] = equity_chart(daily, trades)
        tables[f"trades_{instrument}"] = trades.drop(columns=["entry_bar", "exit_bar", "entry_open_w", "entry_open_c"])
        metrics[instrument] = {"trades": len(trades), "episodes": int(trades["episode"].nunique()),
                               "mean_bp": float(trades["ret"].mean() * 1e4)}

    run_dir = save_run(STRATEGY_DIR, "backtest", cfg, metrics, tables, figures)
    print(f"\nsaved to {run_dir}")
