"""
tearsheet.py
------------
Standard performance report, saved with every validation and holdout run.

Input (the same contract as walk_forward)
  daily  : DataFrame indexed by date with "ret" (additive return per period, e.g. % of notional,
           zero when flat) and "trades" (trades closed that period).
  trades : optional, one row per trade: "ret" (same unit as daily), "exit_date"; optional "pnl"
           (P&L in the strategy's own unit, named by trades.attrs["pnl_unit"]), "holding_minutes",
           "exit_reason". A strategy attaches it as daily.attrs["trades"] inside evaluate().

Returns are additive (equity = cumulative sum), so drawdowns are in the same unit as returns
and Calmar = annual return / |max drawdown|.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.theme import LIGHT, PALETTE, diverging_cmap, set_theme

set_theme()
ROLLING_DAYS = 126            # ~6 months of trading days for the rolling Sharpe


def trades_of(daily: pd.DataFrame) -> pd.DataFrame | None:
    return daily.attrs.get("trades")


def _max_streak(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Year x month table of summed returns."""
    by_month = daily["ret"].groupby([daily.index.year, daily.index.month]).sum()
    return by_month.unstack().reindex(columns=range(1, 13))


def tear_sheet(daily: pd.DataFrame, trades: pd.DataFrame | None = None, periods_per_year: int = 252) -> dict:
    returns = daily["ret"]
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    annual_return = returns.mean() * periods_per_year
    downside = returns[returns < 0]
    max_dd = drawdown.min()
    underwater = (drawdown < 0).values
    months = returns.groupby([returns.index.year, returns.index.month]).sum()
    stats = {
        "start": str(returns.index[0].date()), "end": str(returns.index[-1].date()), "periods": len(returns),
        "total_return": returns.sum(), "annual_return": annual_return,
        "annual_volatility": returns.std() * np.sqrt(periods_per_year),
        "sharpe": returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan,
        "sortino": (returns.mean() / np.sqrt((downside ** 2).sum() / len(returns)) * np.sqrt(periods_per_year)
                    if len(downside) else np.nan),
        "max_drawdown": max_dd,
        "calmar": annual_return / abs(max_dd) if max_dd < 0 else np.nan,
        "longest_drawdown_periods": _max_streak(underwater),
        "pct_months_positive": (months > 0).mean() * 100,
        "best_month": months.max(), "worst_month": months.min(),
    }
    if trades is None or trades.empty:
        return stats

    r = trades["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    n_months = len(pd.period_range(returns.index[0], returns.index[-1], freq="M"))
    stats |= {
        "trades": len(r), "trades_per_month": len(r) / n_months,
        "win_rate": (r > 0).mean() * 100,
        "avg_win": wins.mean(), "avg_loss": losses.mean(),
        "payoff_ratio": wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "expectancy": r.mean(),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() < 0 else np.nan,
        "best_trade": r.max(), "worst_trade": r.min(),
        "max_consecutive_losses": _max_streak((r <= 0).values),
    }
    if "pnl" in trades:
        unit = trades.attrs.get("pnl_unit", "pnl")
        stats |= {f"expectancy_{unit}": trades["pnl"].mean(), f"total_{unit}": trades["pnl"].sum()}
    if "holding_minutes" in trades:
        stats["median_holding_minutes"] = trades["holding_minutes"].median()
    if "exit_reason" in trades:
        stats |= {f"exit_{k}_pct": v * 100 for k, v in trades["exit_reason"].value_counts(normalize=True).items()}
    return stats


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return "n/a" if pd.isna(value) else f"{value:,.2f}"


def plot_tear_sheet(daily: pd.DataFrame, trades: pd.DataFrame | None, stats: dict, title: str,
                    unit: str = "% of notional"):
    """Equity + drawdown, rolling Sharpe, monthly heatmap, trade distribution, key numbers."""
    returns = daily["ret"]
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, x=0.05, ha="left", fontsize=12, fontweight="bold", color=LIGHT["ink"])
    grid = fig.add_gridspec(4, 3, height_ratios=[3, 1.2, 1.6, 2.2], width_ratios=[1, 1, 0.9], hspace=0.5, wspace=0.3,
                            left=0.06, right=0.98, top=0.93, bottom=0.05)

    ax_eq = fig.add_subplot(grid[0, :2])
    ax_eq.plot(equity.index, equity.values, color=PALETTE[0])
    ax_eq.axhline(0, color=LIGHT["muted"], linewidth=0.8)
    ax_eq.set(title=f"Cumulative return ({unit})", ylabel=unit)

    ax_dd = fig.add_subplot(grid[1, :2], sharex=ax_eq)
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color=PALETTE[1], alpha=0.6, linewidth=0)
    ax_dd.set(title="Drawdown", ylabel=unit)

    ax_roll = fig.add_subplot(grid[2, :2], sharex=ax_eq)
    rolling = returns.rolling(ROLLING_DAYS).mean() / returns.rolling(ROLLING_DAYS).std() * np.sqrt(252)
    ax_roll.plot(rolling.index, rolling.values, color=PALETTE[0], linewidth=1.4)
    ax_roll.axhline(0, color=LIGHT["muted"], linewidth=0.8)
    ax_roll.set(title=f"Rolling Sharpe ({ROLLING_DAYS} periods)", ylabel="Sharpe")

    ax_month = fig.add_subplot(grid[3, :2])
    table = monthly_returns(daily)
    limit = np.nanmax(np.abs(table.values)) or 1
    ax_month.imshow(table.values, cmap=diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    for (i, j), value in np.ndenumerate(table.values):
        if not np.isnan(value):
            ax_month.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=LIGHT["ink"])
    ax_month.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax_month.set_yticks(range(len(table)), table.index)
    ax_month.grid(False)
    ax_month.set_title(f"Monthly return ({unit})")

    ax_text = fig.add_subplot(grid[0:2, 2])
    ax_text.axis("off")
    labels = {"total_return": "Total return", "annual_return": "Annual return", "annual_volatility": "Annual volatility",
              "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max drawdown", "calmar": "Calmar",
              "longest_drawdown_periods": "Longest drawdown (periods)", "pct_months_positive": "Months positive (%)",
              "best_month": "Best month", "worst_month": "Worst month", "trades": "Trades",
              "trades_per_month": "Trades / month", "win_rate": "Win rate (%)", "avg_win": "Average win",
              "avg_loss": "Average loss", "payoff_ratio": "Payoff ratio", "expectancy": "Expectancy / trade (ret)",
              "profit_factor": "Profit factor", "max_consecutive_losses": "Max losing streak",
              "median_holding_minutes": "Median hold (min)"}
    for k in stats:                      # strategy P&L unit, e.g. expectancy_c_per_bu -> "Expectancy / trade (c/bu)"
        for prefix, name in [("expectancy_", "Expectancy / trade"), ("total_", "Total")]:
            if k.startswith(prefix) and k != "total_return":
                labels[k] = f"{name} ({k[len(prefix):].replace('_per_', '/')})"
    rows = [(labels[k], _fmt(stats[k])) for k in labels if k in stats]
    ax_text.set_title(f"{stats['start']} to {stats['end']}")
    for i, (label, value) in enumerate(rows):
        y = 1 - i / max(len(rows), 1) * 0.98
        ax_text.text(0.0, y, label, transform=ax_text.transAxes, color=LIGHT["ink_secondary"], fontsize=9, va="top")
        ax_text.text(1.0, y, value, transform=ax_text.transAxes, color=LIGHT["ink"], fontsize=9, va="top", ha="right")

    ax_hist = fig.add_subplot(grid[2:4, 2])
    if trades is not None and not trades.empty:
        r = trades["ret"]
        bins = np.linspace(r.min(), r.max(), 31)
        ax_hist.hist(r[r > 0], bins=bins, color=PALETTE[0], label="wins")
        ax_hist.hist(r[r <= 0], bins=bins, color=PALETTE[1], label="losses")
        ax_hist.axvline(r.mean(), color=LIGHT["ink"], linewidth=1, linestyle="--", label=f"mean {r.mean():.3f}")
        ax_hist.set(title=f"Per-trade return ({unit})", ylabel="trades")
        ax_hist.legend()
    else:
        ax_hist.axis("off")
        ax_hist.text(0.5, 0.5, "no trade-level data", ha="center", color=LIGHT["ink_secondary"])
    return fig
