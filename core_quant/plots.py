### IMPORTS
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns

from core_quant.theme import set_theme, diverging_cmap, PALETTE
set_theme()


# ── Visualizations ──────────────────────────────────────────────────────────

# Line series plot
def plot_time_series(df: pd.DataFrame, title: str, cols: list[str]) -> None:
    fig, ax = plt.subplots()
    ax.plot(df[cols])
    ax.set(
        xlabel="Time",
        ylabel="Price",
        title=title,
    )


# Correlation Matrix
def plot_corr_matrix(df: pd.DataFrame, cols: list[str], title: str) -> None:
    if cols is None:
        cols = df.columns
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap=PALETTE, ax=ax)
    ax.title.set_text(title)


# Return Distribution
def plot_ret_distribution(df: pd.DataFrame, cols: list[str], title: str) -> None:
    for col in cols:
        ax = df[col].hist(bins=40, figsize=(10, 5))
        ax.set(
            xlabel= f"{col}_returns",
            ylabel="Count",
            title=f"Histogram of Returns for {title}",
        );

# AutoCorrelation plots
def plot_pacf(df: pd.DataFrame, col: str, lags: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sm_plot_pacf(df[col].dropna(), lags=lags, ax=ax)
    ax.set_title(f"{col} PACF (Partial Autocorrelation)")
    plt.tight_layout()
    return fig

# Ic Decay
def plot_ic_decay(ic_table: pd.DataFrame, title: str = "IC decay by horizon"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in ic_table.index:
        ax.plot(ic_table.columns, ic_table.loc[feature], marker="o", label=feature)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set(
        xlabel="Forward Horizon (H)",
        ylabel="Rank IC",
        title=title,
    )
    ax.legend()
    plt.tight_layout()
    return fig

# Feature correlation
def feature_corr(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].corr(method="spearman")

def plot_feature_correlation(df: pd.DataFrame, features: list[str], method: str = "spearman", title: str = "Feature correlation"):
    corr = feature_corr(df, features)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, mask=mask, cmap=diverging_cmap(), vmin=-1, vmax=1, ax=ax, annot=True, fmt=".2f")
    ax.set_title(title)
    plt.tight_layout()
    return corr

# Equity curve visuals
def plot_equity_curve(returns: pd.Series, title: str = "Equity curve"):
    """Compounded growth of capital, with the drawdown underneath, both in percent.

    `returns` must be a FRACTION of capital per trade, not raw P&L - use
    model.to_simple_returns() to convert first. Percentages are what make two
    different products comparable on the same axes.

    Pass the NON-OVERLAPPING sample, or the curve counts each trade several
    times and looks far smoother than the strategy really is.
    """
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, (ax_equity, ax_drawdown) = plt.subplots(2, 1, figsize=(10, 6),
    sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax_equity.plot(equity.index, (equity.values - 1.0) * 100, linewidth=1.6)
    ax_equity.axhline(0, color="grey", linewidth=1)
    ax_equity.set_ylabel("Cumulative return (%)")
    ax_equity.set_title(title)

    ax_drawdown.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.6)
    ax_drawdown.set_ylabel("Drawdown (%)")
    ax_drawdown.set_xlabel("Date")

    plt.tight_layout()
    plt.show()

    return equity

# Block Permutation & Bootstrap plots
def plot_permutation_distribution(permuted_scores: np.ndarray, actual_score: float, metric_name: str = "Rank IC"):
    """Plots the null distribution from block permutation against the actual realized score."""
    plt.figure(figsize=(10, 6))

    # Plot the distribution of random block permutations
    ax = sns.histplot(permuted_scores, bins=50, kde=True, color='slategray', stat='density')

    # Plot the actual achieved score
    plt.axvline(actual_score, color='crimson', linestyle='--', linewidth=2,
                label=f'Actual {metric_name}: {actual_score:.3f}')

    # Calculate the 95th percentile (Significance threshold)
    threshold = np.percentile(permuted_scores, 95)
    plt.axvline(threshold, color='gold', linestyle=':', linewidth=2, label=f'95% Threshold: {threshold:.3f}')

    plt.title(f'Permutation Test Null Distribution ({metric_name})', fontsize=14, pad=15)
    plt.xlabel(f'Simulated {metric_name}', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_bootstrap_confidence(bootstrapped_scores: np.ndarray, actual_score: float, ci_lower: float, ci_upper: float):
    """Plots the bootstrapped distribution of Sharpe ratios with confidence intervals."""
    plt.figure(figsize=(10, 6))

    # Plot the distribution of bootstrapped Sharpes
    sns.histplot(bootstrapped_scores, bins=50, kde=True, color='steelblue', stat='density')

    # Plot the actual Sharpe and the CI boundaries
    plt.axvline(actual_score, color='mediumspringgreen', linestyle='-', linewidth=2.5,
                label=f'Actual Sharpe: {actual_score:.2f}')
    plt.axvline(ci_lower, color='lightcoral', linestyle='--', linewidth=2, label=f'Lower 2.5%: {ci_lower:.2f}')
    plt.axvline(ci_upper, color='lightcoral', linestyle='--', linewidth=2, label=f'Upper 97.5%: {ci_upper:.2f}')

    # Shade the confidence interval region
    plt.axvspan(ci_lower, ci_upper, color='lightcoral', alpha=0.1)

    plt.title('Bootstrapped Sharpe Ratio (95% Confidence Interval)', fontsize=14, pad=15)
    plt.xlabel('Annualized Sharpe Ratio', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ── Walk-forward validation (tuning) ────────────────────────────────────────

def _at_fixed(summary: pd.DataFrame, fixed: dict) -> pd.DataFrame:
    """Rows where every param in `fixed` equals its value."""
    mask = np.ones(len(summary), dtype=bool)
    for name, value in fixed.items():
        mask &= (summary[name] == value).values
    return summary[mask]


def plot_param_heatmap(summary: pd.DataFrame, x: str, y: str, selected: dict, metric: str = "mean_sharpe"):
    """Mean fold Sharpe over two params, other params held at the selected values.
    Each cell also shows the share of positive folds; the selected cell is outlined.
    Look for a broad plateau, not the single best cell."""
    rows = _at_fixed(summary, {k: v for k, v in selected.items() if k not in (x, y)})
    values = rows.pivot(index=y, columns=x, values=metric)
    positive = rows.pivot(index=y, columns=x, values="pct_folds_positive")
    limit = np.nanmax(np.abs(values.values)) or 1.0

    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(values.values.astype(float), cmap=diverging_cmap(), vmin=-limit, vmax=limit,
                      aspect="auto", origin="lower")
    ax.set_xticks(range(len(values.columns)), [str(c) for c in values.columns])
    ax.set_yticks(range(len(values.index)), [str(r) for r in values.index])
    ax.set(xlabel=x, ylabel=y, title=f"{metric} by {y} x {x}\n(other params at selected values)")
    ax.grid(False)
    for i, row in enumerate(values.index):
        for j, column in enumerate(values.columns):
            ax.text(j, i, f"{values.loc[row, column]:.2f}\n{positive.loc[row, column]:.0%} +",
                    ha="center", va="center", fontsize=9)
    if selected[x] in values.columns and selected[y] in values.index:
        i, j = list(values.index).index(selected[y]), list(values.columns).index(selected[x])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_param_curves(summary: pd.DataFrame, names: list[str], selected: dict, metric: str = "mean_sharpe"):
    """One panel per param: mean fold Sharpe +/- its std across folds, other params at selected values."""
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.8), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], names):
        rows = _at_fixed(summary, {k: v for k, v in selected.items() if k != name}).sort_values(name)
        ax.errorbar(rows[name], rows[metric], yerr=rows["std_sharpe"], marker="o", capsize=4)
        ax.axvline(selected[name], color="grey", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set(xlabel=name, title=name)
    axes[0][0].set_ylabel(f"{metric} (+/- std across folds)")
    fig.tight_layout()
    return fig


def plot_fold_sharpe(results: pd.DataFrame, picks: pd.DataFrame, selected_combo: int, folds: list):
    """Selected params' Sharpe per fold (bars) vs the walk-forward pick's out-of-sample Sharpe (dots)."""
    selected = results[results["combo"] == selected_combo].set_index("fold")["sharpe"]
    fmt = "%Y-%m" if (folds[-1][-1] - folds[0][0]).days > 60 else "%m-%d %H:%M"   # intraday folds need the time
    labels = [f"{fold[0]:{fmt}}\n{fold[-1]:{fmt}}" for fold in folds]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(selected.index, selected.values, label="selected params")
    ax.scatter(picks["fold"], picks["oos_sharpe"], color="black", zorder=3, label="walk-forward pick (OOS)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel="Sharpe", title="Sharpe by fold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative(curves: dict[str, pd.Series], title: str, boundaries=()):
    """Cumulative sum of additive returns, drawdown underneath; dashed lines at fold starts."""
    fig, (ax_equity, ax_drawdown) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                                 gridspec_kw={"height_ratios": [3, 1]})
    for label, returns in curves.items():
        equity = returns.cumsum()
        ax_equity.plot(equity.index, equity.values, label=label)
        ax_drawdown.plot(equity.index, (equity - equity.cummax()).values, label=label)
    for boundary in boundaries:
        for ax in (ax_equity, ax_drawdown):
            ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax_equity.axhline(0, color="grey", linewidth=0.8)
    ax_equity.set(ylabel="Cumulative return", title=title)
    ax_drawdown.set_ylabel("Drawdown")
    ax_equity.legend()
    fig.tight_layout()
    return fig
