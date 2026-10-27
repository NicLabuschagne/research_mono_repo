# IMPORTS
import pandas as pd
import numpy as np
from typing import Any
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, coint, kpss, pacf
from scipy.stats import spearmanr

# ── Common transforms ──────────────────────────────────────────────────────────
def quick_adf(series: pd.Series) -> dict[str, Any]:
    """
    Run an ADF stationarity test with sensible defaults.
    Returns
    -------
    dict with keys: stat (float), pvalue (float), n_lags (int),
                    stationary_5pct (bool)
    """
    if adfuller is None:
        raise ImportError("statsmodels is not installed")
    clean = series.dropna()
    stat, pvalue, n_lags, *_ = adfuller(clean, autolag="AIC")
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "n_lags": int(n_lags),
        "stationary_5pct": bool(pvalue < 0.05),
    }
# build stationarity test using ADfuller
def ad_fuller_test(df: pd.DataFrame) -> dict[str, Any]:
    """Takes in a log transformed df and outputs P-value statistics for stationarity"""
    results = adfuller(df, result_object=True)
    print(results)
    print(f"p_value: {results.pvalue:.3f}")
    return results

# ── Factor screening ───────────────────────────────────────────────────────────

def ic_rank_spearman(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Rank IC of each feature against one target column.
    Returns one row per feature with the IC, its p-value, and the sample size.
    """

    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()
        ic, p_value = spearmanr(pair[feature], pair[target])
        rows[feature] = {"ic": ic, "p_value": p_value, "n": len(pair)}

    return pd.DataFrame(rows).T


# Newey west t stats to account for overlapping horizons
def nw_t_stat(df: pd.DataFrame, features: list[str], target: str, lags: int) -> pd.DataFrame:
    """Newey-West t-statistic for each feature's rank IC.

    Overlapping forward returns make the plain p-value far too optimistic: a
    10-day target re-uses 9 of its 10 days at every step, so the observations
    are not independent. Setting lags to the forecast horizon corrects for it.

    The slope of standardised ranks IS the Spearman IC, so this is the same
    number as ic_rank_spearman with an honest standard error attached.
    """
    rows = {}
    for feature in features:
        pair = df[[feature, target]].dropna()

        x = pair[feature].rank()
        y = pair[target].rank()
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()

        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[feature] = {"t_nw": fit.tvalues.iloc[1], "p_nw": fit.pvalues.iloc[1], "n": len(pair)}

    return pd.DataFrame(rows).T


# Ic_decay across horizons
def ic_decay(df: pd.DataFrame, features: list[str], horizons: list[int],
             target_template: str = "target_{horizon}_d") -> pd.DataFrame:
    """Rank IC of each feature at several forward horizons.
    The target columns must already exist
    Returns a table with one row per feature and one column per horizon.
    """
    columns = {}
    for horizon in horizons:
        target = target_template.format(horizon=horizon)
        result = ic_rank_spearman(df, features, target)
        columns[horizon] = result["ic"].astype(float)

    return pd.DataFrame(columns)

# Block Permutation - Is the Edge real
def block_permutation_ic(signal: pd.Series, returns: pd.Series, block_size: int = 5, n_perms: int = 1000,
                         seed: int | None = 0) -> dict:
    """Calculates p-value of Rank IC using block permutation to preserve autocorrelation."""
    # Align data strictly to avoid index mismatch
    df = pd.concat([signal, returns], axis=1).dropna()
    sig_arr = df.iloc[:, 0].values
    ret_arr = df.iloc[:, 1].values

    # 1. Calculate the actual Rank IC
    actual_ic, _ = spearmanr(sig_arr, ret_arr)

    # 2. Truncate arrays so they divide perfectly by block_size
    remainder = len(ret_arr) % block_size
    if remainder != 0:
        ret_arr = ret_arr[:-remainder]
        sig_arr = sig_arr[:-remainder]

    num_blocks = len(ret_arr) // block_size

    # 3. Reshape returns into blocks: (num_blocks, block_size)
    ret_blocks = ret_arr.reshape(num_blocks, block_size)

    rng = np.random.default_rng(seed)
    permuted_ics = np.zeros(n_perms)

    # 4. The Permutation Loop
    for i in range(n_perms):
        # Shuffle the order of the blocks (the rows)
        shuffled_indices = rng.permutation(num_blocks)
        shuffled_ret_blocks = ret_blocks[shuffled_indices]

        # Flatten back to 1D and calculate random IC
        shuffled_ret = shuffled_ret_blocks.flatten()
        perm_ic, _ = spearmanr(sig_arr, shuffled_ret)
        permuted_ics[i] = perm_ic

    # 5. P-value: What % of random permutations beat our actual IC?
    # (Assuming we are looking for a positive correlation edge)
    p_value = np.sum(permuted_ics >= actual_ic) / n_perms

    return {
        "actual_ic": actual_ic,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "permuted_ics": permuted_ics
    }

# Permutation - How stable is the edge
def bootstrap_sharpe(returns: pd.Series, n_bootstraps: int = 1000, annualization_factor: int = 252,
                     seed: int | None = 0) -> dict:
    """ 5% Confidence intervals for sharpe ratio stability by reshuffling n times"""
    ret_arr = returns.dropna().values
    n_obs = len(ret_arr)
    rng = np.random.default_rng(seed)
    bootstrapped_sharpes = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        # Sample with replacement
        sample = rng.choice(ret_arr, size=n_obs, replace=True)

        # Calculate annualized Sharpe
        mean_ret = np.mean(sample)
        std_ret = np.std(sample)

        sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor) if std_ret > 0 else 0
        bootstrapped_sharpes[i] = sharpe

    return {
        "actual_sharpe": (np.mean(ret_arr) / np.std(ret_arr)) * np.sqrt(annualization_factor),
        "ci_lower": np.percentile(bootstrapped_sharpes, 2.5),
        "ci_upper": np.percentile(bootstrapped_sharpes, 97.5),
        "bootstrapped_sharpes": bootstrapped_sharpes
    }


# Autocorrelation Validation via ljung box
def ljung_box(
    series: pd.Series,
    lags: list[int] | None = None,
    n_plot_lags: int = 60,
) -> dict[str, Any]:
    """
    Formally test the joint null that a series has no autocorrelation.

    Parameters
    ----------
    series      : per-period returns (not levels, not cumulative)
    lags        : lag horizons to test jointly; defaults to [5, 10, 20]
    n_plot_lags : horizon for the band-exceedance count

    Returns
    -------
    dict with keys: n (int), lags (DataFrame of lb_stat/lb_pvalue for the
                    level series), squared (DataFrame, same for squared),
                    white_noise_5pct (bool), acf (ndarray), band (float),
                    n_outside_band (int), expected_outside_band (float)
    """
    if acorr_ljungbox is None or acf is None:
        raise ImportError("statsmodels is not installed")

    clean = series.dropna()
    if clean.empty:
        raise ValueError("series has no non-null values")
    lags = lags or [5, 10, 20]

    level = acorr_ljungbox(clean, lags=lags, return_df=True)
    squared = acorr_ljungbox(clean ** 2, lags=lags, return_df=True)

    n = int(clean.size)
    n_plot_lags = min(n_plot_lags, n // 2)
    correlations = acf(clean, nlags=n_plot_lags, fft=True)[1:]
    band = 1.96 / np.sqrt(n)

    return {
        "n": n,
        "lags": level,
        "squared": squared,
        "white_noise_5pct": bool((level["lb_pvalue"] > 0.05).all()),
        "acf": correlations,
        "band": float(band),
        "n_outside_band": int((np.abs(correlations) > band).sum()),
        "expected_outside_band": 0.05 * len(correlations),
    }


def print_ljung_box(series: pd.Series, label: str, **kwargs) -> dict[str, Any]:
    """Run `ljung_box` and print it as one readable block. Returns the dict."""
    result = ljung_box(series, **kwargs)

    print(f"{label}  (n={result['n']:,})")
    table = pd.DataFrame({
        "lb_p_level": result["lags"]["lb_pvalue"],
        "lb_p_squared": result["squared"]["lb_pvalue"],
    })
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"  ACF bars outside +/-{result['band']:.4f}: "
        f"{result['n_outside_band']}/{len(result['acf'])} "
        f"(expected by chance: {result['expected_outside_band']:.0f})"
    )
    verdict = "no autocorrelation in the mean" if result["white_noise_5pct"] \
        else "AUTOCORRELATION PRESENT in the mean"
    print(f"  -> {verdict}")
    if not bool((result["squared"]["lb_pvalue"] > 0.05).all()):
        print("  -> squared returns ARE autocorrelated (volatility clustering)")
    return result
