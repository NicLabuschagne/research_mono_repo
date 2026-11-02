"""
balance_sheet.py
----------------
Model 2: wheat's price against its balance sheet (WASDE), one observation per monthly report, point in time.
Development data only: daily prices are cut at the holdout start BEFORE anything is computed, so no target
reaches into 2026. Saved to runs/<time>_balance_sheet/.

    python -m strategies.wheat_corn_lag.balance_sheet

Data
  WASDE   data/raw/wasde: every report as first published (Apr 2010 on). For each report we take the
          PROJECTION marketing year (the latest one in the table: current crop until April, new crop from May).
          stocks-to-use (STU):
            us       U.S. ending stocks / total use
            wxc      World Less China ending stocks / domestic use        (China's stocks do not trade)
            mex      Major Exporters ending stocks / (domestic use + exports)
          corn: us and wxc the same way (for the spread).
  Prices  daily front-month closes (session date, last trade), from Apr 2017. A report released 12:00 ET is in
          that day's close (CBOT closes 14:20 ET), so the report's price = the close on its release date.
          Forward returns, k = 21 / 42 / 63 trading days (about 1 / 2 / 3 months):
            wheat  raw front-month closes: wheat rolls are NOT removed (no contract data); they add 1-3% noise
            corn   roll-free: each day's return is the return of the contract the file tracked that day, from
                   Databento contract closes (the raw series jumps -21% at the 2021 and 2022 July->Dec rolls)

Models (the convexity: price rises faster as stocks get tight)
  log-log    log(price) = a + b log(STU)        price = A x STU^b, b < 0 = constant elasticity
  inverse    price      = a + b / STU           steeper as STU -> 0 (more convex at tight stocks)
  spread     log(wheat/corn) = a + b log(STU_wheat / STU_corn)
Each is fitted on an EXPANDING window: at report i, only reports before i (at least 24) are used, then
report i is priced. The residual  log(price / fair value)  is therefore out of sample.

Deflated ("real") variants: price / CPI known at the report. CPI = BLS CPI-U, not seasonally adjusted (it is
never revised; data/raw/cpi_u_nsa_bls.csv). Month m's CPI is used only from the 20th of month m+1 (BLS
publishes around the 10th-15th). Oct 2025 was never published: the previous month is carried forward.
The fit is in real terms; fair value in today's cents = real fair value x the CPI known today.

Controls: the same tests with NO balance sheet (fair value = average of past log prices). If a control does as
well, the IC is plain mean reversion of the price (or ratio), not balance-sheet information.

Tests on the out-of-sample residual (+ = rich vs the balance sheet, so the signal is -residual)
  fit         full-sample R2 of the level (descriptive), out-of-sample RMSE of log price vs fair value
  half-life   AR(1) of the residual from report to report, in months
  IC          rank IC of -residual with the k-day forward return, Newey-West t (lags = k / 21 reports)
Pass criteria, fixed before the first run: 2-month IC with NW t >= 2.5, same sign in >= 70% of years,
residual half-life < 12 months.
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from core_quant.config import ROOT, save_run
from core_quant.theme import set_theme
from core_quant.validation import nw_t_stat
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

WHEAT_DAILY = "WHEAT_futures_1d_2010-01-23_2026-10-30.parquet"
CORN_DAILY = "CORN_futures_1d_2010-01-23_2026-10-30.parquet"
HORIZONS = [21, 42, 63]
PRIMARY = 42
MIN_FIT = 24
PASS_T, PASS_YEAR_SHARE, PASS_HALF_LIFE = 2.5, 0.7, 12.0


# ------------------------------------------------------------------ WASDE panel
def load_wasde() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "wasde" / "archive" / "*.csv"))) + \
        sorted(glob.glob(str(ROOT / "data" / "raw" / "wasde" / "*.csv")))
    columns = ["ReportTitle", "Attribute", "Commodity", "Region", "MarketYear", "Value", "ReleaseDate", "ReleaseTime"]
    frame = pd.concat([pd.read_csv(f, dtype=str, usecols=columns) for f in files], ignore_index=True)
    frame["attribute"] = frame["Attribute"].str.lower().str.strip()
    frame["region"] = frame["Region"].str.strip().replace({"Major exporters": "Major Exporters"})
    frame["value"] = pd.to_numeric(frame["Value"], errors="coerce")
    release_day = pd.to_datetime(frame["ReleaseDate"], format="mixed")
    clock = pd.to_timedelta(frame["ReleaseTime"].str.slice(0, 8).fillna("12:00:00"))
    frame["release"] = (release_day + clock).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return frame


def table_values(frame, title: str, commodity: str, region: str) -> pd.DataFrame:
    """One row per report: the PROJECTION marketing year's attributes for one table and region."""
    rows = frame[(frame["ReportTitle"] == title) & (frame["Commodity"] == commodity) & (frame["region"] == region)]
    latest_year = rows.groupby("release")["MarketYear"].transform("max")
    rows = rows[rows["MarketYear"] == latest_year]
    return rows.pivot_table(index="release", columns="attribute", values="value", aggfunc="first")


def balance_sheet_panel(frame) -> pd.DataFrame:
    us_wheat = table_values(frame, "U.S. Wheat Supply and Use", "Wheat", "United States")
    world_wheat = table_values(frame, "World Wheat Supply and Use", "Wheat", "World Less China")
    exporters = table_values(frame, "World Wheat Supply and Use", "Wheat", "Major Exporters")
    us_corn = table_values(frame, "U.S. Feed Grain and Corn Supply and Use", "Corn", "United States")
    world_corn = table_values(frame, "World Corn Supply and Use", "Corn", "World Less China")
    return pd.DataFrame({
        "stu_us_wheat": us_wheat["ending stocks"] / us_wheat["use, total"],
        "stu_wxc_wheat": world_wheat["ending stocks"] / world_wheat["domestic total"],
        "stu_mex_wheat": exporters["ending stocks"] / (exporters["domestic total"] + exporters["exports"]),
        "stu_us_corn": us_corn["ending stocks"] / us_corn["use, total"],
        "stu_wxc_corn": world_corn["ending stocks"] / world_corn["domestic total"],
    }).sort_index()


# ------------------------------------------------------------------ prices
def daily_closes(file_name: str, end: pd.Timestamp) -> pd.Series:
    daily = pd.read_parquet(ROOT / "data" / "raw" / file_name)
    closes = pd.Series(daily["close"].values, index=pd.to_datetime(daily["date"]))
    return closes[closes.index < end.tz_localize(None)]                  # holdout cut BEFORE anything else


def roll_free_corn_returns(cfg, corn_close: pd.Series) -> pd.Series:
    """Daily corn returns with the rolls taken out: identify the contract the file tracks each day (its close
    within 0.5 c of one Databento contract and >= 1.5 c from the next), and use THAT contract's own return."""
    import databento

    hourly = databento.DBNStore.from_file(ROOT / "data" / "raw" / cfg.corn_contracts_file).to_df(map_symbols=True)
    hourly = hourly[hourly["symbol"].str.fullmatch(r"ZC[HKNUZ]\d")]
    session = (hourly.index + pd.Timedelta(hours=1)).tz_convert(None).normalize()
    contract_close = hourly.groupby([session, hourly["symbol"]])["close"].last().unstack()
    contract_close = contract_close.reindex(corn_close.index)
    distance = contract_close.sub(corn_close, axis=0).abs()
    ordered = np.sort(distance.values, axis=1)
    clear = (ordered[:, 0] <= 0.5) & (ordered[:, 1] - ordered[:, 0] >= 1.5)
    tracked = distance.idxmin(axis=1).where(clear).ffill()                 # unclear days keep the last contract
    returns = pd.Series(np.nan, index=corn_close.index)
    log_close = np.log(contract_close)
    for day, previous in zip(corn_close.index[1:], corn_close.index[:-1]):
        contract = tracked.get(day)
        if isinstance(contract, str):
            returns[day] = log_close.at[day, contract] - log_close.at[previous, contract]
    return returns


def forward_returns(log_index: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Log change from the close on each report date (or the next trading day) to k trading days later."""
    positions = log_index.index.searchsorted(dates)
    out = {}
    for k in HORIZONS:
        values = []
        for p in positions:
            values.append(log_index.iloc[p + k] - log_index.iloc[p] if p + k < len(log_index) else np.nan)
        out[f"fwd_{k}"] = values
    return pd.DataFrame(out, index=dates)


def cpi_known(dates: pd.DatetimeIndex) -> pd.Series:
    """CPI level known on each date (point in time): month m usable from the 20th of month m+1."""
    cpi = pd.read_csv(ROOT / "data" / "raw" / "cpi_u_nsa_bls.csv")
    month = pd.PeriodIndex(cpi["month"], freq="M")
    usable_from = (month + 1).to_timestamp() + pd.Timedelta(days=19)
    series = pd.Series(cpi["cpi"].values, index=usable_from).sort_index()
    positions = series.index.searchsorted(dates, side="right") - 1
    values = np.where(positions >= 0, series.values[np.maximum(positions, 0)], np.nan)
    return pd.Series(values, index=dates)


# ------------------------------------------------------------------ models
def expanding_fair_value(y: pd.Series, x: pd.Series | None, inverse: bool = False) -> pd.Series:
    """Out-of-sample fair value of log price: fit on reports before i, price report i.
    x=None is the CONTROL: no balance sheet, fair value = the average of past log prices."""
    fair = pd.Series(np.nan, index=y.index)
    for i in range(MIN_FIT, len(y)):
        if x is None:
            fair.iloc[i] = y.iloc[:i].mean()
            continue
        past = pd.DataFrame({"y": y.iloc[:i], "x": x.iloc[:i]}).dropna()
        if len(past) < MIN_FIT or not np.isfinite(x.iloc[i]):
            continue
        if inverse:   # level price = a + b / STU, compared in logs
            fit = sm.OLS(np.exp(past["y"]), sm.add_constant(1 / past["x"])).fit()
            level = fit.params.iloc[0] + fit.params.iloc[1] / x.iloc[i]
            fair.iloc[i] = np.log(level) if level > 0 else np.nan
        else:
            fit = sm.OLS(past["y"], sm.add_constant(np.log(past["x"]))).fit()
            fair.iloc[i] = fit.params.iloc[0] + fit.params.iloc[1] * np.log(x.iloc[i])
    return fair


def evaluate(name: str, log_price: pd.Series, fair: pd.Series, targets: pd.DataFrame, full_r2: float) -> dict:
    residual = (log_price - fair).rename("residual")
    data = pd.concat([-residual.rename("signal"), targets], axis=1).dropna(subset=["signal"])
    row = {"model": name, "reports_oos": int(data["signal"].notna().sum()), "R2_full_sample": full_r2,
           "oos_rmse_log": float(np.sqrt((residual.dropna() ** 2).mean()))}
    lagged = residual.shift(1)
    pair = pd.concat([residual, lagged], axis=1).dropna()
    phi = np.polyfit(pair.iloc[:, 1], pair.iloc[:, 0], 1)[0]
    row["half_life_months"] = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
    for k in HORIZONS:
        block = data[["signal", f"fwd_{k}"]].dropna()
        row[f"ic_{k}"] = spearmanr(block["signal"], block[f"fwd_{k}"])[0]
        row[f"t_{k}"] = float(nw_t_stat(block, ["signal"], f"fwd_{k}", lags=max(k // 21, 1)).loc["signal", "t_nw"])
    block = data[["signal", f"fwd_{PRIMARY}"]].dropna()
    yearly = block.groupby(block.index.year).apply(lambda b: spearmanr(b["signal"], b[f"fwd_{PRIMARY}"])[0])
    row["years_same_sign"] = f"{int((np.sign(yearly) == np.sign(row[f'ic_{PRIMARY}'])).sum())}/{len(yearly)}"
    row["PASS"] = bool(row[f"t_{PRIMARY}"] >= PASS_T
                       and (np.sign(yearly) == np.sign(row[f"ic_{PRIMARY}"])).mean() >= PASS_YEAR_SHARE
                       and row["half_life_months"] < PASS_HALF_LIFE)
    return row, residual


def full_sample_r2(y: pd.Series, x: pd.Series, inverse: bool) -> float:
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if inverse:
        return float(sm.OLS(np.exp(data["y"]), sm.add_constant(1 / data["x"])).fit().rsquared)
    return float(sm.OLS(data["y"], sm.add_constant(np.log(data["x"]))).fit().rsquared)


# ------------------------------------------------------------------ charts
def curve_chart(panel: pd.DataFrame, pairs: list[tuple[str, str, str]]) -> plt.Figure:
    """Price against stocks-to-use, one dot per report coloured by year, with both fitted curves."""
    figure, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 4.6))
    years = panel.index.year
    for axis, (stu, price, title, x_label) in zip(axes, pairs):
        data = panel[[stu, price]].dropna()
        scatter = axis.scatter(data[stu], data[price], c=data.index.year, cmap="viridis", s=18,
                               vmin=years.min(), vmax=years.max())        # one colour scale for every panel
        grid = np.linspace(data[stu].min(), data[stu].max(), 100)
        loglog = sm.OLS(np.log(data[price]), sm.add_constant(np.log(data[stu]))).fit().params
        inverse = sm.OLS(data[price], sm.add_constant(1 / data[stu])).fit().params
        axis.plot(grid, np.exp(loglog.iloc[0]) * grid ** loglog.iloc[1], lw=1.6, label=f"log-log (elasticity {loglog.iloc[1]:.2f})")
        axis.plot(grid, inverse.iloc[0] + inverse.iloc[1] / grid, lw=1.6, ls="--", label="inverse")
        axis.set_xlabel(x_label)
        axis.set_ylabel("price at release (c/bu)" if "ratio" not in title else "wheat / corn price")
        axis.set_title(title, loc="left", fontsize=10)
        axis.legend(fontsize=8)
    figure.colorbar(scatter, ax=axes, shrink=0.8, label="year")
    return figure


def path_chart(log_price: pd.Series, fair: pd.Series, residual: pd.Series, title: str) -> plt.Figure:
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1.6, 1])
    top.plot(log_price.index, np.exp(log_price), marker="o", ms=3, lw=1.2, label="price at each report")
    top.plot(fair.index, np.exp(fair), marker="o", ms=3, lw=1.2, label="balance-sheet fair value (out of sample)")
    top.set_title(title, loc="left")
    top.legend(fontsize=8)
    bottom.bar(residual.index, residual.values, width=20, color=np.where(residual > 0, "#d62728", "#2ca02c"))
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylabel("log(price / fair): + = rich")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    panel = balance_sheet_panel(load_wasde())
    panel = panel[panel.index < holdout]

    wheat = daily_closes(WHEAT_DAILY, holdout)
    corn = daily_closes(CORN_DAILY, holdout)
    report_days = pd.DatetimeIndex(panel.index.tz_convert("America/New_York").tz_localize(None).normalize())
    panel = panel[(report_days >= wheat.index[0])]
    report_days = report_days[report_days >= wheat.index[0]]
    at_report = wheat.index.searchsorted(report_days)                      # release day's close (or the next day's)
    panel["wheat"] = wheat.iloc[at_report].values
    panel["corn"] = corn.reindex(wheat.index).iloc[at_report].values
    panel["ratio"] = panel["wheat"] / panel["corn"]

    corn_returns = roll_free_corn_returns(cfg, corn)
    wheat_index = np.log(wheat)
    corn_index = corn_returns.fillna(0).cumsum().reindex(wheat.index).ffill()
    targets_wheat = forward_returns(wheat_index, report_days).set_index(panel.index)
    targets_spread = forward_returns(wheat_index - corn_index, report_days).set_index(panel.index)

    rows, residuals, fairs = [], {}, {}
    outright = [("us", "U.S."), ("wxc", "World Less China"), ("mex", "Major Exporters")]
    log_wheat = np.log(panel["wheat"])
    for key, label in outright:
        stu = panel[f"stu_{key}_wheat"]
        for inverse in (False, True):
            name = f"wheat vs {label} STU, {'inverse' if inverse else 'log-log'}"
            fair = expanding_fair_value(log_wheat, stu, inverse)
            row, residual = evaluate(name, log_wheat, fair, targets_wheat, full_sample_r2(log_wheat, stu, inverse))
            rows.append(row); residuals[name] = residual; fairs[name] = fair
    # deflated: the same U.S. models and control on log(price / CPI known at the report)
    panel["cpi"] = cpi_known(report_days).values
    log_real = log_wheat - np.log(panel["cpi"])
    for inverse in (False, True):
        name = f"REAL wheat vs U.S. STU, {'inverse' if inverse else 'log-log'}"
        fair = expanding_fair_value(log_real, panel["stu_us_wheat"], inverse)
        row, residual = evaluate(name, log_real, fair, targets_wheat, full_sample_r2(log_real, panel["stu_us_wheat"], inverse))
        rows.append(row); residuals[name] = residual; fairs[name] = fair
    fair = expanding_fair_value(log_real, None)
    row, residual = evaluate("CONTROL REAL wheat, no balance sheet", log_real, fair, targets_wheat, 0.0)
    rows.append(row); residuals["CONTROL REAL wheat, no balance sheet"] = residual
    for name, y, targets in [("CONTROL wheat, no balance sheet (past average)", log_wheat, targets_wheat),
                             ("CONTROL spread, no balance sheet (past average)", np.log(panel["ratio"]), targets_spread)]:
        fair = expanding_fair_value(y, None)
        row, residual = evaluate(name, y, fair, targets, 0.0)
        rows.append(row); residuals[name] = residual; fairs[name] = fair
    log_ratio = np.log(panel["ratio"])
    for key, label in [("us", "U.S."), ("wxc", "World Less China")]:
        relative = panel[f"stu_{key}_wheat"] / panel[f"stu_{key}_corn"]
        name = f"SPREAD wheat/corn vs {label} relative STU, log-log"
        fair = expanding_fair_value(log_ratio, relative)
        row, residual = evaluate(name, log_ratio, fair, targets_spread, full_sample_r2(log_ratio, relative, False))
        rows.append(row); residuals[name] = residual; fairs[name] = fair
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 240)
    print(f"reports: {len(panel)} ({panel.index[0].date()} -> {panel.index[-1].date()}), out of sample from report {MIN_FIT + 1}")
    print(panel[["stu_us_wheat", "stu_wxc_wheat", "stu_mex_wheat", "stu_us_corn", "stu_wxc_corn"]].describe().round(3).to_string())
    print("\n" + table.round(3).to_string(index=False))

    best = table.loc[table["t_42"].idxmax(), "model"]
    figures = {
        "curves": curve_chart(panel.assign(stu_rel_wxc=panel["stu_wxc_wheat"] / panel["stu_wxc_corn"]),
                              [("stu_us_wheat", "wheat", "U.S.: wheat price vs U.S. STU", "U.S. wheat stocks-to-use"),
                               ("stu_wxc_wheat", "wheat", "World ex-China: wheat price vs world STU",
                                "World ex-China wheat stocks-to-use (from May 2019)"),
                               ("stu_rel_wxc", "ratio", "World ex-China: wheat/corn ratio vs relative STU",
                                "wheat STU / corn STU, World ex-China")]),
        "path_best": path_chart(log_wheat if "SPREAD" not in best else log_ratio, fairs[best], residuals[best], best),
    }
    run_dir = save_run(STRATEGY_DIR, "balance_sheet", cfg, {"reports": len(panel), "min_fit": MIN_FIT},
                       {"models": table, "panel": panel, "residuals": pd.DataFrame(residuals)}, figures)
    print(f"\nsaved to {run_dir}")
