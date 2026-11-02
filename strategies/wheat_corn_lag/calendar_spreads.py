"""
calendar_spreads.py
-------------------
Model 2b on CORN (proof of method; wheat follows when contract-level ZW data arrives): does the balance
sheet explain and predict CALENDAR SPREADS, measured as % of full carry? Development data only (everything
is cut at the holdout start before it is used). Saved to runs/<time>_calendar_spreads/.

    python -m strategies.wheat_corn_lag.calendar_spreads

Theory (supply of storage, Working's curve): ample stocks -> the deferred contract pays for storage up to
FULL CARRY (cash-and-carry arbitrage caps it); tight stocks -> the nearby is bid, the spread falls below full
carry and can invert. So % of full carry rises with stocks-to-use and saturates near 100%.

Data
  contracts   Databento ZC outrights, hourly -> each contract's last close per session (2010 on)
  spreads     the NEARBY intra-crop-year storage spreads only: Dec/Mar, Mar/May, May/Jul (Jul/Sep and Sep/Dec
              mix old and new crop and do not follow storage logic). The nearby switches 5 business days
              before its first notice day (FND = last business day of the month before delivery).
              spread = deferred - nearby (cents; + = carry)
  full carry  storage + financing over the days between the two delivery months
              storage  0.165 c/bu/day, the CBOT corn delivery storage rate (assumption: constant over the sample;
                       to verify against the rulebook history)
              finance  nearby price x (13-week T-bill + 1%) x days / 365   (Treasury daily bill rates)
  stocks      WASDE U.S. corn ending stocks / total use for that crop year (Sep-Aug), from the latest report
              released by that day (a 12:00 ET release is in that day's close)

Weekly samples (last trading day of each week). Target: the change in the SAME pair's spread (cents) over the
next 1, 2, 4 weeks, only while that pair is still the nearby.

Models (expanding window: refit every week on past weeks only, at least 52)
  Working curve   %FC = 1 - A exp(-k STU)          saturating at FULL CARRY (the arbitrage ceiling is fixed at
                  100%, not fitted: a free ceiling ran to 300% because the 2012 / 2020-22 inversions, down to
                  -290% of full carry, dominate squared errors), fitted with a robust (soft L1) loss
  log             %FC = a + b log(STU)
  control         %FC = average of past weeks       (no balance sheet)
Tests: level fit (R2 in sample, out-of-sample error vs control) and whether the residual (spread vs its fair
value) predicts the spread's next move (rank IC, Newey-West t with lags = horizon in weeks).
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from core_quant.config import ROOT, save_run
from core_quant.theme import set_theme
from core_quant.validation import nw_t_stat
from strategies.wheat_corn_lag.balance_sheet import load_wasde
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

STORAGE_CENTS_PER_DAY = 0.165
FINANCE_SPREAD = 0.01
PAIRS = {"Z": ("H", 0), "H": ("K", 1), "K": ("N", 1)}     # nearby month -> (deferred month, crop-year offset)
MONTH_NUMBER = {"H": 3, "K": 5, "N": 7, "U": 9, "Z": 12}
SWITCH_BUSINESS_DAYS = 5
MIN_WEEKS = 52
HORIZONS = [1, 2, 4]


# ------------------------------------------------------------------ inputs
def contract_closes(cfg, end: pd.Timestamp) -> pd.DataFrame:
    """Last close per session (UTC date of the bar end) per outright contract, columns like 'H2012'."""
    import databento

    hourly = databento.DBNStore.from_file(ROOT / "data" / "raw" / cfg.corn_contracts_file).to_df(map_symbols=True)
    hourly = hourly[hourly["symbol"].str.fullmatch(r"ZC[HKNUZ]\d")]
    session = (hourly.index + pd.Timedelta(hours=1)).tz_convert(None).normalize()
    hourly = hourly[session < end.tz_localize(None)]
    session = session[session < end.tz_localize(None)]
    digit = hourly["symbol"].str[-1].astype(int).values
    # the contract's year: the first year >= the session's year that ends in that digit
    year = session.year + (digit - session.year % 10) % 10
    names = hourly["symbol"].str[2].values + year.astype(str)
    return pd.Series(hourly["close"].values, index=pd.MultiIndex.from_arrays([session, names])).groupby(level=[0, 1]).last().unstack()


def first_notice_day(month: str, year: int) -> pd.Timestamp:
    return pd.Timestamp(year, MONTH_NUMBER[month], 1) - pd.offsets.BDay(1)


def tbill_rate(end: pd.Timestamp) -> pd.Series:
    parts = []
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "tbill" / "tbill_*.csv"))):
        frame = pd.read_csv(path)
        column = [c for c in frame.columns if "13 WEEKS COUPON" in c.upper()][0]
        parts.append(pd.Series(frame[column].values, index=pd.to_datetime(frame["Date"], format="mixed")))
    rate = pd.concat(parts).sort_index() / 100
    return rate[rate.index < end.tz_localize(None)]


def corn_stocks_by_crop_year(end: pd.Timestamp) -> pd.DataFrame:
    frame = load_wasde()
    rows = frame[(frame["ReportTitle"] == "U.S. Feed Grain and Corn Supply and Use") & (frame["Commodity"] == "Corn")
                 & (frame["region"] == "United States") & frame["attribute"].isin(["ending stocks", "use, total"])]
    table = rows.pivot_table(index=["release", "MarketYear"], columns="attribute", values="value", aggfunc="first").reset_index()
    table["crop_year"] = table["MarketYear"].str.slice(0, 4).astype(int)
    table["stu"] = table["ending stocks"] / table["use, total"]
    table["release"] = table["release"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return table[table["release"] < end.tz_convert("America/New_York").tz_localize(None)]


# ------------------------------------------------------------------ weekly panel
def nearby_pair(day: pd.Timestamp) -> tuple[str, str, int, pd.Timestamp] | None:
    """(nearby, deferred, crop year, nearby FND) for the storage spread in force on `day`, or None."""
    candidates = []
    for year in (day.year, day.year + 1):
        for month in ("H", "K", "N", "U", "Z"):
            fnd = first_notice_day(month, year)
            if fnd - pd.offsets.BDay(SWITCH_BUSINESS_DAYS) > day:
                candidates.append((fnd, month, year))
    fnd, month, year = min(candidates)
    if month not in PAIRS:
        return None                                      # nearby is Jul or Sep: no intra-crop storage spread
    deferred_month, offset = PAIRS[month]
    deferred_year = year + 1 if month == "Z" else year
    crop_year = year if month == "Z" else year - 1
    return f"{month}{year}", f"{deferred_month}{deferred_year}", crop_year, fnd


def weekly_panel(closes: pd.DataFrame, rates: pd.Series, stocks: pd.DataFrame) -> pd.DataFrame:
    weeks = closes.index.to_series().groupby(closes.index.to_period("W-FRI")).max().values
    rows = []
    for day in pd.DatetimeIndex(weeks):
        pair = nearby_pair(day)
        if pair is None:
            continue
        nearby, deferred, crop_year, fnd = pair
        if nearby not in closes or deferred not in closes:
            continue
        near_price, far_price = closes.at[day, nearby], closes.at[day, deferred]
        rate = rates[rates.index <= day]
        known = stocks[(stocks["release"] <= day + pd.Timedelta(hours=14)) & (stocks["crop_year"] == crop_year)]
        if not (np.isfinite(near_price) and np.isfinite(far_price)) or rate.empty or known.empty:
            continue
        delivery_gap = (pd.Timestamp(int(deferred[1:]), MONTH_NUMBER[deferred[0]], 1)
                        - pd.Timestamp(int(nearby[1:]), MONTH_NUMBER[nearby[0]], 1)).days
        full_carry = STORAGE_CENTS_PER_DAY * delivery_gap + near_price * (rate.iloc[-1] + FINANCE_SPREAD) * delivery_gap / 365
        spread = far_price - near_price
        row = {"day": day, "pair": f"{nearby}/{deferred}", "crop_year": crop_year, "spread": spread,
               "full_carry": full_carry, "pct_fc": spread / full_carry, "stu": known.sort_values("release")["stu"].iloc[-1],
               "nearby_price": near_price}
        # target: the same pair's spread change k weeks later, only while it is still the nearby pair
        for k in HORIZONS:
            later = day + pd.Timedelta(weeks=k)
            later_day = closes.index[closes.index <= later][-1] if (closes.index <= later).any() else None
            still = later_day is not None and later_day > day and nearby_pair(later_day) is not None \
                and nearby_pair(later_day)[0] == nearby
            row[f"d_spread_{k}w"] = (closes.at[later_day, deferred] - closes.at[later_day, nearby] - spread) if still else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("day")


# ------------------------------------------------------------------ models
def working_curve(stu, size, rate):
    """% of full carry: approaches 100% (full carry) as stocks grow, falls steeply as they tighten."""
    return 1.0 - size * np.exp(-rate * stu)


def fit_working(stu: np.ndarray, pct_fc: np.ndarray) -> np.ndarray:
    """Robust fit (soft L1): a few extreme inversion weeks cannot drag the whole curve."""
    fit = least_squares(lambda theta: working_curve(stu, *theta) - pct_fc, x0=[5.0, 20.0], loss="soft_l1",
                        f_scale=0.3, bounds=([0, 0], [1e4, 300]))
    return fit.x


def expanding_fits(panel: pd.DataFrame) -> pd.DataFrame:
    fair = pd.DataFrame(index=panel.index, columns=["working", "log", "control"], dtype=float)
    for i in range(MIN_WEEKS, len(panel)):
        past = panel.iloc[:i]
        stu_now = panel["stu"].iloc[i]
        fair.iloc[i, 2] = past["pct_fc"].mean()
        b, a = np.polyfit(np.log(past["stu"]), past["pct_fc"], 1)
        fair.iloc[i, 1] = a + b * np.log(stu_now)
        fair.iloc[i, 0] = working_curve(stu_now, *fit_working(past["stu"].values, past["pct_fc"].values))
    return fair


def evaluate(panel: pd.DataFrame, fair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in fair.columns:
        residual = panel["pct_fc"] - fair[model]
        row = {"model": model, "weeks": int(residual.notna().sum()), "oos_rmse_pct_fc": float(np.sqrt((residual ** 2).mean()))}
        for k in HORIZONS:
            block = pd.DataFrame({"signal": -residual, "y": panel[f"d_spread_{k}w"]}).dropna()
            row[f"ic_{k}w"] = spearmanr(block["signal"], block["y"])[0]
            row[f"t_{k}w"] = float(nw_t_stat(block, ["signal"], "y", lags=k).loc["signal", "t_nw"])
            row[f"n_{k}w"] = len(block)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def curve_chart(panel: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    scatter = axis.scatter(panel["stu"], panel["pct_fc"] * 100, c=panel.index.year, cmap="viridis", s=12)
    params = fit_working(panel["stu"].values, panel["pct_fc"].values)
    grid = np.linspace(panel["stu"].min(), panel["stu"].max(), 100)
    rank = spearmanr(panel["stu"], panel["pct_fc"])[0]
    axis.plot(grid, working_curve(grid, *params) * 100, lw=2, color="black",
              label=f"Working curve, ceiling = full carry (rank corr {rank:+.2f})")
    axis.axhline(100, color="grey", ls=":", lw=1, label="full carry")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xlabel("U.S. corn stocks-to-use, crop year of the spread (WASDE, as known)")
    axis.set_ylabel("nearby storage spread, % of full carry")
    axis.set_title("Corn: supply-of-storage curve, weekly, Dec/Mar, Mar/May, May/Jul", loc="left")
    axis.legend(fontsize=8)
    figure.colorbar(scatter, ax=axis, label="year")
    figure.tight_layout()
    return figure


def path_chart(panel: pd.DataFrame, fair: pd.DataFrame) -> plt.Figure:
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, height_ratios=[1.6, 1])
    top.plot(panel.index, panel["pct_fc"] * 100, ".", ms=3, label="actual % of full carry")
    top.plot(fair.index, fair["working"] * 100, ".", ms=3, label="Working-curve fair value (out of sample)")
    top.plot(fair.index, fair["control"] * 100, ".", ms=2, alpha=0.6, label="control: past average")
    top.axhline(100, color="grey", ls=":", lw=1)
    top.set_ylabel("% of full carry")
    top.legend(fontsize=8)
    top.set_title("Corn nearby storage spread vs its balance-sheet fair value (gaps = Jul-Nov, no storage pair)", loc="left")
    bottom.plot(panel.index, panel["stu"], ".", ms=3)
    bottom.set_ylabel("stocks-to-use")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    closes = contract_closes(cfg, holdout)
    panel = weekly_panel(closes, tbill_rate(holdout), corn_stocks_by_crop_year(holdout))
    fair = expanding_fits(panel)
    table = evaluate(panel, fair)

    pd.set_option("display.width", 220)
    print(f"weekly observations: {len(panel)} ({panel.index[0].date()} -> {panel.index[-1].date()}), pairs: {panel['pair'].str[0].value_counts().to_dict()}")
    print(panel[["spread", "full_carry", "pct_fc", "stu"]].describe().round(3).to_string())
    params = fit_working(panel["stu"].values, panel["pct_fc"].values)
    fitted = working_curve(panel["stu"], *params)
    r2 = 1 - ((panel["pct_fc"] - fitted) ** 2).sum() / ((panel["pct_fc"] - panel["pct_fc"].mean()) ** 2).sum()
    print(f"\nin-sample Working curve (ceiling = full carry): size {params[0]:.2f}, rate {params[1]:.1f}; R2 {r2:.2f},"
          f" median abs error {np.median(np.abs(panel['pct_fc'] - fitted)):.3f}; rank corr STU vs %FC {spearmanr(panel['stu'], panel['pct_fc'])[0]:+.2f}")
    print("rank corr STU vs %FC by nearby pair:",
          {pair: round(spearmanr(g["stu"], g["pct_fc"])[0], 2) for pair, g in panel.groupby(panel["pair"].str[0])})
    print("never above full carry: max %FC", round(panel["pct_fc"].max(), 3),
          "| inverted crop years:", sorted(panel.loc[panel["pct_fc"] < 0, "crop_year"].unique().tolist()))
    print("\nout of sample (expanding, past weeks only):")
    print(table.round(3).to_string(index=False))

    figures = {"working_curve": curve_chart(panel), "path": path_chart(panel, fair)}
    run_dir = save_run(STRATEGY_DIR, "calendar_spreads", cfg, {"weeks": len(panel)},
                       {"panel": panel, "fair": fair, "models": table}, figures)
    print(f"\nsaved to {run_dir}")
