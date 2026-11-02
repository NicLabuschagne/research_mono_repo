"""
nowcast.py
----------
Model 2, weekly: update the U.S. wheat balance sheet between WASDE reports from weekly export INSPECTIONS
(actual shipments), and test whether that helps. Development data only. Saved to runs/<time>_nowcast/.

    python -m strategies.wheat_corn_lag.nowcast

Data: AMS/FGIS certificates (data/raw/inspections/CY*.csv), all wheat classes (WASDE's U.S. wheat is all
classes), pounds / 60 = bushels. Inspections are 94-99% of WASDE's final exports (rail to Mexico and Canada is
mostly not port-inspected). A week ends on Thursday and is published the following MONDAY 11:00 ET, so it is
usable from that Monday's close (CBOT closes 14:20 ET). Export SALES would be the forward-looking half but
their API needs a key.

The nowcast at week k of marketing year y (June - May), everything from the past only:
  s_k     share of a year's inspections normally done by week k (average of earlier, complete years)
  r       inspections / WASDE final exports (average of years whose final number is settled, y-2 and older)
  pace    shipped so far vs what USDA's export number implies by now:
          surprise_k = cumulative_k / r  -  s_k x USDA exports(y)        (mbu; + = exports running AHEAD)
  implied exports = USDA exports + surprise; adjusted ending stocks = USDA ending stocks - surprise
  adjusted STU on the projection marketing year (in May the new crop inherits the surprise as lower carry-in)
  weekly fair value = the nominal log-log U.S. STU model (fitted on monthly reports before the week) at the
  adjusted STU

Tests
  1. Does the surprise known just before a WASDE predict USDA's export revision in that WASDE?
  2. Weekly residual (price vs nowcast fair value) vs the plain monthly fair value: IC on 1 and 2-month returns
  3. Crude next-week forecast of inspections: USDA pace (r x exports x (s_k+1 - s_k)), last week, 4-week
     average: mean absolute error
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
from strategies.wheat_corn_lag.balance_sheet import WHEAT_DAILY, daily_closes, load_wasde
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load

FIRST_MY = 2016                  # first complete marketing year in the files (CY2016 starts in January)
MIN_PAST_YEARS = 2
MIN_FIT = 24
HORIZONS = [21, 42]


# ------------------------------------------------------------------ data
def weekly_inspections(end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "inspections" / "CY*.csv"))):
        frame = pd.read_csv(path, usecols=lambda c: c.strip() in ["Thursday", "Grain", "Pounds"], dtype=str,
                            encoding="latin-1")
        frame.columns = [c.strip() for c in frame.columns]
        parts.append(frame[frame["Grain"].str.strip().str.upper() == "WHEAT"])
    wheat = pd.concat(parts, ignore_index=True)
    wheat["week"] = pd.to_datetime(wheat["Thursday"].str.strip(), format="mixed")
    wheat["mbu"] = pd.to_numeric(wheat["Pounds"], errors="coerce") / 60 / 1e6
    weekly = wheat.groupby("week")["mbu"].sum().to_frame()
    weekly["published"] = weekly.index + pd.Timedelta(days=4)                  # the following Monday
    weekly = weekly[weekly["published"] < end.tz_localize(None)]                 # holdout cut
    weekly["my"] = np.where(weekly.index.month >= 6, weekly.index.year, weekly.index.year - 1)
    weekly = weekly[weekly["my"] >= FIRST_MY]
    weekly["k"] = weekly.groupby("my").cumcount()                               # week number in the marketing year
    weekly["cumulative"] = weekly.groupby("my")["mbu"].cumsum()
    return weekly


def us_wheat_by_year(frame) -> pd.DataFrame:
    """Every report x marketing year: exports, ending stocks, total use (as published in that report)."""
    rows = frame[(frame["ReportTitle"] == "U.S. Wheat Supply and Use") & (frame["region"] == "United States")
                 & frame["attribute"].isin(["exports", "ending stocks", "use, total"])]
    table = rows.pivot_table(index=["release", "MarketYear"], columns="attribute", values="value", aggfunc="first")
    table = table.reset_index()
    table["my"] = table["MarketYear"].str.slice(0, 4).astype(int)
    table["projection"] = table.groupby("release")["my"].transform("max") == table["my"]
    return table


# ------------------------------------------------------------------ nowcast
def build_nowcast(weekly: pd.DataFrame, wasde: pd.DataFrame) -> pd.DataFrame:
    totals = weekly.groupby("my")["mbu"].sum()
    finals = wasde.sort_values("release").groupby("my")["exports"].last()
    shares = weekly.assign(share=weekly["cumulative"] / weekly["my"].map(totals)).pivot_table(
        index="k", columns="my", values="share")
    releases = wasde["release"].dt.tz_convert(None).drop_duplicates().sort_values()
    rows = []
    for week, row in weekly.iterrows():
        year, k = row["my"], row["k"]
        past = [y for y in shares.columns if y < year]
        settled = [y for y in totals.index if y <= year - 2]
        if len(past) < MIN_PAST_YEARS or not settled:
            continue
        s_k = shares.loc[k, past].mean() if k in shares.index else 1.0
        s_next = shares.loc[k + 1, past].mean() if k + 1 in shares.index else s_k
        r = (totals[settled] / finals[settled]).mean()
        known = releases[releases <= row["published"] + pd.Timedelta(hours=11)]          # reports out by then
        latest = wasde[wasde["release"].dt.tz_convert(None) == known.iloc[-1]]
        current = latest[latest["my"] == year]
        projection = latest[latest["projection"]].iloc[0]
        if current.empty:
            continue
        current = current.iloc[0]
        surprise = row["cumulative"] / r - s_k * current["exports"]
        stocks = projection["ending stocks"] - surprise
        use = projection["use, total"] + (surprise if projection["my"] == year else 0.0)
        rows.append({"week": week, "published": row["published"], "my": year, "k": k, "inspections": row["mbu"],
                     "cumulative": row["cumulative"], "usda_exports": current["exports"], "s_k": s_k, "r": r,
                     "surprise": surprise, "stu_wasde": projection["ending stocks"] / projection["use, total"],
                     "stu_nowcast": stocks / use, "forecast_usda_pace": r * current["exports"] * (s_next - s_k),
                     "report_in_use": known.iloc[-1]})
    table = pd.DataFrame(rows).set_index("week")
    table["next_week_actual"] = table.groupby("my")["inspections"].shift(-1)
    table["forecast_last_week"] = table["inspections"]
    table["forecast_4wk"] = table.groupby("my")["inspections"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    return table


def revision_test(nowcast: pd.DataFrame, wasde: pd.DataFrame) -> pd.DataFrame:
    """For each WASDE: the surprise known just before it vs the change USDA made to that year's exports."""
    rows = []
    releases = wasde["release"].dt.tz_convert(None).drop_duplicates().sort_values().tolist()
    for before, release in zip(releases[:-1], releases[1:]):
        known = nowcast[(nowcast["published"] + pd.Timedelta(hours=11) < release) & (nowcast["report_in_use"] == before)]
        if known.empty:
            continue
        last = known.iloc[-1]
        old = wasde[(wasde["release"].dt.tz_convert(None) == before) & (wasde["my"] == last["my"])]["exports"]
        new = wasde[(wasde["release"].dt.tz_convert(None) == release) & (wasde["my"] == last["my"])]["exports"]
        if len(old) and len(new):
            rows.append({"release": release, "my": last["my"], "surprise_before": last["surprise"],
                         "usda_revision": new.iloc[0] - old.iloc[0]})
    return pd.DataFrame(rows).set_index("release")


def fair_values(nowcast: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Weekly fair values from the log-log model fitted on monthly reports BEFORE the week (expanding)."""
    out = []
    for week, row in nowcast.iterrows():
        past = monthly[monthly.index < row["published"]].dropna()
        if len(past) < MIN_FIT:
            continue
        fit = sm.OLS(np.log(past["price"]), sm.add_constant(np.log(past["stu"]))).fit()
        a, b = fit.params
        out.append({"week": week, "fair_monthly": a + b * np.log(row["stu_wasde"]),
                    "fair_nowcast": a + b * np.log(row["stu_nowcast"])})
    return pd.DataFrame(out).set_index("week")


# ------------------------------------------------------------------ charts
def pace_chart(nowcast: pd.DataFrame) -> plt.Figure:
    years = sorted(nowcast["my"].unique())
    figure, axes = plt.subplots(2, (len(years) + 1) // 2, figsize=(16, 7), sharey=True)
    for axis, year in zip(axes.flat, years):
        block = nowcast[nowcast["my"] == year]
        axis.plot(block["k"], block["cumulative"] / block["r"], lw=1.6, label="shipped (inspections / r)")
        axis.plot(block["k"], block["s_k"] * block["usda_exports"], lw=1.2, ls="--", label="USDA pace")
        axis.set_title(f"{year}/{str(year + 1)[2:]}: final surprise {block['surprise'].iloc[-1]:+.0f} mbu", loc="left", fontsize=9)
        axis.set_xlabel("week of marketing year")
    axes.flat[0].set_ylabel("cumulative exports, mbu")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Export pace: actual shipments vs the pace USDA's projection implies", x=0.01, ha="left")
    figure.tight_layout()
    return figure


def revision_chart(revisions: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.5, 5))
    axis.scatter(revisions["surprise_before"], revisions["usda_revision"], s=22)
    axis.axhline(0, color="grey", lw=0.8)
    axis.axvline(0, color="grey", lw=0.8)
    ic = spearmanr(revisions["surprise_before"], revisions["usda_revision"])[0]
    axis.set_xlabel("export surprise known before the WASDE, mbu (+ = shipping ahead of USDA)")
    axis.set_ylabel("USDA's revision to exports in that WASDE, mbu")
    axis.set_title(f"Does the weekly pace predict USDA's revision? rank corr {ic:+.2f}", loc="left")
    figure.tight_layout()
    return figure


def fair_value_chart(merged: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13, 4.8))
    axis.plot(merged.index, merged["price"], lw=1.2, label="wheat (Monday close after inspections)")
    axis.plot(merged.index, np.exp(merged["fair_monthly"]), lw=1.2, label="fair value: monthly WASDE only")
    axis.plot(merged.index, np.exp(merged["fair_nowcast"]), lw=1.2, ls="--", label="fair value: nowcast with weekly pace")
    axis.set_ylabel("c/bu")
    axis.set_title("Wheat vs its balance-sheet fair value, monthly vs weekly-updated (out of sample)", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    frame = load_wasde()
    wasde = us_wheat_by_year(frame)
    wasde = wasde[wasde["release"] < holdout]
    weekly = weekly_inspections(holdout)
    nowcast = build_nowcast(weekly, wasde)

    # monthly fit data: price at each release (close of the release day) vs projection STU
    prices = daily_closes(WHEAT_DAILY, holdout)
    projection = wasde[wasde["projection"]].set_index("release")
    release_day = projection.index.tz_convert("America/New_York").tz_localize(None).normalize()
    usable = release_day >= prices.index[0]
    monthly = pd.DataFrame({"price": prices.iloc[prices.index.searchsorted(release_day[usable])].values,
                            "stu": (projection["ending stocks"] / projection["use, total"]).values[usable]},
                           index=projection.index.tz_convert(None)[usable])

    revisions = revision_test(nowcast, wasde)
    fair = fair_values(nowcast, monthly)
    at = prices.index.searchsorted(nowcast.loc[fair.index, "published"])
    merged = fair.assign(price=prices.iloc[at].values)
    log_price = np.log(prices)
    for k in HORIZONS:
        merged[f"fwd_{k}"] = [log_price.iloc[p + k] - log_price.iloc[p] if p + k < len(prices) else np.nan for p in at]
    merged["signal_monthly"] = -(np.log(merged["price"]) - merged["fair_monthly"])
    merged["signal_nowcast"] = -(np.log(merged["price"]) - merged["fair_nowcast"])

    pd.set_option("display.width", 200)
    print(f"nowcast weeks: {len(nowcast)} ({nowcast.index[0].date()} -> {nowcast.index[-1].date()}), r = {nowcast['r'].iloc[-1]:.3f}")
    ic = spearmanr(revisions["surprise_before"], revisions["usda_revision"])[0]
    hit = (np.sign(revisions["surprise_before"]) == np.sign(revisions["usda_revision"]))[revisions["usda_revision"] != 0].mean()
    print(f"\n1. surprise before a WASDE vs USDA's export revision: rank corr {ic:+.3f}, direction right {hit:.0%}"
          f" of {int((revisions['usda_revision'] != 0).sum())} revisions ({len(revisions)} reports)")
    print("\n2. weekly IC of -(price vs fair value), Monday close after inspections:")
    for k in HORIZONS:
        lags = max(k // 5, 1)
        for name in ["signal_monthly", "signal_nowcast"]:
            block = merged[[name, f"fwd_{k}"]].dropna()
            t = nw_t_stat(block, [name], f"fwd_{k}", lags=lags).loc[name, "t_nw"]
            print(f"   {k:>2} days  {name:15s} IC {spearmanr(block[name], block[f'fwd_{k}'])[0]:+.3f}  t {t:+.2f}  ({len(block)} weeks)")
    errors = nowcast.dropna(subset=["next_week_actual"])
    print("\n3. next-week inspections forecast, mean absolute error (mbu):")
    for name in ["forecast_usda_pace", "forecast_last_week", "forecast_4wk"]:
        print(f"   {name:20s} {(errors[name] - errors['next_week_actual']).abs().mean():.2f}")
    print(f"   (average weekly inspections {errors['next_week_actual'].mean():.1f} mbu)")

    figures = {"export_pace": pace_chart(nowcast), "revision_test": revision_chart(revisions),
               "fair_value_weekly": fair_value_chart(merged)}
    run_dir = save_run(STRATEGY_DIR, "nowcast", cfg, {"weeks": len(nowcast)},
                       {"nowcast": nowcast, "revisions": revisions, "weekly_fair_value": merged}, figures)
    print(f"\nsaved to {run_dir}")
