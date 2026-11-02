"""
conditioners.py
---------------
Layer 1 conditioners from the fundamental side: does the corn fair-value signal (model A composite) work
differently depending on (1) USDA reports and (2) speculative positioning (COT)? Development data only.
Saved to runs/<time>_conditioners/.

    python -m strategies.wheat_corn_lag.conditioners

Signal: model A composite = -(own_ze_106 + own_ze_142 + own_ze_213) / 3   (+ = wheat cheap vs corn, expect UP)

1. USDA reports (WASDE, Grain Stocks, Prospective Plantings, Acreage; data/raw/usda_release_times_utc.csv)
   born over a report   a release fell inside the last 142 bars: the divergence may be news, not flow.
                        Hypothesis: news-born divergences revert LESS (they are "justified").
   report ahead         a release falls inside the forward window (known in advance from the calendar).
2. COT, legacy report (data/raw/ZW_cot_*.parquet), speculators = non-commercials
   spec_net = (non-commercial long - short) / open interest
   spec_z   = spec_net vs its past 156 weekly reports (at least 52), so z exists from early 2021
   point in time: positions are as of Tuesday and released Friday 15:30 New York; a value is usable only on
   bars ending after its release. Release = release_date (else LSE's capture time). Oct-Nov 2025 US
   government shutdown: CFTC published late, so positions dated 30 Sep - 31 Dec 2025 are treated as unknown
   until 1 Jan 2026 (conservative: the last real report, 23 Sep 2025, is held).
   crowding buckets     fade agrees with an unwind of a crowded spec position (|spec_z| > 1 and the signal
                        trades against the specs' side) / fade adds to the crowd / specs not extreme.

A conditioner counts as useful (fixed before the run): |t| >= 2 on the interaction signal x bucket
(Newey-West, lags = horizon) AND the IC gap between the buckets has the same sign in >= 4 of 6 years.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from core_quant.config import ROOT, save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features, change_ahead
from strategies.wheat_corn_lag.layer1 import LOOKBACKS, PRIMARY
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

USDA_FILE = "usda_release_times_utc.csv"
COT_FILE = "ZW_cot_2020-01-06_2026-10-06.parquet"
REPORT_WINDOW = 142
COT_LOOKBACK_WEEKS, COT_MIN_WEEKS, CROWDED = 156, 52, 1.0
SHUTDOWN_DATED = ("2025-09-30", "2025-12-31")
SHUTDOWN_KNOWN_FROM = "2026-01-01"
USEFUL_T, USEFUL_YEARS = 2.0, 4


# ------------------------------------------------------------------ point-in-time inputs
def bars_since_last_release(bars: pd.DataFrame) -> pd.Series:
    """Bars since the bar that contains the latest USDA release (0 = this bar contains one); inf before the first."""
    releases = pd.to_datetime(pd.read_csv(ROOT / "data" / "raw" / USDA_FILE)["release_utc"], utc=True)
    # a release at 16:00 is inside the bar (16:00, 16:15], which is labelled 16:15: first bar END after it
    release_bars = np.unique(bars.index.searchsorted(releases, side="right"))
    release_bars = release_bars[release_bars < len(bars)]
    position = np.arange(len(bars))
    latest = np.searchsorted(release_bars, position, side="right") - 1
    since = np.where(latest >= 0, position - release_bars[np.maximum(latest, 0)], np.inf)
    return pd.Series(since, index=bars.index)


def bars_to_next_release(bars: pd.DataFrame) -> pd.Series:
    """Bars until the next bar containing a release (known in advance from the published calendar)."""
    releases = pd.to_datetime(pd.read_csv(ROOT / "data" / "raw" / USDA_FILE)["release_utc"], utc=True)
    release_bars = np.unique(bars.index.searchsorted(releases, side="right"))
    release_bars = release_bars[release_bars < len(bars)]
    position = np.arange(len(bars))
    following = np.searchsorted(release_bars, position, side="right")
    return pd.Series(np.where(following < len(release_bars),
                              release_bars[np.minimum(following, len(release_bars) - 1)] - position, np.inf),
                     index=bars.index)


def cot_positioning(bars: pd.DataFrame) -> pd.DataFrame:
    """spec_net, spec_z and the release time of the report in use, for every bar (backward as-of join)."""
    cot = pd.read_parquet(ROOT / "data" / "raw" / COT_FILE).sort_values("date")
    position_date = pd.to_datetime(cot["date"])
    release_day = pd.to_datetime(cot["release_date"])
    friday_close = (release_day.fillna(position_date + pd.Timedelta(days=3)) + pd.Timedelta(hours=15, minutes=30))
    release = friday_close.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    captured = pd.concat([release, cot["created_at"].dt.tz_convert("UTC").astype(release.dtype)], axis=1).max(axis=1)
    release = release.where(cot["release_date"].notna(), captured)          # no release date (2026): capture time
    shutdown = (position_date >= SHUTDOWN_DATED[0]) & (position_date <= SHUTDOWN_DATED[1])
    release = release.where(~shutdown, pd.Timestamp(SHUTDOWN_KNOWN_FROM, tz="UTC"))

    weekly = pd.DataFrame({"release": release.astype(bars.index.dtype),
                           "spec_net": (cot["noncomm_long"] - cot["noncomm_short"]) / cot["open_interest"]})
    weekly = weekly.reset_index(drop=True)
    weekly = weekly.sort_values("release")
    # z vs the report's own past (each report is known at its release, so the z is too)
    history = weekly["spec_net"].rolling(COT_LOOKBACK_WEEKS, min_periods=COT_MIN_WEEKS)
    weekly["spec_z"] = (weekly["spec_net"] - history.mean()) / history.std()
    joined = pd.merge_asof(pd.DataFrame({"bar_end": bars.index}), weekly, left_on="bar_end", right_on="release",
                           direction="backward")
    joined.index = bars.index
    # self-check: no bar uses a report released after it
    assert (joined["release"].dropna() <= joined.loc[joined["release"].notna(), "bar_end"]).all()
    return joined[["release", "spec_net", "spec_z"]]


# ------------------------------------------------------------------ tests
def split_test(signal, forward, buckets: pd.Series, base: str, horizon: int) -> pd.DataFrame:
    """IC per bucket, and the Newey-West t of signal x bucket against the base bucket."""
    data = pd.DataFrame({"y": forward, "s": signal, "bucket": buckets}).dropna()
    rows = []
    for name, block in data.groupby("bucket"):
        rows.append({"bucket": name, "bars": len(block), "ic": spearmanr(block["s"], block["y"])[0]})
    table = pd.DataFrame(rows).set_index("bucket")
    design = pd.DataFrame({"s": data["s"]})
    for name in table.index.drop(base):
        flag = (data["bucket"] == name).astype(float)
        design[f"is_{name}"] = flag
        design[f"s_x_{name}"] = data["s"] * flag
    fit = sm.OLS(data["y"] / data["y"].std(), sm.add_constant(design)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    table["interaction_t_vs_base"] = [np.nan if name == base else fit.tvalues[f"s_x_{name}"] for name in table.index]
    return table


def yearly_gap(signal, forward, buckets: pd.Series, first: str, second: str, min_count: int = 200) -> pd.Series:
    """min_count: observations needed per bucket per year (200 suits 15-minute bars; daily data needs fewer)."""
    data = pd.DataFrame({"y": forward, "s": signal, "bucket": buckets}).dropna()
    gaps = {}
    for year, block in data.groupby(data.index.year):
        one, two = block[block["bucket"] == first], block[block["bucket"] == second]
        if len(one) > min_count and len(two) > min_count:
            gaps[year] = spearmanr(one["s"], one["y"])[0] - spearmanr(two["s"], two["y"])[0]
    return pd.Series(gaps, name=f"IC({first}) - IC({second})")


# ------------------------------------------------------------------ charts
def split_chart(tables: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.2), sharey=True)
    for axis, (title, table) in zip(axes, tables.items()):
        axis.bar(range(len(table)), table["ic"], color="#2a78d6")
        for i, (name, row) in enumerate(table.iterrows()):
            note = "" if np.isnan(row["interaction_t_vs_base"]) else f"\nt {row['interaction_t_vs_base']:+.1f}"
            axis.annotate(f"{int(row['bars']):,} bars{note}", (i, row["ic"]), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8)
        axis.set_xticks(range(len(table)), table.index, fontsize=8)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel(f"rank IC of the corn signal, {PRIMARY // 71}-session forward")
    figure.tight_layout()
    return figure


def cot_chart(bars, cot: pd.DataFrame) -> plt.Figure:
    daily = cot.groupby(bars.index.normalize()).last()
    price = bars["W"].groupby(bars.index.normalize()).last()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, height_ratios=[1, 1.2])
    top.plot(price.index, price.values, lw=1)
    top.set_ylabel("wheat, c/bu")
    top.set_title("Speculative positioning as known at the time (steps change on release Fridays)", loc="left")
    bottom.plot(daily.index, daily["spec_z"], lw=1.2, label="spec_z (vs past 156 reports)")
    bottom.plot(daily.index, daily["spec_net"] * 10, lw=1, alpha=0.6, label="spec net % of OI (x10)")
    for level in (CROWDED, -CROWDED):
        bottom.axhline(level, color="grey", ls=":", lw=0.9)
    bottom.axvspan(pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp(SHUTDOWN_KNOWN_FROM, tz="UTC"), color="grey",
                   alpha=0.15, label="shutdown: last real report held")
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    signal = -features[[f"own_ze_{l}" for l in LOOKBACKS]].mean(axis=1, skipna=False)
    forward = change_ahead(bars["log_w"], bars["segment"], PRIMARY)

    since = bars_since_last_release(bars)
    ahead = bars_to_next_release(bars)
    born = pd.Series(np.where(since < REPORT_WINDOW, "born over a report", "no report in window"), index=bars.index)
    upcoming = pd.Series(np.where(ahead <= PRIMARY, "report ahead", "no report ahead"), index=bars.index)

    cot = cot_positioning(bars)
    crowded = cot["spec_z"].abs() > CROWDED
    against = np.sign(signal) == -np.sign(cot["spec_z"])          # the fade trades against the specs' side
    crowding = pd.Series(np.where(cot["spec_z"].isna(), None,
                                  np.where(~crowded, "specs not extreme",
                                           np.where(against, "fade vs crowded specs", "fade with crowded specs"))),
                         index=bars.index)

    tests = {
        "USDA: divergence born over a report?": (born, "no report in window", ("born over a report", "no report in window")),
        "USDA: report inside the forward window?": (upcoming, "no report ahead", ("report ahead", "no report ahead")),
        "COT: speculative crowding": (crowding, "specs not extreme", ("fade vs crowded specs", "fade with crowded specs")),
    }
    tables, gaps = {}, {}
    pd.set_option("display.width", 200)
    for title, (buckets, base, (first, second)) in tests.items():
        tables[title] = split_test(signal, forward, buckets, base, PRIMARY)
        gaps[title] = yearly_gap(signal, forward, buckets, first, second)
        print(f"\n== {title}")
        print(tables[title].round(3).to_string())
        gap = gaps[title]
        overall = tables[title].loc[first, "ic"] - tables[title].loc[second, "ic"]   # pooled gap: one odd year cannot flip it
        print(f"   {gap.name} by year: {gap.round(3).to_dict()}  -> same sign as the overall gap ({overall:+.3f})"
              f" in {int((np.sign(gap) == np.sign(overall)).sum())} of {len(gap)} years")
    print(f"\nCOT coverage: spec_z from {cot['spec_z'].first_valid_index()}; reports in use cover {cot['release'].notna().mean():.1%} of bars")

    figures = {"splits": split_chart(tables), "cot_positioning": cot_chart(bars, cot)}
    run_dir = save_run(STRATEGY_DIR, "conditioners", cfg, {"report_window": REPORT_WINDOW, "crowded_z": CROWDED},
                       {**{f"split_{i}": t for i, t in enumerate(tables.values())},
                        **{f"yearly_gap_{i}": g.to_frame() for i, g in enumerate(gaps.values())}}, figures)
    print(f"saved to {run_dir}")
