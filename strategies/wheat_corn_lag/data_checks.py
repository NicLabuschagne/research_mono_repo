"""
data_checks.py
--------------
Facts about the data (Steps 1 and 3). Development period only: the holdout is never loaded here.

    python -m strategies.wheat_corn_lag.data_checks

1. Coverage   rows per leg, the joined sample, bars per day, which UTC hours trade
2. Prices     price grid (real CBOT prices move in 0.25 c ticks)
3. Timestamps the look-ahead check. Build 15-minute bars from CBOT wheat ticks, labelled by bar END,
              and correlate their returns with the file's returns at lags -1, 0, +1.
              If the re-labelled bars are right, the peak is at lag 0.
              A peak at lag +1 would mean our labels are one bar too early (a leak).
4. Gaps       time between consecutive bars: 15 min normally, plus the daily break, the overnight close,
              weekends and holidays. Anything else is a missing bar.
5. Spikes     bad prints: a bar that jumps > 6 sd and fully reverses on the next bar.
              Uses the NEXT bar, so it is a diagnostic only and never feeds the strategy.
6. Corn rolls every corn contract switch (from Databento contract-level ZC bars) falls between sessions
              and on a segment edge.
7. Segments   how much of the series is stitched; each segment's first bar has no return; the largest
              intraday wheat moves (a mid-session wheat roll would show up as a big jump at a quiet hour).
8. Missed wheat rolls  overnight wheat moves corn does not explain, at the boundaries we stitch.
              A wheat roll outside the cut windows would be a large residual far from any FND.
9. Extra factors      soybeans and USD: coverage, the largest moves (a roll jump would show), and the
              look-ahead check: correlation with wheat's return must peak at lag 0, not at +-1 bar.
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.wheat_corn_lag.strategy import (corn_contract_switches, load, load_bars, load_leg,
                                                 wheat_first_notice_days)


def coverage(cfg, bars: pd.DataFrame) -> None:
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    for name, file_name in [("wheat", cfg.wheat_file), ("corn", cfg.corn_file)]:
        leg = load_leg(file_name, cfg.bar_minutes)
        leg = leg[leg.index < holdout]
        print(f"{name:6s} {len(leg):>7,} bars  {leg.index[0]} -> {leg.index[-1]}"
              f"   {len(bars) / len(leg):.1%} have a bar on the other leg")
    print(f"joined {len(bars):>7,} bars  {bars.index[0]} -> {bars.index[-1]}")

    per_day = bars.groupby(bars.index.normalize()).size()
    print(f"bars per day: median {per_day.median():.0f}, 5th pct {per_day.quantile(0.05):.0f}, max {per_day.max()}")
    hours = bars.groupby(bars.index.hour).size()
    print("bars by UTC hour of bar end:\n" + hours.to_string())


def price_grid(bars: pd.DataFrame) -> None:
    for leg in ["W", "C"]:
        step = bars[leg].diff().abs()
        on_tick = (np.abs(bars[leg] / 0.25 - np.round(bars[leg] / 0.25)) < 1e-6).mean()
        print(f"{leg}: smallest price change {step[step > 1e-9].min():.3f} c,"
              f" {on_tick:.1%} of closes on the 0.25 c CBOT tick grid")


def timestamp_check(cfg, bars: pd.DataFrame) -> None:
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    files = sorted((ROOT / "data" / "raw").glob("WHEAT_futures_tick_*.parquet"))
    ticks = pd.concat(pd.read_parquet(file) for file in files)
    trades = ticks[(ticks["kind"] == "trade") & (ticks["time"] < holdout)].set_index("time")["price"]
    # closed="left", label="right": the bar [14:00, 14:15) is labelled 14:15, the same convention as load_leg
    exchange = trades.resample(f"{cfg.bar_minutes}min", closed="left", label="right").last().dropna()
    print(f"exchange ticks before the holdout: {exchange.index[0]} -> {exchange.index[-1]}, {len(exchange):,} bars")

    file_ret = bars["log_w"].diff()
    exchange_ret = np.log(exchange).diff()
    for lag in [-1, 0, 1]:
        joined = pd.concat([file_ret, exchange_ret.shift(lag)], axis=1, join="inner").dropna()
        print(f"  corr(file return at t, exchange return at t{-lag:+d}): {joined.corr().iloc[0, 1]:.3f}  (n={len(joined):,})")

    level = pd.concat([bars["W"], exchange], axis=1, join="inner").dropna()
    print(f"  file minus exchange price: median {(level.iloc[:, 0] - level.iloc[:, 1]).median():.2f} c"
          " (a constant gap = different contract month / roll adjustment, not a timing problem)")


GAP_NAMES = {15: "normal", 30: "1 bar missing", 45: "2 bars missing", 60: "daily break 07:45-08:30 CT",
             345: "overnight close", 3225: "weekend", 4665: "long weekend (holiday)"}


def gaps(bars: pd.DataFrame) -> None:
    minutes = (bars.index.to_series().diff().dt.total_seconds() / 60).dropna().astype(int)
    counts = minutes.value_counts()
    for length, count in counts.items():
        if count >= 20:
            print(f"  {length:>5} min  {count:>7,}  {GAP_NAMES.get(length, '')}")
    print(f"  other     {counts[counts < 20].sum():>7,}  (holidays, early closes)")


def spikes(bars: pd.DataFrame) -> None:
    for leg in ["W", "C"]:
        move = bars[leg].diff()
        sd = move.rolling(71 * 20, min_periods=500).std().shift(1)     # past 20 days only
        reverses = ((move.abs() > 6 * sd) & (move.shift(-1).abs() > 6 * sd)
                    & ((move + move.shift(-1)).abs() < 0.25 * move.abs()))
        print(f"  {leg}: {int(reverses.sum())} spike-and-revert bars {list(bars.index[reverses])}")


def corn_rolls(cfg, bars: pd.DataFrame) -> None:
    switches = corn_contract_switches(cfg)
    switches = switches[switches["first_on_new"] < pd.Timestamp(cfg.split.holdout_start, tz="UTC")]
    inside = switches[switches["last_on_old"].dt.normalize() == switches["first_on_new"].dt.normalize()]
    print(f"  corn contract switches before the holdout: {len(switches)}, inside a session: {len(inside)}  (must be 0)")
    # every switch must land on a segment edge: the old and new contract never share a segment
    old_segment = bars["segment"].reindex(switches["last_on_old"], method="ffill").values
    new_segment = bars["segment"].reindex(switches["first_on_new"], method="ffill").values
    print(f"  switches with old and new contract in the same segment: {int((old_segment == new_segment).sum())}  (must be 0)")


def segments(cfg, bars: pd.DataFrame) -> None:
    span = bars.groupby("session").apply(lambda s: s.index[-1] - s.index[0])
    print(f"  sessions: {bars['session'].nunique():,}, longest {span.max()}  (must be < 24 h, so session = UTC date)")
    boundaries = bars["session"].nunique() - 1
    cuts = bars["segment"].nunique() - 1
    print(f"  session boundaries: {boundaries:,}, cut as possible rolls: {cuts:,} ({cuts / boundaries:.0%}),"
          f" stitched: {boundaries - cuts:,} ({1 - cuts / boundaries:.0%})")
    per_segment = bars.groupby("segment")["session"].nunique()
    print(f"  sessions per segment: median {per_segment.median():.0f}, max {per_segment.max()},"
          f" single-session segments (inside roll windows): {(per_segment == 1).sum()}")
    first = bars.groupby("segment").head(1)
    print(f"  first bar of a segment with a return: {int(first[['ret_w', 'ret_c']].notna().sum().sum())}  (must be 0)")

    # a mid-session wheat roll would be a big jump at a quiet hour
    intraday = bars["ret_w"].where(bars["session"] == bars["session"].shift())
    sd = intraday.rolling(71 * 20, min_periods=500).std().shift(1)
    top = (intraday / sd).abs().dropna().nlargest(10)
    print("  10 largest intraday wheat moves (sd, UTC bar end):")
    for time, value in top.items():
        print(f"    {time}  {value:5.1f}  {bars['W'].diff()[time]:+.2f} c")


def missed_wheat_rolls(cfg, bars: pd.DataFrame) -> None:
    """Overnight wheat moves that corn does not explain. A wheat roll outside the cut windows would be one."""
    opens = bars.index[bars["session"] != bars["session"].shift()][1:]   # the first session has no previous close
    overnight_w = bars["log_w"].diff().loc[opens]                 # raw: includes any roll
    overnight_c = bars["log_c"].diff().loc[opens]
    stitched = bars["segment"].diff().loc[opens].eq(0)

    slope = np.polyfit(overnight_c[stitched], overnight_w[stitched], 1)[0]
    residual = overnight_w - slope * overnight_c
    scale = residual[stitched].abs().median() / 0.6745              # robust sd: big outliers do not inflate it
    z = (residual / scale).abs()
    print(f"  overnight wheat-on-corn slope {slope:.2f}; robust sd of the residual {scale * 1e4:.0f} bp")
    print(f"  boundaries with |residual| > 5 sd: stitched {int((z[stitched] > 5).sum())} of {int(stitched.sum())},"
          f" cut {int((z[~stitched] > 5).sum())} of {int((~stitched).sum())}")

    fnds = wheat_first_notice_days(bars.index[0].year, bars.index[-1].year + 1)
    print("  10 largest residuals at STITCHED boundaries (a missed roll would show here):")
    for time in z[stitched].nlargest(10).index:
        days = min(abs((fnds - time.normalize()).days))
        print(f"    {time.date()}  {z[time]:5.1f} sd  wheat {np.expm1(overnight_w[time]) * 100:+.1f}%"
              f"  corn {np.expm1(overnight_c[time]) * 100:+.1f}%  {days} days from a wheat FND")


def extra_factors(bars: pd.DataFrame) -> None:
    for column, name in [("ret_s", "soybeans"), ("ret_u", "USD (-EURUSD)")]:
        returns = bars[column]
        lags = {lag: bars["ret_w"].corr(returns.shift(lag)) for lag in [-1, 0, 1]}
        print(f"  {name}: coverage {returns.notna().mean():.1%} from {returns.first_valid_index()},"
              f" corr with wheat at lag -1/0/+1: {lags[-1]:+.3f} / {lags[0]:+.3f} / {lags[1]:+.3f}")
        print(f"    5 largest moves (bp): {[(str(t)[:16], round(v * 1e4)) for t, v in returns.abs().nlargest(5).items()]}")


if __name__ == "__main__":
    cfg = load()
    bars = load_bars(cfg)                     # include_holdout=False
    print("== 1. coverage");   coverage(cfg, bars)
    print("\n== 2. prices");   price_grid(bars)
    print("\n== 3. timestamps"); timestamp_check(cfg, bars)
    print("\n== 4. gaps");     gaps(bars)
    print("\n== 5. spikes");   spikes(bars)
    print("\n== 6. corn rolls"); corn_rolls(cfg, bars)
    print("\n== 7. segments"); segments(cfg, bars)
    print("\n== 8. missed wheat rolls"); missed_wheat_rolls(cfg, bars)
    print("\n== 9. extra factors"); extra_factors(bars)
