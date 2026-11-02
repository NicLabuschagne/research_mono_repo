"""
layer1.py
---------
Layer 1: is our fair value of wheat, and the divergence from it, predictive? Statistics only: no entries,
thresholds, exits or costs (that is Layer 2). Development data only. Saved to runs/<time>_layer1/.

    python -m strategies.wheat_corn_lag.layer1

Signals (all "fade": positive = wheat is cheap vs its fair value, so expect wheat UP; all known at t)
  A  move fair value, rolling beta      -own_ze_L     L = 106, 142, 213 bars, and their average (composite)
  B  move fair value, Kalman beta       -own_kze_L    same lookbacks and composite
  C  level fair value                   -level_z_W    regression window W = 10, 20, 40 sessions
  K  Kalman level fair value (bean      -kz_Ms        live gap / its 20-session sd (= A's scaling); memory M = 5, 10, 20
     convention, designed for wheat)                  sessions (plateau test across M); 1-8 h horizons also shown
  R  the same gap / its 20-session RMS  -kzr_Ms       the BASE (adopted 2026-11-02); scale comparison: the same gap
     raw (log, cents), / wheat's 1-session move, / the filter's own prediction sd
  control: wheat alone, no corn         -wheat_ze_L   if this does as well, corn adds nothing

Measured against forward wheat returns at 1, 2, 3 and 4 sessions (71, 142, 213, 284 bars)
  rank IC with a Newey-West t (lags = horizon: overlapping windows are not independent)
  IC by calendar year, IC decay over horizons, deciles (mean forward move per signal decile)
  corn interaction: does the signal work better when corn moved >1 sd the same way wheat overshot?

Pass criteria, fixed BEFORE the first run, judged at the primary horizon of 142 bars (2 sessions):
  Newey-West t >= 3;  same IC sign in >= 5 of 6 years;  deciles monotonic (rank corr of decile vs mean
  forward move >= 0.8);  for A, B and the control: all three lookbacks the same sign with t >= 2 (plateau).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from core_quant.config import save_run
from core_quant.theme import set_theme
from core_quant.validation import ic_rank_spearman, nw_t_stat
from strategies.wheat_corn_lag.features import build_features, change_ahead
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

LOOKBACKS = [106, 142, 213]
HORIZONS = [4, 16, 32, 71, 142, 213, 284]      # 1 h, 4 h, 8 h, then 1-4 sessions (shorter ones added for model K)
PRIMARY = 142
PASS_T, PASS_YEARS, PASS_MONOTONIC, PLATEAU_T = 3.0, 5, 0.8, 2.0
FAMILIES = {"A move, rolling beta": "own_ze", "B move, Kalman beta": "own_kze", "control: wheat alone": "wheat_ze"}
KALMAN = "K Kalman level"                       # plateau test across its three speeds, like A / B across lookbacks
KALMAN_RMS = "R Kalman level, RMS scale"        # same gap, z = gap / RMS around zero (features.py kzr)
KALMAN_SCALES = {KALMAN: "kz", KALMAN_RMS: "kzr", "G raw gap, log": "kgap", "G raw gap, cents": "kgapc",
                 "V gap / wheat 1-session move": "kzv", "I Kalman prediction z, fixed R": "kzk"}   # same gap, sized 6 ways
BASE, SECONDARY = f"{KALMAN_RMS} | M20s", "A move, rolling beta | composite"   # adopted 2026-11-02 (RMS scale)
LONG_HORIZONS = [16, 32, 71, 142, 213, 284, 426, 568, 710, 1065, 1420]     # 4 hours .. 20 sessions
SPEEDS = [("R_M5", f"{KALMAN_RMS} | M5s"), ("R_M10", f"{KALMAN_RMS} | M10s"), ("base", BASE),
          ("K_M20_sd", f"{KALMAN} | M20s"), ("secondary_A", SECONDARY)]
LEG_HORIZONS = [32, 71, 142, 284, 568]                                     # 8 hours .. 8 sessions


def build_signals(features: pd.DataFrame, level_windows: list[int], kalman_memories: list[int]) -> pd.DataFrame:
    signals = {}
    for family, prefix in FAMILIES.items():
        for lookback in LOOKBACKS:
            signals[f"{family} | L{lookback}"] = -features[f"{prefix}_{lookback}"]
        signals[f"{family} | composite"] = -features[[f"{prefix}_{l}" for l in LOOKBACKS]].mean(axis=1, skipna=False)
    for window in level_windows:
        signals[f"C level fair value | W{window // 71}s"] = -features[f"level_z_{window}"]
    for memory in kalman_memories:
        for family, prefix in KALMAN_SCALES.items():
            signals[f"{family} | M{memory}s"] = -features[f"{prefix}_{memory}s"]
    return pd.DataFrame(signals)


def ic_table(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        target = f"fwd_{horizon}"
        ic = ic_rank_spearman(frame, names, target)
        t = nw_t_stat(frame, names, target, lags=horizon)
        for name in names:
            rows.append({"signal": name, "h": horizon, "ic": float(ic.loc[name, "ic"]),
                         "t_nw": float(t.loc[name, "t_nw"]), "n": int(ic.loc[name, "n"])})
    return pd.DataFrame(rows)


def ic_by_year(frame: pd.DataFrame, names: list[str], horizon: int) -> pd.DataFrame:
    rows = {}
    for year, block in frame.groupby(frame.index.year):
        rows[year] = {name: spearmanr(*block[[name, f"fwd_{horizon}"]].dropna().T.values)[0] for name in names}
    return pd.DataFrame(rows).T


def deciles(frame: pd.DataFrame, name: str, horizon: int) -> pd.Series:
    pair = frame[[name, f"fwd_{horizon}"]].dropna()
    bucket = pd.qcut(pair[name], 10, labels=False) + 1
    return pair[f"fwd_{horizon}"].groupby(bucket).mean() * 1e4


def corn_interaction(frame: pd.DataFrame, features: pd.DataFrame, lookback: int, horizon: int) -> dict:
    """IC by corn context, and the t of signal x (corn moved >1 sd the way wheat overshot) in a regression."""
    own, corn = features[f"own_ze_{lookback}"], features[f"corn_ze_{lookback}"]
    signal, forward = -own, frame[f"fwd_{horizon}"]
    with_corn = ((np.sign(corn) == np.sign(own)) & (corn.abs() > 1)).astype(float)
    contexts = {"corn moved >1 sd the same way (overshoot)": with_corn == 1,
                "corn flat (< 0.5 sd)": corn.abs() < 0.5,
                "corn moved >1 sd the other way (lag)": (np.sign(corn) != np.sign(own)) & (corn.abs() > 1)}
    out = {}
    for label, mask in contexts.items():
        pair = pd.concat([signal[mask], forward[mask]], axis=1).dropna()
        out[label] = {"ic": spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0], "bars": len(pair)}
    data = pd.DataFrame({"y": forward, "s": signal, "with_corn": with_corn, "s_x_with_corn": signal * with_corn}).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data[["s", "with_corn", "s_x_with_corn"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon})
    out["interaction t (signal x overshoot)"] = {"ic": float(fit.tvalues["s_x_with_corn"]), "bars": len(data)}
    return out


def incremental_value(frame: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Does corn add anything beyond wheat's own price move? Two tests per lookback:
      1. y = a x corn_ze + b x own_ze: if a ~ b, wheat's TOTAL move reverts and the split into corn's part and
         wheat's own part adds nothing; if |b| > |a|, the part corn does not explain reverts more (relative value).
      2. the corn context IC for the CONTROL signal: if wheat-alone also jumps in the overshoot context, the
         jump is "big moves revert more", not corn."""
    rows = []
    forward = frame[f"fwd_{horizon}"]
    for lookback in LOOKBACKS:
        own, corn, wheat = features[f"own_ze_{lookback}"], features[f"corn_ze_{lookback}"], features[f"wheat_ze_{lookback}"]
        data = pd.DataFrame({"y": forward / forward.std(), "corn": corn, "own": own}).dropna()
        fit = sm.OLS(data["y"], sm.add_constant(data[["corn", "own"]])).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
        overshoot = (np.sign(corn) == np.sign(own)) & (corn.abs() > 1)
        flat = corn.abs() < 0.5
        pair_over = pd.concat([-wheat[overshoot], forward[overshoot]], axis=1).dropna()
        pair_flat = pd.concat([-wheat[flat], forward[flat]], axis=1).dropna()
        rows.append({"L": lookback, "a_corn": fit.params["corn"], "t_a": fit.tvalues["corn"],
                     "b_own": fit.params["own"], "t_b": fit.tvalues["own"],
                     "control_ic_overshoot": spearmanr(pair_over.iloc[:, 0], pair_over.iloc[:, 1])[0],
                     "control_ic_corn_flat": spearmanr(pair_flat.iloc[:, 0], pair_flat.iloc[:, 1])[0]})
    return pd.DataFrame(rows)


def verdicts(table: pd.DataFrame, years: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    primary = table[table["h"] == PRIMARY].set_index("signal")
    rows = []
    for name in primary.index:
        ic, t = primary.loc[name, "ic"], primary.loc[name, "t_nw"]
        same_sign_years = int((np.sign(years[name]) == np.sign(ic)).sum())
        means = deciles(frame, name, PRIMARY)
        monotonic = spearmanr(means.index, means.values)[0]
        family, variant = name.split(" | ")
        plateau = None
        if family in FAMILIES or family in KALMAN_SCALES:
            names = [f"{family} | L{l}" for l in LOOKBACKS] if family in FAMILIES else \
                [n for n in primary.index if n.startswith(f"{family} | ")]
            members = primary.loc[names]
            plateau = bool((np.sign(members["ic"]) == np.sign(ic)).all() and (members["t_nw"].abs() >= PLATEAU_T).all())
        checks = [abs(t) >= PASS_T and t > 0, same_sign_years >= PASS_YEARS, monotonic >= PASS_MONOTONIC]
        if plateau is not None:
            checks.append(plateau)
        rows.append({"signal": name, "ic": ic, "t_nw": t, "years_same_sign": same_sign_years,
                     "decile_monotonic": monotonic, "plateau": plateau, "PASS": all(checks)})
    return pd.DataFrame(rows)


def long_decay(bars, features, signals: pd.DataFrame, common: pd.Series) -> pd.DataFrame:
    """Where the IC peaks and fades: base and secondary out to 20 sessions on the SAME bars, plus how persistent the
    Kalman gap itself is (its correlation with itself h bars later). A forward window may not cross a possible roll,
    so the longest horizons keep fewer bars (shown)."""
    gap = features["kgap_20s"]
    rows = []
    for h in LONG_HORIZONS:
        forward = change_ahead(bars["log_w"], bars["segment"], h)
        row = {"sessions": h / 71}
        for label, name in SPEEDS:                     # faster Kalman memories next to the base: what speed costs
            pair = pd.DataFrame({"s": signals[name], "y": forward})[common].dropna()
            row[f"ic_{label}"] = spearmanr(pair["s"], pair["y"])[0]
            row[f"t_{label}"] = float(nw_t_stat(pair, ["s"], "y", lags=h).loc["s", "t_nw"])
        row["bars"] = len(pair)
        later = pd.DataFrame({"now": gap, "later": gap.groupby(bars["segment"]).shift(-h)})[common].dropna()
        row["gap_self_corr"] = later["now"].corr(later["later"])
        rows.append(row)
    return pd.DataFrame(rows)


def which_leg(bars, features, signals: pd.DataFrame, common: pd.Series) -> pd.DataFrame:
    """The gap is wheat - (alpha + beta corn): does it close through wheat, corn, or the spread? Same signal (base),
    three forward returns. Corn's IC is NEGATIVE if corn moves to close the gap. Edge = mean move in the top signal
    decile minus the bottom decile (bp), next to the sd of that forward move (its risk)."""
    signal, beta = signals[BASE], features["kbeta_20s"]
    rows = []
    for h in LEG_HORIZONS:
        wheat = change_ahead(bars["log_w"], bars["segment"], h)
        corn = change_ahead(bars["log_c"], bars["segment"], h)
        for label, forward in [("wheat outright", wheat), ("corn outright", corn), ("spread wheat - beta x corn", wheat - beta * corn)]:
            pair = pd.DataFrame({"s": signal, "y": forward})[common].dropna()
            decile = pd.qcut(pair["s"], 10, labels=False)
            edge = (pair["y"][decile == 9].mean() - pair["y"][decile == 0].mean()) * 1e4
            rows.append({"sessions": h / 71, "instrument": label, "ic": spearmanr(pair["s"], pair["y"])[0],
                         "t": float(nw_t_stat(pair, ["s"], "y", lags=h).loc["s", "t_nw"]),
                         "top_minus_bottom_decile_bp": edge, "sd_of_move_bp": pair["y"].std() * 1e4,
                         "edge_per_sd": edge / (pair["y"].std() * 1e4)})
    return pd.DataFrame(rows)


def scale_comparison(frame: pd.DataFrame, common: pd.Series, memory: int = 20) -> pd.DataFrame:
    """The same Kalman gap sized six ways, on the tradable bars, at 1 and 2 sessions."""
    names = [f"{family} | M{memory}s" for family in KALMAN_SCALES]
    sample = frame[common]
    table = ic_table(sample, names)
    rows = []
    for horizon in (71, 142):
        years = ic_by_year(sample, names, horizon)
        for name in names:
            line = table[(table["signal"] == name) & (table["h"] == horizon)].iloc[0]
            means = deciles(sample, name, horizon)
            rows.append({"scale": name.split(" | ")[0], "sessions": horizon / 71, "ic": line["ic"], "t_nw": line["t_nw"],
                         "years_same_sign": int((np.sign(years[name]) == np.sign(line["ic"])).sum()),
                         "decile_monotonic": spearmanr(means.index, means.values)[0],
                         "top_minus_bottom_bp": means.iloc[-1] - means.iloc[0]})
    return pd.DataFrame(rows)


def scale_choice(comparison: pd.DataFrame) -> str:
    """Rule fixed before the run: keep the RMS base unless another scale beats its 1-session t by more than 0.5
    while passing years (>= 5) and deciles (>= 0.8) at 1 session."""
    one = comparison[comparison["sessions"] == 1].set_index("scale")
    base_t = one.loc[KALMAN_RMS, "t_nw"]
    better = one[(one["t_nw"] > base_t + 0.5) & (one["years_same_sign"] >= 5) & (one["decile_monotonic"] >= 0.8)]
    return KALMAN_RMS if better.empty else better["t_nw"].idxmax()


# ------------------------------------------------------------------ charts
def long_decay_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for label, name in SPEEDS:
        axis.plot(table["sessions"], table[f"ic_{label}"], marker="o", lw=2.5 if label == "base" else 1.2, label=name)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xscale("log")
    axis.set_xticks([0.25, 0.5, 1, 2, 4, 8, 20], ["4h", "8h", "1", "2", "4", "8", "20"])
    axis.minorticks_off()
    axis.set_xlabel("forward horizon, sessions")
    axis.set_ylabel("rank IC with wheat's forward return")
    axis.legend(fontsize=8, loc="upper left")
    twin = axis.twinx()
    twin.plot(table["sessions"], table["gap_self_corr"], ls=":", color="grey", label="Kalman gap's correlation with itself")
    twin.set_ylabel("gap self-correlation (dotted)")
    twin.legend(fontsize=8, loc="upper right")
    axis.set_title("IC decay out to 20 sessions (same bars), and how persistent the gap is", loc="left")
    figure.tight_layout()
    return figure


def leg_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, column, title in [(axes[0], "ic", "rank IC of the base signal"),
                                (axes[1], "edge_per_sd", "top-minus-bottom decile move / sd of the move")]:
        pivot = table.pivot(index="sessions", columns="instrument", values=column)
        pivot.plot.bar(ax=axis, rot=0)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_xlabel("horizon, sessions")
        axis.set_title(title, loc="left")
        axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def decay_chart(table: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 5))
    for name in names:
        block = table[table["signal"] == name]
        error = 2 * block["ic"] / block["t_nw"]
        axis.plot(block["h"] / 71, block["ic"], marker="o", label=name)
        axis.fill_between(block["h"] / 71, block["ic"] - error.abs(), block["ic"] + error.abs(), alpha=0.12)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xlabel("forward horizon, sessions")
    axis.set_ylabel("rank IC (band = +-2 Newey-West se)")
    axis.set_title("IC decay: does the divergence predict wheat, and over how long?", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def year_chart(years: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        axis.bar(np.arange(len(years)) + (i - (len(names) - 1) / 2) * width, years[name], width=width * 0.95, label=name)
    axis.set_xticks(np.arange(len(years)), years.index)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_ylabel(f"rank IC, {PRIMARY // 71}-session forward return")
    axis.set_title("IC by calendar year (stability)", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def decile_chart(frame: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    for axis, name in zip(axes, names):
        means = deciles(frame, name, PRIMARY)
        axis.bar(means.index, means.values, color=["#d62728" if v < 0 else "#2ca02c" for v in means.values])
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(name, loc="left", fontsize=9)
        axis.set_xlabel("signal decile (10 = wheat cheapest vs fair value)")
    axes[0].set_ylabel(f"mean wheat move next {PRIMARY // 71} sessions, bp")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    signals = build_signals(features, cfg.features.level_windows_bars, cfg.features.kalman_memory_sessions)
    targets = pd.DataFrame({f"fwd_{h}": change_ahead(bars["log_w"], bars["segment"], h) for h in HORIZONS})
    frame = signals.join(targets)
    names = list(signals.columns)

    table = ic_table(frame, names)
    years = ic_by_year(frame, names, PRIMARY)
    verdict = verdicts(table, years, frame)
    corn = corn_interaction(frame, features, 142, PRIMARY)
    increment = incremental_value(frame, features, PRIMARY)

    pd.set_option("display.width", 220)
    print("== rank IC by horizon (bars; 71 = 1 session)")
    print(table.pivot(index="signal", columns="h", values="ic").round(3).to_string())
    print("\n== Newey-West t")
    print(table.pivot(index="signal", columns="h", values="t_nw").round(2).to_string())
    print(f"\n== verdict at the primary horizon ({PRIMARY} bars)")
    print(verdict.round(3).to_string(index=False))
    print(f"\n== corn interaction (model A, L142, h{PRIMARY})")
    for label, values in corn.items():
        print(f"  {label:48s} {values['ic']:+.3f}   ({values['bars']:,} bars)")
    print(f"\n== does corn add beyond wheat's own move? (h{PRIMARY}; y in sd; negative coefficient = reverts)")
    print(increment.round(3).to_string(index=False))
    common = signals[[name for _, name in SPEEDS]].notna().all(axis=1)   # like for like: bars where all exist
    scales = scale_comparison(frame, common)
    print("\n== the same Kalman gap sized six ways (tradable bars; M20)")
    print(scales.round(3).to_string(index=False))
    print(f"scale choice (rule fixed before the run): {scale_choice(scales)}")
    decay = long_decay(bars, features, signals, common)
    legs = which_leg(bars, features, signals, common)
    print(f"\n== IC decay out to 20 sessions, base vs secondary on the same bars, and the gap's persistence")
    print(decay.round(3).to_string(index=False))
    print(f"\n== which leg closes the gap (base signal; corn IC NEGATIVE = corn moves to close it)")
    print(legs.round(3).to_string(index=False))
    kalman_names = [n for n in names if n.startswith(KALMAN)]
    print("\n== correlation of the Kalman signals with model A composite and with each other")
    print(frame[kalman_names + ["A move, rolling beta | composite"]].corr().round(2).to_string())

    shown = ["A move, rolling beta | composite", f"{KALMAN} | M5s", f"{KALMAN} | M10s", f"{KALMAN} | M20s", f"{KALMAN_RMS} | M20s",
             "control: wheat alone | composite"]
    figures = {"ic_decay": decay_chart(table, shown), "ic_by_year": year_chart(years, shown),
               "deciles": decile_chart(frame, shown), "long_decay": long_decay_chart(decay), "which_leg": leg_chart(legs)}
    run_dir = save_run(STRATEGY_DIR, "layer1", cfg, {"primary_horizon": PRIMARY, "base": BASE, "secondary": SECONDARY},
                       {"ic": table, "ic_by_year": years, "verdict": verdict, "long_decay": decay, "which_leg": legs, "scales": scales,
                        "corn_interaction": pd.DataFrame(corn).T, "incremental_value": increment}, figures)
    print(f"\nsaved to {run_dir}")
