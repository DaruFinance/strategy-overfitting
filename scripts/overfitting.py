"""Empirical variance decomposition of IS->OOS Sharpe degradation.

For every (strategy, window) the production framework already runs five
robustness perturbations: raw, ENT (entry drift), FEE (fee shock),
SLI (slippage shock), and ENT+IND (entry drift + indicator variance).
These perturbations are exactly parameter-sensitivity probes — exploit them.

Decomposition (model-free, per-strategy):

    V_param(s, w) = Var_test{ Sharpe_OOS(s, w, test) over {raw, ENT, FEE, SLI, ENT+IND} }
    V_window(s)   = Var_w{ Sharpe_OOS_raw(s, w) }
    D(s, w)       = Sharpe_IS_raw(s, w) - Sharpe_OOS_raw(s, w)

Across the corpus:

    Var(Sharpe_OOS) = E[V_param] + E[V_window | mean_per_strategy] + E[V_strategy] + ε

The headline question is predictive: does in-sample V_param forecast live-proxy
out-of-sample profitability? The "live-proxy" is the last two WFO windows of each
strategy, which were never used for optimisation.

Usage:
    python scripts/overfitting.py                 # synthetic demo (default)
    python scripts/overfitting.py --from-data \\
        --parquet-root /mnt/d/strategies_parquet/strategies
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds

OUT_FIG = Path(__file__).resolve().parent.parent / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_FIG.parent / "decomposition.json"

TEST_TAGS = ["raw", "ENT", "FEE", "SLI", "ENT+IND"]


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_metrics(parquet_root: str, assets: list[str] | None = None,
                 min_trades: int = 20, sharpe_clip: float = 5.0) -> pd.DataFrame:
    """Load per-window metrics from the strategies/ Parquet store.

    Applies basic data hygiene:
    - drops rows with n_trades < min_trades (small-N Sharpe is unreliable)
    - winsorises sharpe to [-sharpe_clip, +sharpe_clip] to bound the impact of
      a handful of strategies with one trade and a large PnL.
    """
    d = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    flt = None
    if assets:
        flt = ds.field("asset").isin(assets)
    cols = ["asset", "family", "strategy_name", "primary", "transform", "confluence",
            "sl", "window_idx", "sample", "test_tag", "lb", "n_trades",
            "roi", "pf", "sharpe", "max_dd"]
    table = d.to_table(columns=cols, filter=flt)
    df = table.to_pandas()
    n_pre = len(df)
    df = df[df["n_trades"].fillna(0) >= min_trades].copy()
    df["sharpe"] = df["sharpe"].clip(lower=-sharpe_clip, upper=sharpe_clip)
    print(f"[load_metrics] kept {len(df):,}/{n_pre:,} rows after n_trades>={min_trades} "
          f"and sharpe winsorised to [-{sharpe_clip}, +{sharpe_clip}]")
    return df


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (asset, strategy_name, window_idx, sample) with one
    sharpe column per test_tag. Metadata columns (family, primary, ...) are
    rejoined after the pivot — pivot_table drops rows whose index has NaN.
    """
    sub = df[df["test_tag"].isin(TEST_TAGS)].copy()
    wide = sub.pivot_table(
        index=["asset", "strategy_name", "window_idx", "sample"],
        columns="test_tag",
        values=["sharpe", "n_trades"],
        aggfunc="first",
    )
    wide.columns = [f"{m}_{t}" for m, t in wide.columns]
    wide = wide.reset_index()
    # Re-attach metadata: one (asset, strategy_name) -> one (family, primary, ...)
    meta_cols = ["asset", "strategy_name", "family", "primary", "transform",
                 "confluence", "sl"]
    meta = (sub[meta_cols].drop_duplicates(subset=["asset", "strategy_name"])
            .set_index(["asset", "strategy_name"]))
    wide = wide.join(meta, on=["asset", "strategy_name"])
    return wide


# -------------------------------------------------------------------
# Decomposition
# -------------------------------------------------------------------

def per_strategy_decomp(wide: pd.DataFrame) -> pd.DataFrame:
    """For each (asset, strategy, window) compute V_param across test_tags.
    Returns one row per (asset, strategy, window, sample).
    """
    cols = [f"sharpe_{t}" for t in TEST_TAGS]
    tr_cols = [f"n_trades_{t}" for t in TEST_TAGS]
    arr = wide[cols].to_numpy(dtype=np.float64)
    v_param = np.nanvar(arr, axis=1, ddof=0)
    mean_sharpe = np.nanmean(arr, axis=1)
    out = wide[["asset", "family", "strategy_name", "primary", "transform",
                "confluence", "sl", "window_idx", "sample"]].copy()
    out["sharpe_raw"] = wide["sharpe_raw"]
    out["mean_sharpe"] = mean_sharpe
    out["v_param"] = v_param
    out["n_trades_raw"] = wide["n_trades_raw"]
    # Bootstrap-style finite-sample variance proxy: var of a Sharpe estimate
    # is approximately (1 + 0.5 * sharpe^2) / n_trades.
    n = wide["n_trades_raw"].astype(np.float64).clip(lower=1)
    out["v_finite"] = (1.0 + 0.5 * np.nan_to_num(out["sharpe_raw"]) ** 2) / n
    return out


def degradation(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-(asset, strategy, window) IS->OOS Sharpe degradation on the raw signal."""
    is_ = wide[wide["sample"] == "IS"].set_index(
        ["asset", "family", "strategy_name", "window_idx"])
    oos = wide[wide["sample"] == "OOS"].set_index(
        ["asset", "family", "strategy_name", "window_idx"])
    common = is_.index.intersection(oos.index)
    is_, oos = is_.loc[common], oos.loc[common]
    df = pd.DataFrame(index=common)
    df["sharpe_is_raw"] = is_["sharpe_raw"]
    df["sharpe_oos_raw"] = oos["sharpe_raw"]
    df["degradation"] = is_["sharpe_raw"] - oos["sharpe_raw"]
    df["v_param_is"] = (is_[[f"sharpe_{t}" for t in TEST_TAGS]]
                        .var(axis=1, ddof=0))
    return df.reset_index()


def aggregate_variance(per_row: pd.DataFrame) -> dict:
    """Across the OOS rows, decompose total variance of sharpe_raw into
    components explained by parameter perturbation, by-strategy mean, and
    finite-sample noise. Returns shares (sum to 1)."""
    oos = per_row[per_row["sample"] == "OOS"].dropna(subset=["sharpe_raw"])
    total = float(np.var(oos["sharpe_raw"], ddof=0))
    e_param = float(oos["v_param"].mean())
    e_finite = float(oos["v_finite"].mean())
    # Strategy main effect: variance of strategy means around the global mean
    strat_mean = oos.groupby(["asset", "strategy_name"])["sharpe_raw"].mean()
    v_strategy = float(np.var(strat_mean, ddof=0))
    # Window main effect (proxy for regime): variance of window-level means
    win_mean = oos.groupby("window_idx")["sharpe_raw"].mean()
    v_window = float(np.var(win_mean, ddof=0))
    residual = max(total - (e_param + v_strategy + v_window + e_finite), 0.0)
    parts = {
        "total": total,
        "v_param_avg": e_param,
        "v_strategy_main": v_strategy,
        "v_window_main": v_window,
        "v_finite_avg": e_finite,
        "residual": residual,
    }
    parts["share_v_param"] = e_param / total if total else 0
    parts["share_v_strategy"] = v_strategy / total if total else 0
    parts["share_v_window"] = v_window / total if total else 0
    parts["share_v_finite"] = e_finite / total if total else 0
    parts["share_residual"] = residual / total if total else 0
    return parts


# -------------------------------------------------------------------
# Live-proxy: predictive validity of V_param
# -------------------------------------------------------------------

def live_proxy_outcomes(wide: pd.DataFrame, n_live_windows: int = 2) -> pd.DataFrame:
    """Pick last N windows per strategy as 'live proxy', vectorised.

    Returns one row per strategy with:
        v_param_hist: mean V_param across IS rows in history windows
        v_param_oos: mean V_param across the live OOS windows
        live_sharpe: mean OOS-raw Sharpe in the live windows
        live_profitable: True iff every live window had Sharpe > 0
    """
    df = wide.copy()
    cols = [f"sharpe_{t}" for t in TEST_TAGS]
    df["v_param"] = df[cols].var(axis=1, ddof=0)

    # Compute the cutoff window per (asset, strategy) so the last N windows are live.
    grp = df.groupby(["asset", "strategy_name"])["window_idx"]
    max_w = grp.transform("max")
    df["is_live"] = (max_w - df["window_idx"]) < n_live_windows

    is_rows = df[df["sample"] == "IS"]
    oos_rows = df[df["sample"] == "OOS"]
    hist_is = is_rows[~is_rows["is_live"]]
    live_oos = oos_rows[oos_rows["is_live"]]

    keys = ["asset", "strategy_name"]
    fam = (df[keys + ["family"]].drop_duplicates(keys).set_index(keys)["family"])
    v_hist = (hist_is.groupby(keys)["v_param"].mean()
              .rename("v_param_hist"))
    v_oos = (live_oos.groupby(keys)["v_param"].mean()
             .rename("v_param_oos"))
    live_sharpe = (live_oos.groupby(keys)["sharpe_raw"].mean()
                   .rename("live_sharpe"))
    n_live = (live_oos.groupby(keys)["window_idx"].nunique()
              .rename("n_live_w"))
    profitable = (live_oos.assign(_pos=live_oos["sharpe_raw"] > 0)
                  .groupby(keys)["_pos"].all().rename("live_profitable"))
    out = pd.concat([v_hist, v_oos, live_sharpe, n_live, profitable, fam],
                    axis=1, join="inner").reset_index()
    out = out[out["n_live_w"] >= n_live_windows].drop(columns="n_live_w")
    return out


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_decomposition_pie(parts: dict, out_path: Path):
    labels = ["V_param\n(robustness perts)",
              "V_strategy\n(strategy main)",
              "V_window\n(regime proxy)",
              "V_finite\n(finite-sample)",
              "Residual\n(interaction)"]
    sizes = [parts["share_v_param"], parts["share_v_strategy"],
             parts["share_v_window"], parts["share_v_finite"],
             parts["share_residual"]]
    sizes = [max(s, 0) for s in sizes]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#cc4c4c", "#4c8acc", "#4ccc7c", "#ccac4c", "#999999"]
    wedges, _, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=90, pctdistance=0.78, textprops={"fontsize": 10})
    ax.set_title(f"Decomposition of OOS Sharpe variance\n(total Var = {parts['total']:.3f})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_vs_live(live: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(0, np.nanpercentile(live["v_param_hist"], 99), 25)
    cuts = pd.cut(live["v_param_hist"], bins=bins, include_lowest=True)
    rate = live.groupby(cuts, observed=True)["live_profitable"].mean() * 100
    centers = [(b.left + b.right) / 2 for b in rate.index]
    ax.plot(centers, rate.values, "o-", color="#cc4c4c")
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("In-sample V_param (variance across robustness perturbations)")
    ax.set_ylabel("Live-proxy profitability rate (%)")
    ax.set_title("Higher in-sample V_param → lower live-proxy profitability")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bins = pd.qcut(live["v_param_hist"], q=10, duplicates="drop")
    grp = live.groupby(bins, observed=True)
    decile_rate = grp["live_profitable"].mean() * 100
    decile_count = grp.size()
    ax.bar(range(len(decile_rate)), decile_rate, color="#cc4c4c", alpha=0.85)
    ax.set_xticks(range(len(decile_rate)))
    ax.set_xticklabels([f"D{i+1}\nn={n}" for i, n in enumerate(decile_count)],
                       fontsize=8)
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("In-sample V_param decile (D1 = lowest sensitivity)")
    ax.set_ylabel("Live-proxy profitability rate (%)")
    ax.set_title("Predictive validity by V_param decile")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decomp_by_family(per_row: pd.DataFrame, out_path: Path):
    oos = per_row[per_row["sample"] == "OOS"].copy()
    fams = sorted(oos["family"].dropna().unique())
    rows = []
    for f in fams:
        sub = oos[oos["family"] == f]
        if sub.empty:
            continue
        total = float(np.var(sub["sharpe_raw"], ddof=0))
        e_param = float(sub["v_param"].mean())
        rows.append({"family": f, "total": total, "v_param_avg": e_param,
                     "share": e_param / total if total else 0,
                     "n": len(sub)})
    df = pd.DataFrame(rows).sort_values("share", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["family"], df["share"] * 100, color="#cc4c4c", alpha=0.85)
    for i, (s, n) in enumerate(zip(df["share"], df["n"])):
        ax.text(i, s * 100 + 1, f"n={n:,}", ha="center", fontsize=8, color="#444")
    ax.set_ylabel("Share of OOS Sharpe variance from V_param (%)")
    ax.set_xlabel("Indicator family")
    ax.set_title("Parameter-sensitivity dominance by indicator family")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Synthetic demo
# -------------------------------------------------------------------

def synthetic_demo(n_strategies: int = 1500, n_windows: int = 6,
                   seed: int = 42) -> pd.DataFrame:
    """Build a fake long-format strategy table with three regimes of strategies:
    'robust' (low V_param), 'fragile' (high V_param), 'noise'.
    """
    rng = np.random.default_rng(seed)
    rows = []
    families = ["EMA", "RSI", "ATR", "MACD"]
    classes = rng.choice(["robust", "fragile", "noise"], size=n_strategies,
                         p=[0.35, 0.25, 0.40])
    for s in range(n_strategies):
        cls = classes[s]
        fam = rng.choice(families)
        base_sharpe_is = rng.normal(0.4, 0.6)
        base_sharpe_oos = base_sharpe_is - rng.normal(0.3, 0.4)
        sigma_param = {"robust": 0.05, "fragile": 0.5, "noise": 0.2}[cls]
        for w in range(1, n_windows + 1):
            for sample in ["IS", "OOS"]:
                center = base_sharpe_is if sample == "IS" else base_sharpe_oos
                # Per-window noise
                center += rng.normal(0, 0.25)
                for tag in TEST_TAGS:
                    perturbation = 0.0 if tag == "raw" else rng.normal(0, sigma_param)
                    rows.append({
                        "asset": "SYNTH",
                        "family": fam,
                        "strategy_name": f"strat{s:05d}",
                        "primary": fam,
                        "transform": "EMA100",
                        "confluence": None,
                        "sl": 1.0,
                        "window_idx": w,
                        "sample": sample,
                        "test_tag": tag,
                        "lb": 50,
                        "n_trades": int(rng.integers(60, 220)),
                        "roi": float(rng.normal(50, 100)),
                        "pf": float(np.exp(rng.normal(0, 0.2))),
                        "sharpe": float(center + perturbation),
                        "max_dd": float(rng.normal(400, 150)),
                    })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def per_asset_decile_table(live: pd.DataFrame) -> pd.DataFrame:
    """For each asset: D1, D5, D10 live-proxy profitable rates and the
    D1 vs D10 lift (in percentage points). Reports the headline predictive
    validity asset-by-asset.
    """
    rows = []
    for asset, sub in live.groupby("asset"):
        if len(sub) < 100:
            continue
        try:
            bins = pd.qcut(sub["v_param_hist"], q=10, duplicates="drop")
        except Exception:
            continue
        rate = sub.groupby(bins, observed=True)["live_profitable"].mean() * 100
        if len(rate) < 5:
            continue
        d1 = float(rate.iloc[0])
        d10 = float(rate.iloc[-1])
        d5 = float(rate.iloc[len(rate) // 2])
        rows.append({
            "asset": asset,
            "n": len(sub),
            "live_profit_pct": float(sub["live_profitable"].mean() * 100),
            "D1_pct": d1, "D5_pct": d5, "D10_pct": d10,
            "D1_minus_D10_pp": d1 - d10,
        })
    return pd.DataFrame(rows).sort_values("D1_minus_D10_pp", ascending=False)


def run(df: pd.DataFrame, label: str):
    t0 = time.time()
    wide = pivot_wide(df)
    print(f"[{label}] pivoted to {len(wide):,} rows in {time.time()-t0:.1f}s")
    per_row = per_strategy_decomp(wide)
    parts = aggregate_variance(per_row)
    print(f"[{label}] OOS Sharpe variance decomposition:")
    for k, v in parts.items():
        print(f"  {k:25s} {v:.4f}")
    deg = degradation(wide)
    print(f"[{label}] mean IS->OOS degradation: {deg['degradation'].mean():.3f} "
          f"(median {deg['degradation'].median():.3f})")
    live = live_proxy_outcomes(wide)
    print(f"[{label}] live-proxy strategies: {len(live):,}, "
          f"profitable={live['live_profitable'].mean()*100:.1f}%")
    if len(live) > 0:
        # Decile predictive validity (pooled)
        bins = pd.qcut(live["v_param_hist"], q=10, duplicates="drop")
        rate = live.groupby(bins, observed=True)["live_profitable"].mean() * 100
        print(f"[{label}] live-proxy profitability by V_param decile (D1=low → D10=high):")
        for i, r in enumerate(rate.values):
            print(f"  D{i+1:>2} {r:.1f}%")
        # Per-asset breakdown
        per_asset = per_asset_decile_table(live)
        if not per_asset.empty:
            print(f"[{label}] per-asset D1 vs D10 lift "
                  f"(positive = robustness predicts edge):")
            print(per_asset.to_string(index=False))
            per_asset.to_csv(OUT_FIG.parent / f"per_asset_decile_{label}.csv",
                             index=False)

    # Plots
    plot_decomposition_pie(parts, OUT_FIG / f"fig_decomposition_{label}.png")
    plot_param_vs_live(live, OUT_FIG / f"fig_param_vs_live_{label}.png")
    plot_decomp_by_family(per_row, OUT_FIG / f"fig_decomp_by_family_{label}.png")

    return parts, per_row, live


def list_asset_partitions(parquet_root: str) -> list[str]:
    out = []
    for d in sorted(os.listdir(parquet_root)):
        if d.startswith("asset="):
            out.append(d.replace("asset=", ""))
    return out


def run_pooled(parquet_root: str, assets: list[str] | None = None,
               label: str = "data"):
    """Stream per-asset, aggregate variance components and live-proxy rows
    pooled. Bounded memory: only one asset's frame in RAM at a time.
    """
    if assets is None:
        assets = list_asset_partitions(parquet_root)
    print(f"[{label}] streaming {len(assets)} assets from {parquet_root}",
          flush=True)

    pooled_live = []
    pooled_oos_count = 0
    # accumulators for variance decomposition
    sum_sharpe = 0.0
    sum_sharpe_sq = 0.0
    sum_v_param = 0.0
    sum_v_finite = 0.0
    n_oos = 0
    # for v_strategy_main: collect per-strategy means
    strat_means_list = []
    # for v_window_main: collect per-window means
    window_means: dict = {}  # window_idx -> [sum, count]
    pooled_deg_means = []

    per_asset_summary = []
    for ai, asset in enumerate(assets):
        t0 = time.time()
        df = load_metrics(parquet_root, assets=[asset])
        wide = pivot_wide(df)
        if wide.empty:
            print(f"  [{ai+1}/{len(assets)}] {asset}: empty after pivot, skipping",
                  flush=True)
            continue
        per_row = per_strategy_decomp(wide)
        oos = per_row[per_row["sample"] == "OOS"].dropna(subset=["sharpe_raw"])
        if oos.empty:
            print(f"  [{ai+1}/{len(assets)}] {asset}: no OOS rows", flush=True)
            continue

        # pooled accumulators
        s = oos["sharpe_raw"].to_numpy()
        sum_sharpe += s.sum()
        sum_sharpe_sq += (s * s).sum()
        sum_v_param += float(oos["v_param"].sum())
        sum_v_finite += float(oos["v_finite"].sum())
        n_oos += len(s)
        strat_mean_g = oos.groupby(["asset", "strategy_name"])["sharpe_raw"].mean()
        strat_means_list.append(strat_mean_g.values)
        for w, g in oos.groupby("window_idx")["sharpe_raw"]:
            row = window_means.setdefault(int(w), [0.0, 0])
            row[0] += float(g.sum())
            row[1] += int(len(g))

        # degradation
        deg = degradation(wide)
        pooled_deg_means.append(deg["degradation"].mean())

        # live-proxy rows
        live = live_proxy_outcomes(wide)
        if not live.empty:
            pooled_live.append(live)

        per_asset_summary.append({
            "asset": asset,
            "n_oos_rows": len(s),
            "mean_oos_sharpe": float(s.mean()),
            "v_param_avg": float(oos["v_param"].mean()),
            "live_n": int(len(live)),
            "live_profit_pct": float(live["live_profitable"].mean() * 100)
                if len(live) else None,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  [{ai+1}/{len(assets)}] {asset}: "
              f"{len(s):,} OOS rows, live n={len(live):,}, "
              f"t={time.time()-t0:.1f}s", flush=True)

    # final pooled variance decomposition
    mean_sharpe = sum_sharpe / max(1, n_oos)
    var_sharpe = sum_sharpe_sq / max(1, n_oos) - mean_sharpe * mean_sharpe
    e_param = sum_v_param / max(1, n_oos)
    e_finite = sum_v_finite / max(1, n_oos)
    strat_means = np.concatenate(strat_means_list) if strat_means_list else np.array([0.0])
    v_strategy = float(np.var(strat_means, ddof=0))
    win_means = np.array([s / max(1, c) for (s, c) in window_means.values()])
    v_window = float(np.var(win_means, ddof=0)) if len(win_means) else 0.0
    residual = max(var_sharpe - (e_param + v_strategy + v_window + e_finite), 0.0)
    parts = {
        "total": float(var_sharpe),
        "v_param_avg": float(e_param),
        "v_strategy_main": v_strategy,
        "v_window_main": v_window,
        "v_finite_avg": float(e_finite),
        "residual": residual,
        "share_v_param": e_param / var_sharpe if var_sharpe else 0,
        "share_v_strategy": v_strategy / var_sharpe if var_sharpe else 0,
        "share_v_window": v_window / var_sharpe if var_sharpe else 0,
        "share_v_finite": e_finite / var_sharpe if var_sharpe else 0,
        "share_residual": residual / var_sharpe if var_sharpe else 0,
    }

    print(f"\n[{label}] pooled OOS Sharpe variance decomposition (n={n_oos:,}):")
    for k, v in parts.items():
        print(f"  {k:25s} {v:.4f}")
    print(f"[{label}] mean IS->OOS degradation: "
          f"{np.mean(pooled_deg_means):.3f}")

    if pooled_live:
        live_all = pd.concat(pooled_live, ignore_index=True)
        print(f"[{label}] live-proxy strategies: {len(live_all):,}, "
              f"profitable={live_all['live_profitable'].mean()*100:.1f}%")
        bins = pd.qcut(live_all["v_param_hist"], q=10, duplicates="drop")
        rate = live_all.groupby(bins, observed=True)["live_profitable"].mean() * 100
        print(f"[{label}] pooled live-proxy profitability by V_param decile:")
        for i, r in enumerate(rate.values):
            print(f"  D{i+1:>2} {r:.1f}%")
        per_asset_table = per_asset_decile_table(live_all)
        if not per_asset_table.empty:
            print(f"[{label}] per-asset D1 vs D10 lift:")
            print(per_asset_table.to_string(index=False))
            per_asset_table.to_csv(OUT_FIG.parent / f"per_asset_decile_{label}.csv",
                                   index=False)
        # plots
        per_row_all = pd.DataFrame()  # not used for pie plot; use parts directly
        plot_decomposition_pie(parts, OUT_FIG / f"fig_decomposition_{label}.png")
        plot_param_vs_live(live_all, OUT_FIG / f"fig_param_vs_live_{label}.png")

    pd.DataFrame(per_asset_summary).to_csv(
        OUT_FIG.parent / f"per_asset_summary_{label}.csv", index=False)
    return parts, per_asset_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root",
                    default="/mnt/d/strategies_parquet/strategies")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="(rare) run on a synthetic three-class demo instead "
                         "of the real corpus; only use if explicitly testing "
                         "the analysis machinery in isolation.")
    args = ap.parse_args()

    out_summary = {"mode": "data"}
    if args.synthetic:
        out_summary["mode"] = "synthetic"
        df = synthetic_demo()
        parts, per_row, live = run(df, label="synthetic")
        out_summary["live_proxy_n"] = int(len(live))
        out_summary["live_profit_rate_pct"] = (
            float(live["live_profitable"].mean() * 100) if len(live) else None
        )
    else:
        parts, per_asset = run_pooled(args.parquet_root, assets=args.assets,
                                      label="data")
        out_summary["per_asset"] = per_asset
        out_summary["decomposition"] = parts

    with open(RESULTS_JSON, "w") as f:
        json.dump(out_summary, f, indent=2, default=float)
    print(f"\nresults summary -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
