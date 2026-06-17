"""Generate the standard set of diagnostic plots for a TAGC run.

Usage:
    python scripts/visualize.py runs/local
    python scripts/visualize.py runs/local --out runs/local/figures

Inputs (all auto-detected inside the run dir):
    metrics.csv               per-epoch train/val curves
    predictions_test.csv      per-day p_mean, p_var, pred, label, position_size
    graphs/*.npz              per-day adjacency snapshots from the test pass
    config.json               (read for the target ticker label)

Outputs (PNG files, written to --out, default <run>/figures):
    1_training_curves.png     loss + val accuracy per epoch
    2_predictions_overview.png  p_mean / labels / position size / cum-correct / confusion
    3_calibration.png         reliability diagram (binned p vs empirical up-rate)
    4_graph_summary.png       eps_t, edge density, degree dist, neighbour heatmap
    5_top_neighbours.png      most frequent neighbours of the target stock
    6_strategy_pnl.png        cumulative simulated PnL of position_size × next-day return
    7_graph_network.png       target-centric network plot on 4 days (graph evolution)
    8_full_graph.png          spring-layout of the FULL universe (all 98 stocks)
                              averaged across the test period, reveals clusters
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _safe_load_metrics(run: Path) -> Optional[pd.DataFrame]:
    p = run / "metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    # Fallback: parse train.log lines like  "epoch 03  train_loss=… val_loss=… val_acc=… pos=…"
    log_path = run / "train.log"
    if not log_path.exists():
        return None
    pat = re.compile(
        r"epoch (\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)\s+pos=([\d.]+)"
    )
    rows = []
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            rows.append([int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["epoch", "train_loss", "val_loss", "val_acc", "val_pos_rate"])
    df["best_val_so_far"] = df["val_loss"].cummin()
    return df


def _load_predictions(run: Path) -> Optional[pd.DataFrame]:
    p = run / "predictions_test.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "pred_score" in df.columns:        # v3.6 universe-wide ranking schema
        df.attrs["schema"] = "ranking"
        return df
    return _alias_v2_to_v1(df)


def _run_horizon(run: Path) -> int:
    cfg = run / "config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text()).get("target_horizon", 5))
        except Exception:
            pass
    return 5


def plot_ranking_overview(preds: pd.DataFrame, horizon: int, out: Path) -> None:
    """v3.6 universe-wide ranking diagnostics from the per-(date,ticker) CSV
    (pred_score, pred_rank, pred_direction, true_fwd_return): daily Rank-IC,
    long-short decile PnL, decile monotonicity, and a summary."""
    from scipy.stats import spearmanr
    days, rics, ls = [], [], []
    for dt, g in preds.groupby("date"):
        s = g["pred_score"].to_numpy(); r = g["true_fwd_return"].to_numpy()
        ok = np.isfinite(s) & np.isfinite(r); s, r = s[ok], r[ok]; K = s.size
        if K < 10 or s.std() == 0:
            continue
        days.append(pd.Timestamp(dt)); rics.append(float(spearmanr(s, r).statistic))
        nleg = max(1, int(round(0.1 * K))); o = np.argsort(s)
        ls.append(float(r[o[-nleg:]].mean() - r[o[:nleg]].mean()))
    if not days:
        return
    days = np.array(days); rics = np.array(rics); ls = np.array(ls)
    idx = np.arange(0, len(days), max(1, horizon))      # non-overlapping for PnL
    ls_no = ls[idx]; d_no = days[idx]
    cum = np.cumprod(1.0 + ls_no) - 1.0
    ap = preds.dropna(subset=["pred_rank", "true_fwd_return"])
    dec = np.clip((ap["pred_rank"].to_numpy() * 10).astype(int), 0, 9)
    rr = ap["true_fwd_return"].to_numpy()
    dmean = [float(rr[dec == k].mean()) if (dec == k).any() else 0.0 for k in range(10)]
    diracc = float(((preds["pred_direction"] > 0) == (preds["true_fwd_return"] > 0)).mean())
    ic_m = float(np.nanmean(rics)); ic_s = float(np.nanstd(rics))
    sharpe = float(ls_no.mean() / ls_no.std() * np.sqrt(252 / horizon)) if ls_no.std() > 0 else float("nan")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(days, np.cumsum(rics), color="#1f77b4"); ax[0, 0].axhline(0, color="gray", lw=.6)
    ax[0, 0].set_title(f"(a) cumulative daily Rank-IC   (mean {ic_m:+.4f}, ICIR {ic_m/max(ic_s,1e-9):+.2f})")
    ax[0, 0].grid(alpha=.3)
    ax[0, 1].plot(d_no, cum * 100, color="#2ca02c"); ax[0, 1].axhline(0, color="gray", lw=.6)
    ax[0, 1].set_title(f"(b) long-short cumulative return (top/bottom decile, {horizon}d)\n"
                       f"Sharpe {sharpe:+.2f} · hit {np.mean(ls_no>0):.2f}")
    ax[0, 1].set_ylabel("cumulative %"); ax[0, 1].grid(alpha=.3)
    ax[1, 0].bar(range(10), [x * 100 for x in dmean], color="#4c78a8")
    ax[1, 0].set_title("(c) mean fwd return by predicted-rank decile (monotone ⇒ ranking works)")
    ax[1, 0].set_xlabel("predicted decile (0=bottom, 9=top)"); ax[1, 0].set_ylabel("mean fwd return %")
    ax[1, 0].grid(alpha=.3)
    ax[1, 1].axis("off")
    txt = (f"horizon         {horizon}d\n"
           f"test days       {len(days)}\n"
           f"universe/day    {int(preds.groupby('date').size().mean())}\n"
           f"Rank-IC         {ic_m:+.4f} ± {ic_s:.4f}\n"
           f"Rank-ICIR       {ic_m/max(ic_s,1e-9):+.3f}\n"
           f"direction acc   {diracc:.3f}\n"
           f"L/S Sharpe      {sharpe:+.2f}\n"
           f"L/S hit-rate    {np.mean(ls_no>0):.3f}\n"
           f"cum return      {cum[-1]:+.1%}\n")
    ax[1, 1].text(0.02, 0.95, txt, family="monospace", va="top", fontsize=12)
    ax[1, 1].set_title("(d) summary")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def _alias_v2_to_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Make v2 prediction CSVs look like v1's column schema for the existing
    plot code. v2 columns: predicted_return, predicted_std, true_return,
    predicted_direction, true_direction, direction_correct. v1 had: p_mean,
    p_var, label, pred, correct. We alias the v2 columns under the v1 names
    so downstream plots don't need to branch, and we keep the v2 originals too."""
    if "predicted_direction" in df.columns and "pred" not in df.columns:
        df = df.copy()
        df["pred"] = df["predicted_direction"]
        df["label"] = df["true_direction"]
        df["correct"] = df["direction_correct"]
        # NOTE: in v2 "p_mean" is the predicted RETURN (raw), not a probability.
        # Plot code that assumes p_mean in [0,1] will get larger, centered-at-0
        # values, the y-axis labels still read "predicted return".
        df["p_mean"] = df["predicted_return"]
        df["p_var"] = df["predicted_std"] ** 2
        df.attrs["schema"] = "v2"
    else:
        df.attrs["schema"] = "v1"
    return df


def _load_graphs(run: Path):
    files = sorted(glob.glob(str(run / "graphs" / "test_*.npz")))
    out = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        out.append({
            "date": str(d["date"]),
            "adj": d["adj"].astype(np.float32),
            "eps": float(d["eps"]),
            "tickers": d["tickers"].tolist(),
        })
    return out


def _target_ticker(run: Path) -> str:
    cfg = run / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text()).get("target_ticker", "TARGET").upper()
        except Exception:
            pass
    return "TARGET"


# ----------------------------------------------------------------------------
# 1. training curves
# ----------------------------------------------------------------------------
def plot_training_curves(metrics: pd.DataFrame, out: Path) -> None:
    """Per-epoch training/validation diagnostics.

    For v2 (regression) runs we show a 2×2 grid that surfaces the metric the
    model is actually SELECTED on, the validation Rank-IC, which the old
    2-panel layout hid entirely:

        [0,0] Loss          train_mse vs val_mse
        [0,1] Ranking loss  train_rank_loss vs val_rank_loss (ListMLE)
        [1,0] X-sec skill    val IC + val Rank-IC per epoch, with the running
                             best-Rank-IC (the early-stop selection metric) and
                             a 0 chance line, the headline learning curve
        [1,1] Direction acc  val_dir_acc + val RMSE (twin axis)

    v1 (legacy BCE) runs fall back to the original 2-panel loss+accuracy view.
    """
    is_v2 = "val_rank_ic" in metrics.columns or "train_mse" in metrics.columns
    ep = metrics["epoch"]

    # ---- v1 legacy fallback (BCE classification) ---------------------------
    if not is_v2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ep, metrics["train_loss"], "o-", label="train", color="#1f77b4")
        axes[0].plot(ep, metrics["val_loss"], "o-", label="val", color="#d62728")
        if "best_val_so_far" in metrics:
            axes[0].plot(ep, metrics["best_val_so_far"], "--", label="best val",
                         color="#2ca02c", alpha=0.7)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        if "val_acc" in metrics:
            axes[1].plot(ep, metrics["val_acc"], "o-", color="#9467bd", label="val acc")
        axes[1].axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
        axes[1].set_title("Validation accuracy"); axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
        return

    # ---- v2 (regression + ranking) 2×2 grid --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) MSE loss
    ax = axes[0, 0]
    if "train_mse" in metrics:
        ax.plot(ep, metrics["train_mse"], "o-", label="train", color="#1f77b4", ms=3)
    if "val_mse" in metrics:
        ax.plot(ep, metrics["val_mse"], "o-", label="val", color="#d62728", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled target units)")
    ax.set_title("(a) Huber/MSE loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Ranking (ListMLE) loss
    ax = axes[0, 1]
    if "train_rank_loss" in metrics:
        ax.plot(ep, metrics["train_rank_loss"], "o-", label="train", color="#1f77b4", ms=3)
    if "val_rank_loss" in metrics:
        ax.plot(ep, metrics["val_rank_loss"], "o-", label="val", color="#d62728", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("ListMLE ranking loss")
    ax.set_title("(b) Cross-sectional ranking loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) THE headline curve, validation IC + Rank-IC + running best
    ax = axes[1, 0]
    if "val_ic" in metrics:
        ax.plot(ep, metrics["val_ic"], "o-", label="val IC (Pearson)", color="#1f77b4", ms=3)
    if "val_rank_ic" in metrics:
        ax.plot(ep, metrics["val_rank_ic"], "o-", label="val Rank-IC (Spearman)",
                color="#9467bd", ms=4, lw=2)
    if "best_rank_ic_so_far" in metrics:
        ax.plot(ep, metrics["best_rank_ic_so_far"], "--", label="best Rank-IC (selected)",
                color="#2ca02c", alpha=0.8)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, label="0 (no skill)")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-sectional correlation")
    ax.set_title("(c) Validation IC / Rank-IC  ←  the selection metric")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Direction accuracy + RMSE on a twin axis
    ax = axes[1, 1]
    if "val_dir_acc" in metrics:
        ax.plot(ep, metrics["val_dir_acc"], "o-", label="val dir-acc", color="#9467bd", ms=3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch"); ax.set_ylabel("direction accuracy")
    ax.set_title("(d) Direction accuracy + RMSE")
    if "val_rmse_pct" in metrics:
        ax2 = ax.twinx()
        ax2.plot(ep, metrics["val_rmse_pct"] * 100, "^-", alpha=0.6,
                 color="#8c564b", label="val RMSE (%)", ms=3)
        ax2.set_ylabel("RMSE (% return)", color="#8c564b")
        ax2.tick_params(axis="y", colors="#8c564b")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 2. predictions overview
# ----------------------------------------------------------------------------
def plot_predictions_overview(preds: pd.DataFrame, target: str, out: Path) -> None:
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1.4])

    # (a) prediction over time with true labels as colored dots.
    # Schema-aware: v1 = probability in [0,1], v2 = predicted RETURN (real number).
    is_v2 = preds.attrs.get("schema") == "v2"
    ax0 = fig.add_subplot(gs[0, :])
    y_lbl = "predicted return" if is_v2 else "p_mean"
    title_suffix = "predicted return vs. actual moves" if is_v2 else "predicted up-probability vs. actual moves"
    centre = 0.0 if is_v2 else 0.5
    pmax = max(abs(preds["p_mean"].min()), abs(preds["p_mean"].max())) if is_v2 else 0.55
    pad  = pmax * 0.25 if is_v2 else 0.05
    ax0.plot(preds["date"], preds["p_mean"], color="#1f77b4", lw=1, label=y_lbl)
    ax0.fill_between(preds["date"],
                     preds["p_mean"] - preds["p_var"]**0.5,
                     preds["p_mean"] + preds["p_var"]**0.5,
                     alpha=0.2, color="#1f77b4", label="±1 std (MC dropout)")
    ax0.axhline(centre, color="gray", ls="--", lw=0.8)
    up = preds[preds["label"] == 1]; down = preds[preds["label"] == 0]
    y_up   = (pmax + pad) if is_v2 else 1.02
    y_down = -(pmax + pad) if is_v2 else -0.02
    ax0.scatter(up["date"],   [y_up] * len(up),     marker="^", color="#2ca02c", s=18, label="actual UP")
    ax0.scatter(down["date"], [y_down] * len(down), marker="v", color="#d62728", s=18, label="actual DOWN")
    if is_v2:
        ax0.set_ylim(y_down - pad * 0.2, y_up + pad * 0.2)
    else:
        ax0.set_ylim(-0.05, 1.10)
    ax0.set_ylabel(y_lbl)
    ax0.set_title(f"{target} — {title_suffix}")
    ax0.legend(loc="upper right", ncols=4); ax0.grid(alpha=0.3)

    # (b) position size over time
    ax1 = fig.add_subplot(gs[1, :])
    colors = ["#2ca02c" if c == 1 else "#d62728" for c in preds["correct"]]
    ax1.bar(preds["date"], preds["position_size"], color=colors, width=0.9)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("position_size  [-1, +1]")
    ax1.set_title("Daily position size (green = correct direction, red = wrong)")
    ax1.grid(alpha=0.3)

    # (c) cumulative correct vs random walk
    ax2 = fig.add_subplot(gs[2, 0])
    cum_correct = preds["correct"].cumsum().values
    ax2.plot(range(1, len(preds) + 1), cum_correct, label="model", color="#1f77b4")
    ax2.plot(range(1, len(preds) + 1), 0.5 * np.arange(1, len(preds) + 1),
             "--", color="gray", label="50% chance")
    ax2.set_xlabel("test day #"); ax2.set_ylabel("cumulative # correct")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_title("Cumulative correct calls")

    # (d) p_var over time (uncertainty)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(preds["date"], preds["p_var"], color="#ff7f0e", lw=1.2)
    ax3.set_ylabel("p_var (MC)"); ax3.set_title("Prediction uncertainty"); ax3.grid(alpha=0.3)

    # (e) confusion matrix
    ax4 = fig.add_subplot(gs[3, 0])
    cm = pd.crosstab(preds["label"], preds["pred"], rownames=["actual"], colnames=["pred"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0)
    ax4.imshow(cm.values, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm.values):
        ax4.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.values.max() / 2 else "black", fontsize=14)
    ax4.set_xticks([0, 1]); ax4.set_xticklabels(["DOWN", "UP"])
    ax4.set_yticks([0, 1]); ax4.set_yticklabels(["DOWN", "UP"])
    ax4.set_xlabel("predicted"); ax4.set_ylabel("actual")
    ax4.set_title(f"Confusion matrix  (acc = {preds['correct'].mean():.3f})")

    # (f) summary text
    ax5 = fig.add_subplot(gs[3, 1]); ax5.axis("off")
    acc = preds["correct"].mean()
    avg_pos = preds["position_size"].abs().mean()
    long_pct = (preds["position_size"] > 0).mean()
    pos_w_acc = (preds["correct"] * preds["position_size"].abs()).sum() / max(
        preds["position_size"].abs().sum(), 1e-9)
    text = (
        f"target stock        {target}\n"
        f"test days           {len(preds)}\n"
        f"date range          {preds['date'].min().date()}  →  {preds['date'].max().date()}\n"
        f"raw accuracy        {acc:.3f}\n"
        f"|position| mean     {avg_pos:.3f}\n"
        f"long days           {long_pct:.1%}\n"
        f"p_mean range        {preds['p_mean'].min():.3f}  →  {preds['p_mean'].max():.3f}\n"
        f"|pos|-weighted acc  {pos_w_acc:.3f}\n"
    )
    ax5.text(0.02, 0.95, text, family="monospace", va="top", fontsize=11)
    ax5.set_title("Run summary")

    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 3. reliability diagram (calibration)
# ----------------------------------------------------------------------------
def plot_calibration(preds: pd.DataFrame, out: Path, n_bins: int = 10) -> None:
    if preds.attrs.get("schema") == "v2":
        # Regression mode: reliability is meaningless. Show a residual scatter
        # of predicted vs actual return instead.
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
        actual = preds.get("true_return", preds["label"]).astype(float)
        pred = preds["p_mean"]
        a0.scatter(actual, pred, s=14, alpha=0.6, color="#1f77b4")
        lim = max(abs(actual.min()), abs(actual.max()), abs(pred.min()), abs(pred.max())) * 1.1
        a0.plot([-lim, lim], [-lim, lim], "--", color="gray", label="y=x (perfect)")
        a0.set_xlabel("true next-day return"); a0.set_ylabel("predicted return")
        a0.set_title("Predicted vs. actual return (regression scatter)")
        a0.set_xlim(-lim, lim); a0.set_ylim(-lim, lim); a0.legend(); a0.grid(alpha=0.3)

        resid = pred - actual
        a1.hist(resid, bins=30, color="#ff7f0e", edgecolor="white")
        a1.axvline(0, color="gray", ls="--", lw=0.8)
        a1.set_xlabel("residual (predicted − actual)"); a1.set_ylabel("count")
        a1.set_title(f"Residual distribution  (RMSE = {float(np.sqrt((resid**2).mean())):.4f})")
        a1.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
        return

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(preds["p_mean"].values, bins[1:-1])
    bin_centers, frac_up, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_centers.append(preds["p_mean"].values[mask].mean())
        frac_up.append(preds["label"].values[mask].mean())
        counts.append(int(mask.sum()))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    a0.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    a0.plot(bin_centers, frac_up, "o-", color="#1f77b4")
    a0.set_xlabel("predicted p_mean"); a0.set_ylabel("empirical up rate")
    a0.set_title("Reliability diagram"); a0.set_xlim(0, 1); a0.set_ylim(0, 1)
    a0.legend(); a0.grid(alpha=0.3)

    a1.hist(preds["p_mean"], bins=bins, color="#1f77b4", edgecolor="white")
    a1.set_xlabel("p_mean"); a1.set_ylabel("count")
    a1.set_title("Predicted-probability histogram"); a1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 4. graph summary
# ----------------------------------------------------------------------------
def plot_graph_summary(graphs, target: str, out: Path) -> None:
    if not graphs:
        return
    dates = [pd.Timestamp(g["date"]) for g in graphs]
    eps   = np.array([g["eps"] for g in graphs])
    # Edge density = mean off-diagonal of A.
    densities = []
    out_deg, in_deg = [], []
    for g in graphs:
        A = g["adj"]
        n = A.shape[0]
        off_diag = A.sum() / (n * (n - 1))
        densities.append(off_diag)
        binary = (A > 0).astype(np.float32)
        out_deg.append(binary.sum(axis=1).mean())
        in_deg.append(binary.sum(axis=0).mean())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(dates, eps, color="#1f77b4")
    # eps is the learned edge-sphere threshold (eps_base). It is only
    # "macro-conditioned" when cfg.use_macro_eps=True (default False), so the
    # generic title would be misleading in a thesis appendix.
    # KNOW this only reads eps_base, not the per-day macro-conditioned eps_t
    axes[0, 0].set_title("ε  (learned edge-sphere threshold, eps_base)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(dates, densities, color="#ff7f0e")
    axes[0, 1].set_title("Mean edge weight (off-diagonal of A)")
    axes[0, 1].set_ylim(0, max(densities) * 1.1 if max(densities) > 0 else 1)
    axes[0, 1].grid(alpha=0.3)

    # Degree distribution histogram aggregated across all test days.
    all_out_deg = np.concatenate([(g["adj"] > 0).sum(axis=1) for g in graphs])
    axes[1, 0].hist(all_out_deg, bins=30, color="#2ca02c", edgecolor="white")
    axes[1, 0].set_xlabel("out-degree (# of edges per stock per day)")
    axes[1, 0].set_ylabel("count (across stocks × days)")
    axes[1, 0].set_title("Out-degree distribution")
    axes[1, 0].grid(alpha=0.3)

    # Target stock's neighbour heatmap: rows = days, cols = top-20 stocks by avg edge weight to target.
    tickers = graphs[0]["tickers"]
    if target in tickers:
        ti = tickers.index(target)
        target_rows = np.stack([g["adj"][ti] for g in graphs], axis=0)   # [D, N]
        avg_weight = target_rows.mean(axis=0)
        top20 = np.argsort(-avg_weight)[:20]
        heat = target_rows[:, top20]
        im = axes[1, 1].imshow(heat, aspect="auto", cmap="viridis")
        axes[1, 1].set_yticks(range(0, len(dates), max(1, len(dates) // 8)))
        axes[1, 1].set_yticklabels([dates[i].strftime("%Y-%m-%d")
                                    for i in range(0, len(dates), max(1, len(dates) // 8))])
        axes[1, 1].set_xticks(range(len(top20)))
        axes[1, 1].set_xticklabels([tickers[j] for j in top20], rotation=90, fontsize=8)
        axes[1, 1].set_title(f"{target} → top-20 neighbours by mean edge weight")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 5. top neighbours table-plot
# ----------------------------------------------------------------------------
def plot_top_neighbours(graphs, target: str, out: Path, top_k: int = 15) -> None:
    if not graphs:
        return
    tickers = graphs[0]["tickers"]
    if target not in tickers:
        return
    ti = tickers.index(target)
    # Average outgoing edge weight from target.
    avg_w = np.mean([g["adj"][ti] for g in graphs], axis=0)
    edge_count = np.sum([(g["adj"][ti] > 0).astype(int) for g in graphs], axis=0)

    order = np.argsort(-avg_w)
    order = [j for j in order if j != ti][:top_k]
    labels = [tickers[j] for j in order]
    weights = [avg_w[j] for j in order]
    counts = [edge_count[j] for j in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(labels[::-1], weights[::-1], color="#1f77b4")
    axes[0].set_xlabel("mean edge weight"); axes[0].set_title(f"{target} — avg edge weight to neighbours")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(labels[::-1], counts[::-1], color="#ff7f0e")
    axes[1].set_xlabel(f"# days neighbour (of {len(graphs)})")
    axes[1].set_title(f"{target} — neighbour persistence")
    axes[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 6. crude PnL chart
# ----------------------------------------------------------------------------
def plot_strategy_pnl(preds: pd.DataFrame, target: str, run: Path, out: Path) -> None:
    """Cumulative simulated PnL of position_size_t × ret_t  (direction-only).

    We approximate ret_t from the label: +1 for UP, -1 for DOWN (so this is a
    direction-only proxy, not real return magnitude). Useful for a quick "is
    the position-sizing rule helping?" check.
    """
    # KNOW this is a sign-only proxy, it ignores how big each move actually was
    direction_ret = np.where(preds["label"].values == 1, 1.0, -1.0)
    pnl = preds["position_size"].values * direction_ret
    cum = np.cumsum(pnl)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(preds["date"], cum, color="#1f77b4", label="cum PnL (pos × sign(ret))")
    ax.axhline(0, color="gray", lw=0.6)
    ax.fill_between(preds["date"], cum, 0, where=(cum >= 0), color="#2ca02c", alpha=0.2)
    ax.fill_between(preds["date"], cum, 0, where=(cum < 0),  color="#d62728", alpha=0.2)
    ax.set_title(f"{target} — cumulative direction-proxy PnL of position_size strategy")
    ax.set_ylabel("cumulative ∑ pos·sign(ret)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 7. graph network, target-centric radial layout, 4 days side by side
# ----------------------------------------------------------------------------
def _draw_radial_graph(ax, adj, tickers, target, top_k=20, title=""):
    """Plot target at the centre, top-k neighbours by edge weight on a ring,
    every other stock on an outer ring (greyed out)."""
    ti = tickers.index(target)
    row = adj[ti].copy()
    row[ti] = 0.0
    nz_mask = row > 0
    # Pick the k strongest neighbours of the target.
    order = np.argsort(-row)
    top = [j for j in order if j != ti][:top_k]
    top_weights = row[top]

    # Radial layout: target at origin, top-k on inner ring, rest on outer ring.
    n_total = len(tickers)
    other = [j for j in range(n_total) if j != ti and j not in top]
    inner_angles = np.linspace(0, 2 * np.pi, len(top), endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, max(len(other), 1), endpoint=False)
    inner_r, outer_r = 1.0, 1.7

    pos = {ti: (0.0, 0.0)}
    for j, ang in zip(top, inner_angles):
        pos[j] = (inner_r * np.cos(ang), inner_r * np.sin(ang))
    for j, ang in zip(other, outer_angles):
        pos[j] = (outer_r * np.cos(ang), outer_r * np.sin(ang))

    # Draw edges from target to ALL non-zero neighbours, colored by weight.
    max_w = float(row.max()) if row.max() > 0 else 1.0
    for j in range(n_total):
        if j == ti or row[j] <= 0:
            continue
        w = row[j] / max_w
        x0, y0 = pos[ti]; x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1],
                color=plt.cm.viridis(w),
                lw=0.5 + 2.5 * w, alpha=0.4 + 0.6 * w, zorder=1)

    # Draw outer-ring (non-neighbour) nodes
    other_xs = [pos[j][0] for j in other]
    other_ys = [pos[j][1] for j in other]
    ax.scatter(other_xs, other_ys, s=20, color="lightgray", zorder=2, edgecolor="white", lw=0.5)

    # Draw top-k neighbour nodes with size ∝ edge weight, labelled
    inner_xs = [pos[j][0] for j in top]
    inner_ys = [pos[j][1] for j in top]
    sizes = 60 + 240 * (top_weights / max_w)
    ax.scatter(inner_xs, inner_ys, s=sizes, color="#ff7f0e",
               zorder=3, edgecolor="black", lw=0.8)
    for j, x, y in zip(top, inner_xs, inner_ys):
        ax.annotate(tickers[j], (x, y), fontsize=7, ha="center", va="center",
                    xytext=(0, 12), textcoords="offset points")

    # Draw the target node
    ax.scatter([0], [0], s=400, color="#d62728", zorder=4, edgecolor="black", lw=1.2)
    ax.annotate(target, (0, 0), fontsize=10, fontweight="bold",
                ha="center", va="center", color="white")

    # eps circle: visual cue showing roughly where the sphere boundary sits.
    # We don't store h_t so this is conceptual, just placed at inner ring radius.
    circle = plt.Circle((0, 0), inner_r, fill=False, ls="--",
                        color="#1f77b4", alpha=0.35, lw=1.2)
    ax.add_patch(circle)

    # Total edges + degree from this graph row, for the panel title.
    deg = int(nz_mask.sum())
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}\n{target} → {deg} edges, top {top_k} shown",
                 fontsize=10)


def plot_graph_network(graphs, target: str, out: Path, n_panels: int = 4, top_k: int = 20) -> None:
    if not graphs:
        return
    tickers = graphs[0]["tickers"]
    if target not in tickers:
        return

    # Pick n_panels days roughly evenly spaced across the test window.
    n = len(graphs)
    if n_panels >= n:
        idxs = list(range(n))
    else:
        idxs = [int(i * (n - 1) / (n_panels - 1)) for i in range(n_panels)]

    cols = min(len(idxs), 2)
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows),
                              squeeze=False)
    for k, gi in enumerate(idxs):
        ax = axes[k // cols][k % cols]
        g = graphs[gi]
        _draw_radial_graph(ax, g["adj"], tickers, target, top_k=top_k,
                            title=f"{g['date']}   ε={g['eps']:.3f}")

    # Hide any extra empty subplots.
    for k in range(len(idxs), rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    fig.suptitle(f"Learned graph around {target} — evolution across test period",
                  fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# 8. full universe, spring layout of the average learned graph
# ----------------------------------------------------------------------------
def plot_full_graph(graphs, target: str, out: Path,
                    edge_weight_pct: float = 0.5, label_top_pct: float = 0.4) -> None:
    """Layout the entire universe at once using a force-directed (spring)
    embedding of the mean adjacency. Reveals the clustering structure the
    GraphConstructor has learned: what other names AAPL gravitates toward,
    which stocks form tight communities, which sit on the periphery.

    edge_weight_pct: only draw edges in the top fraction of mean weights
                     (avoid hairball with 98 nodes).
    label_top_pct:   label this fraction of nodes (those with the highest
                     mean degree); others are dots.
    """
    if not graphs:
        return
    try:
        import networkx as nx
    except ImportError:
        print("plot_full_graph: networkx not installed — skipping")
        return

    tickers = graphs[0]["tickers"]
    n = len(tickers)
    ti = tickers.index(target) if target in tickers else -1

    # Mean adjacency across the test period, symmetrise (max) so the layout
    # treats reciprocal-ish neighbourships consistently.
    # KNOW averaging over time blurs any regime change in the learned graph
    mean_adj = np.mean([g["adj"] for g in graphs], axis=0)
    mean_adj_sym = np.maximum(mean_adj, mean_adj.T)
    np.fill_diagonal(mean_adj_sym, 0.0)

    flat = mean_adj_sym[mean_adj_sym > 0]
    if flat.size == 0:
        print("plot_full_graph: empty graph — skipping")
        return

    # Build the FULL graph (all non-zero edges) for layout, this captures the
    # true connectivity of the universe. Edges below `cutoff` are still
    # *drawn* less prominently or hidden, but they participate in the spring
    # forces so isolated-looking stocks aren't pushed to a meaningless ring.
    cutoff_draw = float(np.quantile(flat, 1.0 - edge_weight_pct))

    g = nx.Graph()
    for j in range(n):
        g.add_node(j)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(mean_adj_sym[i, j])
            if w > 0:                              # ALL nonzero edges for layout
                g.add_edge(i, j, weight=w)
    # Same graph, but only top-X% edges for *drawing*.
    g_draw = nx.Graph()
    for j in range(n):
        g_draw.add_node(j)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(mean_adj_sym[i, j])
            if w >= cutoff_draw:
                g_draw.add_edge(i, j, weight=w)

    # Layout strategy: lay out the largest connected component with
    # Kamada-Kawai (it makes prettier sparse-graph drawings than spring),
    # then place any isolates in a ring around it. Pure spring_layout on a
    # graph with disconnected components puts the isolates on a wide perimeter
    # circle which dominates the picture and crushes the interesting structure.
    # KNOW Kamada-Kawai needs scipy, the spring fallback is the no-scipy path
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    if components:
        lcc = list(components[0])
        sub = g.subgraph(lcc)
        if sub.number_of_edges() == 0:
            pos_lcc = nx.circular_layout(sub)
        else:
            # Prefer Kamada-Kawai (needs scipy), fall back to a well-tuned
            # spring layout that spreads the LCC properly (large k, many iter).
            try:
                pos_lcc = nx.kamada_kawai_layout(sub, weight="weight")
            except (ImportError, ModuleNotFoundError):
                pos_lcc = nx.spring_layout(
                    sub, seed=42, weight="weight",
                    k=3.0 / max(np.sqrt(len(lcc)), 1.0), iterations=500,
                )
    else:
        lcc = []
        pos_lcc = {}

    pos = dict(pos_lcc)
    if pos_lcc:
        lcc_arr = np.array(list(pos_lcc.values()))
        lcc_radius = np.linalg.norm(lcc_arr - lcc_arr.mean(axis=0), axis=1).max() or 1.0
    else:
        lcc_radius = 1.0

    isolates = [j for j in range(n) if j not in pos_lcc]
    if isolates:
        ring_r = lcc_radius * 1.7 + 0.4
        for k, j in enumerate(isolates):
            ang = 2 * np.pi * k / max(len(isolates), 1)
            pos[j] = np.array([ring_r * np.cos(ang), ring_r * np.sin(ang)])

    # Node degree (weighted) on the full graph drives both size and colour.
    weighted_deg = mean_adj_sym.sum(axis=1)
    deg = weighted_deg.copy()
    label_cutoff = np.quantile(deg, 1.0 - label_top_pct) if deg.size else 0.0

    fig, ax = plt.subplots(figsize=(13, 13))

    # Edges first (background), only the high-weight ones we kept for drawing.
    if g_draw.number_of_edges() > 0:
        ews = np.array([g_draw[u][v]["weight"] for u, v in g_draw.edges()])
        ew_max = float(ews.max()) if ews.size else 1.0
        for (u, v), w in zip(g_draw.edges(), ews):
            alpha = 0.15 + 0.55 * (w / ew_max)
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color="#666666", lw=0.3 + 2.5 * (w / ew_max),
                    alpha=alpha, zorder=1)

    # Non-target nodes.
    xs = np.array([pos[j][0] for j in range(n)])
    ys = np.array([pos[j][1] for j in range(n)])
    sizes = 30 + 220 * (weighted_deg / max(weighted_deg.max(), 1e-9))

    mask_other = np.array([j != ti for j in range(n)])
    sc = ax.scatter(xs[mask_other], ys[mask_other],
                    s=sizes[mask_other], c=weighted_deg[mask_other],
                    cmap="viridis", edgecolor="black", linewidth=0.4,
                    zorder=2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mean weighted degree (centrality in the learned graph)")

    # Target, make it pop.
    if ti >= 0:
        ax.scatter([pos[ti][0]], [pos[ti][1]], s=600, c="#d62728",
                    edgecolor="black", linewidth=1.5, zorder=4)
        ax.annotate(target, (pos[ti][0], pos[ti][1]),
                    fontsize=13, fontweight="bold", color="white",
                    ha="center", va="center", zorder=5)

    # Labels for the most-central non-target stocks.
    for j in range(n):
        if j == ti:
            continue
        if deg[j] >= label_cutoff:
            ax.annotate(tickers[j], (pos[j][0], pos[j][1]),
                        fontsize=8, ha="center", va="center",
                        xytext=(0, 9), textcoords="offset points",
                        color="#222222", zorder=3)

    ax.set_title(
        f"Learned graph — full S&P universe ({n} stocks, averaged over "
        f"{len(graphs)} test days)\n"
        f"target {target} highlighted · top {int(edge_weight_pct*100)}% edges shown · "
        f"top {int(label_top_pct*100)}% labelled",
        fontsize=11,
    )
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="path to a run directory (e.g. runs/local)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: <run>/figures)")
    args = p.parse_args()

    run = args.run_dir
    out = args.out or run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    target = _target_ticker(run)

    metrics = _safe_load_metrics(run)
    preds   = _load_predictions(run)
    graphs  = _load_graphs(run)

    produced = []
    if metrics is not None and len(metrics):
        plot_training_curves(metrics, out / "1_training_curves.png")
        produced.append("1_training_curves.png")
    if preds is not None and len(preds):
        if preds.attrs.get("schema") == "ranking":
            # v3.6 universe-wide ranking model, so use ranking/backtest diagnostics
            plot_ranking_overview(preds, _run_horizon(run), out / "2_ranking_overview.png")
            produced.append("2_ranking_overview.png")
        else:
            plot_predictions_overview(preds, target, out / "2_predictions_overview.png")
            plot_calibration(preds, out / "3_calibration.png")
            plot_strategy_pnl(preds, target, run, out / "6_strategy_pnl.png")
            produced += ["2_predictions_overview.png", "3_calibration.png", "6_strategy_pnl.png"]
    if graphs:
        plot_graph_summary(graphs, target, out / "4_graph_summary.png")
        plot_top_neighbours(graphs, target, out / "5_top_neighbours.png")
        plot_graph_network(graphs, target, out / "7_graph_network.png")
        plot_full_graph(graphs, target, out / "8_full_graph.png")
        produced += ["4_graph_summary.png", "5_top_neighbours.png",
                     "7_graph_network.png", "8_full_graph.png"]

    print(f"target = {target}")
    print(f"wrote {len(produced)} figures to {out}:")
    for f in produced:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
