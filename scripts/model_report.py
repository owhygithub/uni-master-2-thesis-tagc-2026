"""Question-driven model report — one PNG per question about the trained model.

## Overview
A clean, paper-ready set of figures where **each figure answers exactly one
question** about the trained TAGC run (rather than a dense dashboard). Reads only
the artefacts a run already writes — `metrics.csv` (per-epoch) and
`predictions_test.csv` (universe-wide, one row per date/ticker) — so the numbers
are computed from scratch here and don't depend on the older visualize.py logic.

Figures written to `<run>/figures/`:
    q1_is_it_learning.png          val Rank-IC per epoch (+ the selected best)
    q2_did_predictions_collapse.png  val pred-std per epoch (collapse detector)
    q3_are_both_objectives_training.png  train total loss vs ListMLE rank loss
    q4_do_higher_ranks_earn_more.png  mean realized return by predicted decile
    q5_pred_vs_realized_rank.png     pred_rank vs true_rank density (+ Rank-IC)
    q6_is_the_edge_stable.png        per-day Rank-IC across the test period
    q7_is_the_strategy_profitable.png  cumulative long-short return (+ Sharpe)
    q8_is_direction_right.png        direction accuracy overall + by decile

Usage:
    python scripts/model_report.py runs_final/run30
    python scripts/model_report.py runs_final/run30 --out somewhere/figs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# house style — readable, consistent, paper-friendly
ACCENT = "#9467bd"
BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
GREY = "#888888"


def _question_title(ax, question: str, answer: str = "") -> None:
    """Put the QUESTION as the headline and a one-line ANSWER as the subtitle."""
    ax.set_title(question, fontsize=13, fontweight="600", loc="left", pad=20)
    if answer:
        ax.text(0.0, 1.03, answer, transform=ax.transAxes, fontsize=8.5,
                color=GREY, ha="left", va="bottom")


# ──────────────────────────────────────────────────────────────────────────
# loaders
# ──────────────────────────────────────────────────────────────────────────
def _load_metrics(run: Path) -> Optional[pd.DataFrame]:
    p = run / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if len(df) else None


def _load_preds(run: Path) -> Optional[pd.DataFrame]:
    p = run / "predictions_test.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df if len(df) else None


def _horizon(run: Path, default: int = 30) -> int:
    cfg = run / "config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text()).get("target_horizon", default))
        except Exception:
            pass
    return default


# ──────────────────────────────────────────────────────────────────────────
# Q1 — is the model learning?
# ──────────────────────────────────────────────────────────────────────────
def q1_is_it_learning(m: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = m["epoch"]
    if "val_rank_ic" in m:
        ax.plot(ep, m["val_rank_ic"], "o-", color=ACCENT, lw=2, ms=4,
                label="val Rank-IC (Spearman)")
    if "best_rank_ic_so_far" in m:
        ax.plot(ep, m["best_rank_ic_so_far"], "--", color=GREEN, alpha=0.85,
                label="best so far (model selected on this)")
    if "val_ic" in m:
        ax.plot(ep, m["val_ic"], "o-", color=BLUE, ms=3, alpha=0.5,
                label="val IC (Pearson)")
    ax.axhline(0.0, color=GREY, ls="--", lw=0.8, label="0 = no skill")
    best = float(m["val_rank_ic"].max()) if "val_rank_ic" in m else float("nan")
    _question_title(ax, "Q1 · Is the model actually learning?",
                    f"best val Rank-IC = {best:+.4f}  —  above 0 = real cross-sectional skill")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-sectional correlation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q2 — did the predictions collapse?
# ──────────────────────────────────────────────────────────────────────────
def q2_did_predictions_collapse(m: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = m["epoch"]
    if "val_pred_std" in m:
        ax.plot(ep, m["val_pred_std"], "o-", color=RED, lw=2, ms=4,
                label="val prediction std")
    ax.axhline(0.0, color=GREY, ls="--", lw=0.8)
    last = float(m["val_pred_std"].iloc[-1]) if "val_pred_std" in m else float("nan")
    verdict = "healthy spread" if last > 0.05 else "WARNING: near-constant output (collapse)"
    _question_title(ax, "Q2 · Did the predictions collapse to a constant?",
                    f"final pred-std = {last:.4f}  ·  {verdict}")
    if "val_pred_mean" in m:
        ax2 = ax.twinx()
        ax2.plot(ep, m["val_pred_mean"], "^-", color=BLUE, alpha=0.5, ms=3,
                 label="val pred mean")
        ax2.set_ylabel("prediction mean", color=BLUE)
        ax2.tick_params(axis="y", colors=BLUE)
    ax.set_xlabel("epoch"); ax.set_ylabel("prediction std (scaled units)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q3 — are BOTH objectives (rank + magnitude) training?
# ──────────────────────────────────────────────────────────────────────────
def q3_are_both_objectives_training(m: pd.DataFrame, out: Path) -> None:
    # the two REAL terms of rank_mag: ListMLE (ranking) + Huber (magnitude).
    # train_rank_loss = ListMLE; train_mag_loss = the genuine Huber term.
    # (older runs predate train_mag_loss — fall back & say so.)
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = m["epoch"]
    ax.plot(ep, m["train_rank_loss"], "o-", color=ACCENT, lw=2, ms=4,
            label="train ListMLE (ranking, weight 0.6)")
    ax.set_xlabel("epoch"); ax.set_ylabel("ListMLE ranking loss", color=ACCENT)
    ax.tick_params(axis="y", colors=ACCENT)
    have_huber = "train_mag_loss" in m and m["train_mag_loss"].notna().any()
    if have_huber:
        ax2 = ax.twinx()
        ax2.plot(ep, m["train_mag_loss"], "s-", color=BLUE, alpha=0.75, ms=3,
                 label="train Huber (magnitude, weight 0.4)")
        ax2.set_ylabel("Huber magnitude loss", color=BLUE)
        ax2.tick_params(axis="y", colors=BLUE)
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
        sub = ("rank_mag = 0.6·ListMLE + 0.4·Huber — both should drift down; a flat Huber "
               "with pred-std→0 means the magnitude target has no signal (see Q2)")
    else:
        ax.legend(fontsize=8, loc="upper right")
        sub = "this run predates train_mag_loss logging — only the ListMLE term is recorded"
    _question_title(ax, "Q3 · Are both loss terms (rank + magnitude) decreasing?", sub)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q4 — do higher predicted ranks actually earn higher returns?
# ──────────────────────────────────────────────────────────────────────────
def q4_do_higher_ranks_earn_more(p: pd.DataFrame, out: Path, n_dec: int = 10) -> None:
    # decile each day by pred_score, then average the realized forward return.
    d = p.dropna(subset=["pred_score", "true_fwd_return"]).copy()
    d["dec"] = d.groupby("date")["pred_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_dec, labels=False, duplicates="drop")
        if s.notna().sum() >= n_dec else np.nan)
    d = d.dropna(subset=["dec"])
    means = d.groupby("dec")["true_fwd_return"].mean() * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [RED if i < n_dec / 2 else GREEN for i in means.index]
    ax.bar(means.index, means.values, color=colors, alpha=0.85)
    ax.axhline(0.0, color=GREY, lw=0.8)
    # monotonicity: Spearman of decile index vs mean return
    mono = pd.Series(means.values).corr(pd.Series(means.index), method="spearman")
    spread = float(means.iloc[-1] - means.iloc[0])
    _question_title(ax, "Q4 · Do higher predicted ranks earn higher real returns?",
                    f"top−bottom decile spread = {spread:+.2f}%  ·  monotonicity (Spearman) = {mono:+.2f}")
    ax.set_xlabel(f"predicted-score decile (0 = model's worst, {n_dec-1} = best)")
    ax.set_ylabel("mean realized forward return (%)")
    ax.set_xticks(range(n_dec)); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q5 — do predicted ranks line up with realized ranks?
# ──────────────────────────────────────────────────────────────────────────
def q5_pred_vs_realized_rank(p: pd.DataFrame, out: Path) -> None:
    d = p.dropna(subset=["pred_rank", "true_rank"])
    fig, ax = plt.subplots(figsize=(7.2, 6))
    hb = ax.hexbin(d["pred_rank"], d["true_rank"], gridsize=40, cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count (date×stock)")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ric = d["pred_rank"].corr(d["true_rank"], method="spearman")
    _question_title(ax, "Q5 · Do predicted ranks match realized ranks?",
                    f"overall Rank-IC = {ric:+.4f}  —  mass on the diagonal = correct ordering")
    ax.set_xlabel("predicted percentile rank"); ax.set_ylabel("realized percentile rank")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q6 — is the edge stable through the test period (or regime-dependent)?
# ──────────────────────────────────────────────────────────────────────────
def q6_is_the_edge_stable(p: pd.DataFrame, out: Path) -> None:
    # per-day cross-sectional Spearman(pred_score, true_fwd_return)
    daily = (p.dropna(subset=["pred_score", "true_fwd_return"])
               .groupby("date")
               .apply(lambda g: g["pred_score"].corr(g["true_fwd_return"], method="spearman"))
               .dropna())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(daily.index, daily.values, color=ACCENT, alpha=0.35, lw=0.8, label="daily Rank-IC")
    roll = daily.rolling(21, min_periods=5).mean()
    ax.plot(roll.index, roll.values, color=GREEN, lw=2, label="21-day rolling mean")
    ax.axhline(0.0, color=GREY, ls="--", lw=0.8)
    ax.axhline(daily.mean(), color=RED, ls=":", lw=1.2, label=f"period mean {daily.mean():+.3f}")
    _question_title(ax, "Q6 · Is the ranking edge stable over time?",
                    f"mean daily Rank-IC = {daily.mean():+.4f} ± {daily.std():.4f}  ·  0-crossings = regime-dependent edge")
    ax.set_xlabel("test date"); ax.set_ylabel("daily cross-sectional Rank-IC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q7 — would a long-short book on these ranks have made money?
# ──────────────────────────────────────────────────────────────────────────
def q7_is_the_strategy_profitable(p: pd.DataFrame, out: Path, horizon: int) -> None:
    # non-overlapping rebalances spaced by the horizon (matches backtest headline)
    d = p.dropna(subset=["pred_score", "true_fwd_return"])
    dates = np.array(sorted(d["date"].unique()))
    reb = dates[::max(horizon, 1)]
    rets = []
    for dt in reb:
        g = d[d["date"] == dt]
        if len(g) < 10:
            continue
        cut = max(int(len(g) * 0.1), 1)
        order = g.sort_values("pred_score")
        short = order.head(cut)["true_fwd_return"].mean()
        long_ = order.tail(cut)["true_fwd_return"].mean()
        rets.append(long_ - short)
    rets = np.array(rets, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    if rets.size:
        cum = np.cumsum(rets) * 100.0
        ax.plot(reb[:len(rets)], cum, color=GREEN, lw=2)
        ax.fill_between(reb[:len(rets)], 0, cum, color=GREEN, alpha=0.12)
        sharpe = (rets.mean() / rets.std() * math.sqrt(252 / horizon)) if rets.std() > 0 else float("nan")
        hit = float(np.mean(rets > 0))
        ans = (f"annualised Sharpe = {sharpe:+.2f}  ·  hit-rate = {hit:.0%}  ·  "
               f"{len(rets)} non-overlapping {horizon}-day rebalances  (gross, before costs)")
    else:
        ans = "not enough rebalances to evaluate"
    ax.axhline(0.0, color=GREY, ls="--", lw=0.8)
    _question_title(ax, "Q7 · Would a long-short book on these ranks make money?", ans)
    ax.set_xlabel("test date"); ax.set_ylabel("cumulative long−short return (%)")
    ax.grid(alpha=0.3); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Q8 — does it get the direction right?
# ──────────────────────────────────────────────────────────────────────────
def q8_is_direction_right(p: pd.DataFrame, out: Path, n_dec: int = 10) -> None:
    if "direction_correct" not in p:
        return
    d = p.dropna(subset=["direction_correct", "pred_score"]).copy()
    overall = float(d["direction_correct"].mean())
    d["dec"] = d.groupby("date")["pred_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_dec, labels=False, duplicates="drop")
        if s.notna().sum() >= n_dec else np.nan)
    by_dec = d.dropna(subset=["dec"]).groupby("dec")["direction_correct"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_dec.index, by_dec.values, color=BLUE, alpha=0.85)
    ax.axhline(0.5, color=GREY, ls="--", lw=1.0, label="coin flip (0.5)")
    ax.axhline(overall, color=RED, ls=":", lw=1.2, label=f"overall {overall:.3f}")
    _question_title(ax, "Q8 · Does the model get the direction right?",
                    f"overall direction accuracy = {overall:.1%}  —  confident deciles should beat 0.5 by more")
    ax.set_xlabel(f"predicted-score decile (0 = worst, {n_dec-1} = best)")
    ax.set_ylabel("fraction with correct direction")
    ax.set_ylim(0, 1); ax.set_xticks(range(n_dec))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────
def generate(run, out=None) -> List[str]:
    """Build every question figure for a run dir. Importable so train.py can call
    it straight after training. Returns the filenames written."""
    run = Path(run)
    out = Path(out) if out is not None else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    horizon = _horizon(run)
    m = _load_metrics(run)
    p = _load_preds(run)
    produced: List[str] = []

    def _try(name, fn):
        try:
            fn(out / name); produced.append(name)
        except Exception as e:  # one bad figure shouldn't kill the rest
            print(f"  [skip] {name}: {e}")

    if m is not None:
        _try("q1_is_it_learning.png", lambda o: q1_is_it_learning(m, o))
        _try("q2_did_predictions_collapse.png", lambda o: q2_did_predictions_collapse(m, o))
        _try("q3_are_both_objectives_training.png", lambda o: q3_are_both_objectives_training(m, o))
    if p is not None:
        _try("q4_do_higher_ranks_earn_more.png", lambda o: q4_do_higher_ranks_earn_more(p, o))
        _try("q5_pred_vs_realized_rank.png", lambda o: q5_pred_vs_realized_rank(p, o))
        _try("q6_is_the_edge_stable.png", lambda o: q6_is_the_edge_stable(p, o))
        _try("q7_is_the_strategy_profitable.png", lambda o: q7_is_the_strategy_profitable(p, o, horizon))
        _try("q8_is_direction_right.png", lambda o: q8_is_direction_right(p, o))
    return produced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output dir (default <run>/figures)")
    args = ap.parse_args()
    produced = generate(args.run_dir, args.out)
    where = args.out or (Path(args.run_dir) / "figures")
    print(f"wrote {len(produced)} question-figures to {where}:")
    for f in produced:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
