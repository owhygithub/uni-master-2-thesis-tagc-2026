"""Long-short backtest + full-history cross-sectional evaluation.

Consumes the UNIVERSE-WIDE predictions CSV written by `train.evaluate()`:
    date, ticker, pred_score, pred_rank, pred_direction, true_fwd_return
(`true_fwd_return` is the RAW H-day forward return for that (date, ticker)).

Computes, over the FULL test period:
  - daily cross-sectional Rank-IC (Spearman(pred_score, true_fwd_return)), with
    mean / std / ICIR;
  - a daily long-short portfolio: long the top `decile`, short the bottom
    `decile` by pred_score (equal-weight), so
        ls_ret[d] = mean(true_fwd_return[long]) - mean(true_fwd_return[short]);
  - because each `true_fwd_return` spans H days, consecutive daily portfolios
    OVERLAP and are autocorrelated. We report BOTH an overlapping series (with a
    sqrt(252/H) annualisation correction) and a NON-OVERLAPPING series
    (`dates[::H]`, the rigorous headline); the cumulative PnL uses the
    non-overlapping series to avoid double-counting.

Usage:
    python scripts/backtest.py --predictions runs/x/predictions_test.csv --horizon 5
or imported:  from scripts.backtest import backtest
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _sharpe(series: np.ndarray, horizon: int, trading_days: int = 252) -> float:
    """Annualised Sharpe of an H-day-return series. sqrt(trading_days/H) annualises
    an H-day-period IR (each period is one H-day holding)."""
    s = series[np.isfinite(series)]
    if s.size < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(trading_days / horizon))


def backtest(predictions_csv,
             horizon: Optional[int] = None,
             decile: float = 0.1,
             trading_days: int = 252,
             out_dir=None) -> dict:
    predictions_csv = Path(predictions_csv)
    out_dir = Path(out_dir) if out_dir is not None else predictions_csv.parent

    df = pd.read_csv(predictions_csv, parse_dates=["date"]).sort_values("date")
    # horizon: arg > sibling config.json > infer 5
    if horizon is None:
        cfgp = predictions_csv.parent / "config.json"
        if cfgp.exists():
            try:
                horizon = int(json.loads(cfgp.read_text()).get("target_horizon", 5))
            except Exception:
                horizon = 5
        else:
            horizon = 5

    daily_rank_ic, daily_ls, daily_dir_acc, dates, ks = [], [], [], [], []
    for dt, g in df.groupby("date"):
        s = g["pred_score"].to_numpy(); r = g["true_fwd_return"].to_numpy()
        ok = np.isfinite(s) & np.isfinite(r)
        s, r = s[ok], r[ok]; K = s.size
        if K < 10 or s.std() == 0:
            continue
        dates.append(pd.Timestamp(dt)); ks.append(K)
        daily_rank_ic.append(_spearman(s, r))
        # long-short legs
        # KNOW nleg is at least 1, so tiny universes still form a portfolio
        nleg = max(1, int(round(decile * K)))
        order = np.argsort(s)                      # ascending
        short = r[order[:nleg]].mean()
        long_ = r[order[-nleg:]].mean()
        daily_ls.append(long_ - short)
        # direction accuracy: sign(score) vs sign(raw fwd return)
        daily_dir_acc.append(float(np.mean((s > 0) == (r > 0))))

    if not dates:
        raise ValueError("no valid test days in predictions CSV")

    dates = np.array(dates)
    rank_ic = np.array(daily_rank_ic, dtype=float)
    ls_ov   = np.array(daily_ls, dtype=float)          # overlapping (one per trading day)

    # non-overlapping series: every H-th trading day (holding period ends before next forms)
    # KNOW this is the honest headline series, the overlapping one is autocorrelated
    idx_no = np.arange(0, len(dates), max(1, horizon))
    ls_no  = ls_ov[idx_no]
    dates_no = dates[idx_no]

    # cumulative PnL (compounded + additive) on the non-overlapping series
    cum_comp = np.cumprod(1.0 + ls_no) - 1.0
    cum_add  = np.cumsum(ls_no)

    ic_mean = float(np.nanmean(rank_ic)); ic_std = float(np.nanstd(rank_ic, ddof=1))
    summary = {
        "horizon": int(horizon), "decile": decile,
        "n_days": int(len(dates)), "n_periods_nonoverlap": int(len(ls_no)),
        "avg_K": float(np.mean(ks)),
        "rank_ic_mean": round(ic_mean, 5),
        "rank_ic_std": round(ic_std, 5),
        "rank_icir": round(ic_mean / ic_std, 4) if ic_std > 0 else float("nan"),
        "dir_acc": round(float(np.nanmean(daily_dir_acc)), 4),
        "ls_ret_mean_period": round(float(np.nanmean(ls_no)), 5),     # per H-day period
        "ls_ret_std_period": round(float(np.nanstd(ls_no, ddof=1)), 5),
        "ls_sharpe_overlap": round(_sharpe(ls_ov, horizon, trading_days), 3),
        "ls_sharpe_nonoverlap": round(_sharpe(ls_no, horizon, trading_days), 3),  # headline IR
        "ls_hit_rate": round(float(np.mean(ls_no > 0)), 4),
        "ls_hit_rate_overlap": round(float(np.mean(ls_ov > 0)), 4),
        "cum_return_compounded": round(float(cum_comp[-1]), 4),
        "cum_return_additive": round(float(cum_add[-1]), 4),
    }

    # write outputs
    pd.DataFrame({"date": dates, "k": ks, "rank_ic": rank_ic,
                  "ls_ret_overlap": ls_ov}).to_csv(out_dir / "backtest_daily.csv", index=False)
    pd.DataFrame({"date": dates_no, "ls_ret": ls_no,
                  "cum_pnl_compounded": cum_comp,
                  "cum_pnl_additive": cum_add}).to_csv(out_dir / "backtest_nonoverlap.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "backtest_summary.csv", index=False)

    # optional PnL plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        ax[0].plot(dates_no, cum_comp * 100, color="#2ca02c")
        ax[0].axhline(0, color="gray", lw=0.6)
        ax[0].set_title(f"Long-short cumulative return  ({horizon}d, decile {decile:.0%})\n"
                        f"Sharpe {summary['ls_sharpe_nonoverlap']:.2f} · hit {summary['ls_hit_rate']:.2f}")
        ax[0].set_ylabel("cumulative %"); ax[0].grid(alpha=.3)
        ax[1].plot(dates, np.cumsum(np.nan_to_num(rank_ic)), color="#1f77b4")
        ax[1].set_title(f"Cumulative daily Rank-IC  (mean {ic_mean:+.4f}, ICIR {summary['rank_icir']})")
        ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / "backtest_pnl.png", dpi=130); plt.close(fig)
    except Exception:
        pass

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--decile", type=float, default=0.1)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    s = backtest(a.predictions, horizon=a.horizon, decile=a.decile, out_dir=a.out_dir)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
