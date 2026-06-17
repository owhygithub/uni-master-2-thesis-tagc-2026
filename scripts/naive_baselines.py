"""Naive non-learned baselines for return prediction.

These are the "floor" any real model has to beat to be non-vacuous:

  constant       predict 0 every day                       (no model, no params)
  climatology    predict the training-period mean return    (1 param per ticker)
  persistence    predict yesterday's return as today's      (no params)
  ar1            linear regression of ret[t] on ret[t-1]    (2 params per ticker)
  ridge_lag5     ridge on the prev-5-day returns            (5 params per ticker)

Each baseline writes a `predictions_test.csv` in the same schema as TAGC's, so
the comparison tool downstream can treat them all uniformly.

Usage:
    python scripts/naive_baselines.py AAPL                       # writes runs/baseline_<kind>_aapl/
    python scripts/naive_baselines.py AAPL --kind persistence    # one specific
    python scripts/naive_baselines.py AAPL --last-n-days 504     # match TAGC's window
    python scripts/naive_baselines.py AAPL --no-news             # use the no-news parquet
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.config import Config


CSV_COLS = [
    "date", "ticker",
    "predicted_return", "predicted_std",
    "true_return",
    "predicted_direction", "true_direction",
    "position_size", "direction_correct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_target(cfg: Config, target: str):
    """Return (dates, ticker_returns_series) for the target ticker in cfg's
    canonical window, mirroring how model.data filters the panel."""
    # KNOW we intersect on macro dates too so the window matches what TAGC sees
    df = pd.read_parquet(cfg.stocks_parquet)
    if df.index.names != ["date", "ticker"]:
        raise ValueError(f"unexpected index: {df.index.names}")
    macro = pd.read_parquet(cfg.macro_parquet)
    common = pd.DatetimeIndex(
        sorted(set(df.index.get_level_values("date")) & set(macro.index.get_level_values("date")))
    )
    if cfg.last_n_days is not None:
        common = common[-cfg.last_n_days:]
    sub = df.xs(target.upper(), level="ticker").reindex(common)
    if "close_ret" not in sub.columns:
        raise ValueError("close_ret missing from stocks parquet")
    return common, sub["close_ret"].astype(float).to_numpy()


def _splits(D: int, train_frac: float, val_frac: float):
    return (int(D * train_frac),
            int(D * (train_frac + val_frac)),
            D)


def _z_position(pred: float, std: float, cap: float = 2.0) -> float:
    if std <= 1e-9:
        return max(-1.0, min(1.0, pred / (cap * 0.01)))
    z = pred / max(std, 1e-9)
    z = max(-cap, min(cap, z))
    return z / cap


def _write_csv(out_dir: Path, dates: pd.DatetimeIndex, test_t: np.ndarray,
                preds: np.ndarray, stds: np.ndarray, actuals: np.ndarray,
                target: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p_test = out_dir / "predictions_test.csv"
    with open(p_test, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for k, t in enumerate(test_t):
            d = str(dates[t].date())
            pr = float(preds[k])
            st = float(stds[k])
            ac = float(actuals[k])
            sign_match = int((pr > 0) == (ac > 0))
            pos = _z_position(pr, st)
            w.writerow([d, target.upper(),
                         f"{pr:.6f}", f"{st:.6f}", f"{ac:.6f}",
                         int(pr > 0), int(ac > 0),
                         f"{pos:.6f}", sign_match])
    return p_test


def _metrics(preds, actuals):
    n = len(preds)
    mse = float(np.mean((preds - actuals) ** 2))
    rmse = math.sqrt(mse)
    dir_acc = float(np.mean((preds > 0) == (actuals > 0)))
    return {"n": n, "mse": mse, "rmse": rmse, "dir_acc": dir_acc}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_constant(cfg, target, value=0.0):
    """Always predict `value`. Sanity-floor, beats nothing."""
    dates, returns = _load_target(cfg, target)
    D = len(returns)
    train_end, val_end, total = _splits(D, cfg.train_frac, cfg.val_frac)
    test_t = np.arange(val_end, total)
    preds   = np.full(len(test_t), float(value), dtype=np.float32)
    actuals = returns[test_t]
    stds    = np.zeros_like(preds)
    return preds, stds, actuals, test_t, dates


def baseline_climatology(cfg, target):
    """Predict the per-ticker TRAINING-split mean return for every test day."""
    dates, returns = _load_target(cfg, target)
    D = len(returns)
    train_end, val_end, total = _splits(D, cfg.train_frac, cfg.val_frac)
    mu = float(np.nanmean(returns[:train_end]))
    test_t = np.arange(val_end, total)
    preds   = np.full(len(test_t), mu, dtype=np.float32)
    actuals = returns[test_t]
    stds    = np.zeros_like(preds)
    return preds, stds, actuals, test_t, dates


def baseline_persistence(cfg, target):
    """Predict yesterday's return as today's return."""
    dates, returns = _load_target(cfg, target)
    D = len(returns)
    train_end, val_end, total = _splits(D, cfg.train_frac, cfg.val_frac)
    test_t = np.arange(val_end, total)
    # KNOW test_t starts at val_end > 0 so returns[test_t - 1] never underflows
    preds   = returns[test_t - 1].astype(np.float32)
    actuals = returns[test_t]
    stds    = np.zeros_like(preds)
    return preds, stds, actuals, test_t, dates


def baseline_ar1(cfg, target):
    """Fit a linear AR(1): ret[t] = a + b * ret[t-1] on TRAIN, predict on TEST."""
    dates, returns = _load_target(cfg, target)
    D = len(returns)
    train_end, val_end, total = _splits(D, cfg.train_frac, cfg.val_frac)

    # OLS on train pairs (ret[t-1], ret[t])
    x_tr = returns[:train_end - 1]
    y_tr = returns[1:train_end]
    A = np.column_stack([np.ones_like(x_tr), x_tr])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    test_t = np.arange(val_end, total)
    preds   = (a + b * returns[test_t - 1]).astype(np.float32)
    actuals = returns[test_t]
    # Std: residual std on the training fit (homoscedastic AR(1)).
    # KNOW this assumes constant variance, fine for a baseline but not realistic
    resid = y_tr - (a + b * x_tr)
    res_std = float(np.std(resid))
    stds = np.full_like(preds, res_std)
    return preds, stds, actuals, test_t, dates


def baseline_ridge_lag(cfg, target, K=5, alpha=1.0):
    """Ridge regression of ret[t] on (ret[t-1], ret[t-2], ..., ret[t-K])."""
    from sklearn.linear_model import Ridge
    dates, returns = _load_target(cfg, target)
    D = len(returns)
    train_end, val_end, total = _splits(D, cfg.train_frac, cfg.val_frac)

    def lag_matrix(end_idx):
        # rows for t in [K, end_idx); columns are returns[t-1..t-K]
        rows = []
        ys = []
        for t in range(K, end_idx):
            rows.append(returns[t-K:t][::-1])
            ys.append(returns[t])
        return np.array(rows), np.array(ys)

    X_tr, y_tr = lag_matrix(train_end)
    model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    # Predict on test
    test_t = np.arange(val_end, total)
    X_te = np.array([returns[t-K:t][::-1] for t in test_t])
    preds = model.predict(X_te).astype(np.float32)
    actuals = returns[test_t]
    resid = y_tr - model.predict(X_tr)
    res_std = float(np.std(resid))
    stds = np.full_like(preds, res_std)
    return preds, stds, actuals, test_t, dates


BASELINES = {
    "constant":     lambda cfg, t: baseline_constant(cfg, t, value=0.0),
    "climatology":  baseline_climatology,
    "persistence":  baseline_persistence,
    "ar1":          baseline_ar1,
    "ridge_lag5":   lambda cfg, t: baseline_ridge_lag(cfg, t, K=5, alpha=1.0),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="ticker, e.g. AAPL")
    ap.add_argument("--kind", default=None, choices=list(BASELINES.keys()) + ["all"],
                    help="which baseline to run; default = 'all'")
    ap.add_argument("--last-n-days", type=int, default=504)
    ap.add_argument("--no-news", action="store_true",
                    help="use data/stocks_dataset.parquet (no-news variant)")
    ap.add_argument("--out-root", default=None,
                    help="output dir root (default: runs/)")
    args = ap.parse_args()

    cfg = Config(use_news=not args.no_news)
    cfg.last_n_days = args.last_n_days
    cfg.target_ticker = args.target.upper()
    out_root = Path(args.out_root or REPO / "runs")

    kinds = [args.kind] if args.kind and args.kind != "all" else list(BASELINES.keys())

    print(f"target = {args.target.upper()}    use_news = {not args.no_news}    last_n_days = {args.last_n_days}")
    print(f"{'baseline':14s}  {'n':>5s}  {'MSE':>9s}  {'RMSE':>8s}  {'dir_acc':>9s}")
    print("─" * 56)
    for kind in kinds:
        preds, stds, actuals, test_t, dates = BASELINES[kind](cfg, args.target)
        m = _metrics(preds, actuals)
        out_dir = out_root / f"baseline_{kind}_{args.target.lower()}"
        _write_csv(out_dir, dates, test_t, preds, stds, actuals, args.target)
        # Save a tiny summary.json
        (out_dir / "summary.json").write_text(json.dumps({
            "baseline":   kind,
            "target":     args.target.upper(),
            **m,
        }, indent=2))
        print(f"{kind:14s}  {m['n']:>5d}  {m['mse']:9.5f}  {m['rmse']:8.5f}  {m['dir_acc']:8.4f}")

    print(f"\n→ predictions written under {out_root}/baseline_*_{args.target.lower()}/")


if __name__ == "__main__":
    main()
