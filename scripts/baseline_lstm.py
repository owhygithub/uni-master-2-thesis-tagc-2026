"""Proper single-stock LSTM baseline — the "is the graph worth it?" reference.

A real trained sequence model on the TARGET STOCK ONLY. Aim: a baseline that
actually exercises modern best practices, so the comparison to TAGC is fair.

Design — based on published peer-reviewed SOTA on daily stock direction:
  • Architecture (v3.3): 3-layer LSTM, 128 hidden, 0.25 dropout, layer-norm head.
  • Window: 60 days (the most common choice in the literature — Patel 2015,
    StockNet 2018, Adv-ALSTM 2019, and most LSTM tutorials use 60).
  • Features: target ticker's full feature panel (price, technicals,
    fundamentals, options, seasonality, engineered, optionally news).
  • Training: AdamW with warmup+cosine LR, gradient clipping, Huber loss with
    asymmetric direction penalty (same as TAGC), MC-Dropout inference.
  • Data: FULL 13-year history by default (last_n_days=0). Train on real
    examples, not 1.5 years.

Reference range (peer-reviewed SOTA on similar daily multi-stock benchmarks):
  StockNet      54.96 %  (ACL18, 88 stocks)
  Adv-ALSTM     57.2 %   (ACL18)
  DGDNN         54.6 %   (NASDAQ)
  MAN-SF/DTML   57-60 %  (NYSE/NASDAQ)
A well-tuned single-stock LSTM should land in the same 53-58 % range. Anything
above ~60 % on daily prediction without leakage is suspicious in the literature.

Usage:
    python scripts/baseline_lstm.py AAPL --fresh
    python scripts/baseline_lstm.py AAPL --fresh --last-n-days 0 --epochs 80
    python scripts/baseline_lstm.py AAPL --fresh --window 252        # 1-year window
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ───────────────────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────────────────
def _load_target_panel(cfg: Config, target: str):
    """Single-ticker [D, F] panel + [D] target series, identically split."""
    df = pd.read_parquet(cfg.stocks_parquet)
    macro = pd.read_parquet(cfg.macro_parquet)
    common = pd.DatetimeIndex(
        sorted(set(df.index.get_level_values("date")) & set(macro.index.get_level_values("date")))
    )
    if cfg.last_n_days is not None:
        common = common[-cfg.last_n_days:]
    sub = df.xs(target.upper(), level="ticker").reindex(common)
    feats = cfg.stock_feature_columns
    X = sub[feats].astype(np.float32).to_numpy()
    y_raw = sub["close_ret"].astype(np.float32).to_numpy()

    D = len(X)
    train_end = int(D * cfg.train_frac)
    val_end   = int(D * (cfg.train_frac + cfg.val_frac))

    # Per-ticker training-mean centering (same convention as TAGC).
    center = float(np.nanmean(y_raw[:train_end]))
    y = y_raw - center

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return X, y, common, train_end, val_end, center


def _windowed(X, y, indices, W):
    Xs = np.stack([X[t - W : t] for t in indices], axis=0)
    Ys = y[indices]
    return torch.from_numpy(Xs), torch.from_numpy(Ys)


# ───────────────────────────────────────────────────────────────────────────
# Model — LSTM
# ───────────────────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    """3-layer LSTM → LayerNorm → MLP head with MC-Dropout.

    v3.3 thesis-depth sizes (hidden=128, layers=3, dropout=0.25) — matches
    the StockNet (ACL-18) / Adv-ALSTM (ACL-18) reference size class. Earlier
    versions ran a toy-sized LSTM (64 hidden, 2 layers) — too small for a
    master-thesis-level "is the encoder worth it?" comparison.
    """

    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 3, dropout: float = 0.25,
                 head_hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.head(last).squeeze(-1)


# ───────────────────────────────────────────────────────────────────────────
# Loss (same shape as TAGC: Huber + direction-aware + hinge)
# ───────────────────────────────────────────────────────────────────────────
def _loss(pred, y, cfg):
    err = pred - y
    if cfg.loss_type == "huber":
        a = err.abs(); d = cfg.huber_delta
        base = torch.where(a < d, 0.5 * err.pow(2), d * (a - 0.5 * d))
    else:
        base = err.pow(2)
    if cfg.dir_loss_weight > 0:
        disagree = ((pred * y) < 0).float()
        base = base * (1.0 + cfg.dir_loss_weight * disagree)
        if cfg.dir_hinge_weight > 0:
            base = base + cfg.dir_hinge_weight * F.relu(-pred * y)
    return base.mean()


def _z_position(pred, std, cap=2.0):
    if std <= 1e-9:
        return max(-1.0, min(1.0, pred / (cap * 0.01)))
    z = pred / max(std, 1e-9)
    z = max(-cap, min(cap, z))
    return z / cap


# ───────────────────────────────────────────────────────────────────────────
# Train + eval
# ───────────────────────────────────────────────────────────────────────────
def train_and_eval(cfg, target, out_dir, *, epochs, window, hidden, n_layers,
                    dropout, lr, batch_size, mc_samples, weight_decay,
                    warmup_frac, patience,
                    pretrained=None, finetune_lr_mult=0.3):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "train.log")],
                        force=True)
    log = logging.getLogger("baseline.lstm")

    # Override the model's window via the local var (Config's window is for TAGC).
    cfg.window = window

    X, y, dates, train_end, val_end, center = _load_target_panel(cfg, target)
    W = window
    train_idx = np.arange(W, train_end)
    val_idx   = np.arange(train_end, val_end)
    test_idx  = np.arange(val_end, len(X))

    Xtr, Ytr = _windowed(X, y, train_idx, W)
    Xv,  Yv  = _windowed(X, y, val_idx,   W)
    Xte, Yte = _windowed(X, y, test_idx,  W)

    log.info("target=%s  features=%d  window=%d", target.upper(), X.shape[1], W)
    log.info("samples  train=%d  val=%d  test=%d", len(Xtr), len(Xv), len(Xte))
    log.info("split    train_end=%d  val_end=%d  total=%d", train_end, val_end, len(X))

    device = torch.device(cfg.device)
    model = LSTMBaseline(n_features=X.shape[1], hidden=hidden, n_layers=n_layers,
                         dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("LSTM baseline params: %d", n_params)

    # ── Fine-tune: load weights from another run, drop LR ─────────────────
    if pretrained is not None:
        pt = Path(pretrained)
        if not pt.exists():
            raise FileNotFoundError(f"--pretrained: {pt} does not exist")
        state = torch.load(pt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(state["model"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"--pretrained: checkpoint shape mismatch. Make sure --hidden / "
                f"--n-layers / --window / --no-news match the original training.\n"
                f"  underlying: {str(e).splitlines()[0]}"
            ) from None
        lr = lr * finetune_lr_mult
        log.info("loaded pretrained from %s  → fine-tune at lr=%.2e (×%.2f)",
                 pt, lr, finetune_lr_mult)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_batches = max(1, math.ceil(len(Xtr) / batch_size))
    total_steps = n_batches * epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)
    log.info("optimizer: %d steps total, %d warmup", total_steps, warmup_steps)

    best_val = float("inf"); no_improve = 0
    metrics_rows = [["epoch", "train_loss", "val_mse", "val_dir_acc",
                      "val_pred_mean", "val_pred_std", "best_val_so_far"]]
    best_path = out_dir / "best.pt"

    step = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        running = 0.0
        for i in range(n_batches):
            sl = perm[i*batch_size : (i+1)*batch_size]
            xb, yb = Xtr[sl].to(device), Ytr[sl].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = _loss(pred, yb, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step(); step += 1
            running += float(loss.item()) * len(xb)
        train_loss = running / max(len(Xtr), 1)

        # val
        model.eval()
        with torch.no_grad():
            vp = model(Xv.to(device)).cpu().numpy()
        v_err = vp - Yv.numpy()
        v_mse = float(np.mean(v_err ** 2))
        v_dacc = float(np.mean((vp + center > 0) == (Yv.numpy() + center > 0)))
        log.info("epoch %02d  train_loss=%.5f  val_mse=%.5f  dir_acc=%.3f  "
                 "pred_mean=%+.4f  pred_std=%.4f",
                 ep, train_loss, v_mse, v_dacc, float(vp.mean()), float(vp.std()))

        if v_mse < best_val - 1e-5:
            best_val = v_mse; no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "center": center},
                        best_path)
            log.info("  new best val_mse=%.5f → saved best.pt", v_mse)
        else:
            no_improve += 1
        metrics_rows.append([ep, train_loss, v_mse, v_dacc,
                              float(vp.mean()), float(vp.std()), best_val])

        if no_improve >= patience and ep >= max(20, epochs // 4):
            log.info("early stop at epoch %d (no improvement for %d epochs)", ep, no_improve)
            break

    # Test with MC-Dropout
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

    Xte_dev = Xte.to(device)
    samples = []
    for _ in range(mc_samples):
        with torch.no_grad():
            samples.append(model(Xte_dev).cpu().numpy())
    samples = np.stack(samples)
    pred_mean = samples.mean(0); pred_std = samples.std(0)
    actuals = Yte.numpy()

    test_mse = float(np.mean((pred_mean - actuals) ** 2))
    pred_uc = pred_mean + center
    actual_uc = actuals + center
    dir_acc = float(np.mean((pred_uc > 0) == (actual_uc > 0)))
    log.info("TEST  mse=%.5f  rmse=%.5f  dir_acc=%.3f", test_mse, math.sqrt(test_mse), dir_acc)

    # Write CSV in TAGC schema
    with open(out_dir / "predictions_test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(CSV_COLS)
        for k, t in enumerate(test_idx):
            d = str(dates[t].date())
            pr = float(pred_mean[k]) + center
            st = float(pred_std[k])
            ac = float(actuals[k]) + center
            sign_match = int((pr > 0) == (ac > 0))
            pos = _z_position(pr - center, st)
            w.writerow([d, target.upper(),
                         f"{pr:.6f}", f"{st:.6f}", f"{ac:.6f}",
                         int(pr > 0), int(ac > 0),
                         f"{pos:.6f}", sign_match])

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerows(metrics_rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "single_ticker_lstm",
        "target": target.upper(),
        "n_features": int(X.shape[1]),
        "window": W,
        "hidden": hidden, "n_layers": n_layers, "dropout": dropout,
        "params": n_params,
        "epochs_trained": ep + 1,
        "total_optimizer_steps": step,
        "best_val_mse": best_val,
        "test_mse": test_mse, "test_rmse": math.sqrt(test_mse),
        "test_dir_acc": dir_acc,
        "center_value": center,
    }, indent=2))

    # Dashboard
    log.info("")
    log.info("═" * 60)
    log.info("LSTM BASELINE  target=%s  params=%d  steps=%d", target.upper(), n_params, step)
    log.info("  TEST  MSE=%.5f  RMSE=%.5f  dir_acc=%.3f", test_mse, math.sqrt(test_mse), dir_acc)
    log.info("═" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs",  type=int, default=80)
    ap.add_argument("--last-n-days", type=int, default=0,
                    help="0 = full history (default for baseline). Pass 504 for short window.")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)        # v3.3: was 64
    ap.add_argument("--n-layers", type=int, default=3)        # v3.3: was 2
    ap.add_argument("--dropout", type=float, default=0.25)    # v3.3: was 0.20
    ap.add_argument("--lr",     type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--fresh",  action="store_true")
    ap.add_argument("--pretrained", default=None,
                    help="path to an existing best.pt — load weights then "
                         "continue training (fine-tune) with reduced LR")
    ap.add_argument("--finetune-lr-mult", type=float, default=0.3,
                    help="multiplier on --lr when --pretrained is given (default 0.3)")
    args = ap.parse_args()

    cfg = Config(use_news=not args.no_news)
    cfg.last_n_days = None if args.last_n_days == 0 else args.last_n_days
    cfg.target_ticker = args.target.upper()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(args.out_dir or REPO / "runs" / f"baseline_lstm_{args.target.lower()}")
    if args.fresh and out.exists():
        shutil.rmtree(out)

    train_and_eval(cfg, args.target, out,
                    epochs=args.epochs,
                    window=args.window,
                    hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
                    lr=args.lr, batch_size=args.batch_size,
                    mc_samples=args.mc_samples,
                    weight_decay=args.weight_decay,
                    warmup_frac=args.warmup_frac,
                    patience=args.patience,
                    pretrained=args.pretrained,
                    finetune_lr_mult=args.finetune_lr_mult)


if __name__ == "__main__":
    main()
