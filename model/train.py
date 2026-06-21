"""training loop + eval for TAGC.

default loss is rank_mag (the v5 one), two pieces, BOTH over the whole cross-section
each day, on the z-scored target:

    pred[N]   = head(embeds)                          # one score per stock
    rank_loss = ListMLE(pred[universe], y[universe])  # get the cross-sectional order right
    mag_loss  = Huber(pred[universe],  y[universe])   # anchor the scale so the levels aren't junk
    loss      = rank_weight*rank_loss + mag_weight*mag_loss + l1_edge_weight*mean(|A|)
                                                      # defaults 0.6 / 0.4. no BCE, order implies sign.

    # KNOW the OLD loss was a point-Huber on ONE target ticker + an aux ranking term,
    # which basically let a constant prediction win. rank_mag is universe-wide so that
    # can't happen. legacy loss_kinds (combined/huber/listmle/rank_dir) still exist but
    # rank_mag is the default now. see THESIS_LOG 12.19.

stuff i log per epoch (also goes to metrics.csv):
    val_mse        MSE on target (scaled units)
    val_rmse_pct   sqrt(MSE) / scale, easier to read
    dir_acc        how often i got the sign right
    val_ic         per-day Pearson across the universe, averaged
    val_rank_ic    per-day Spearman, averaged
    val_rank_icir  rank_ic mean / std (basically Sharpe-of-IC, stability)

early stop: do NOT stop on val_mse, way too noisy. instead i use a smoothed
(5-epoch rolling mean) val_rank_ic, on the EMA weights, and only after a burn-in.
that combo makes "best at epoch 1" basically impossible, which was the whole
point. see THESIS_LOG 12.11. # WORKING
# TODO double-check the 5-epoch smoothing still helps now that runs use full history.

best.pt = EMA weights from the best epoch. last.pt = live weights, only for
resume. inference always reads best.pt.
"""
from __future__ import annotations

import contextlib
import csv
import logging
import math
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import CKPT_BEST, CKPT_LAST, maybe_resume, save_checkpoint, save_config
from .config import Config
from .data import TAGCWindows, build_loaders
from .graph_logger import GraphLogger
from .model import TAGC


log = logging.getLogger("tagc.train")

_GRAD_GROUPS = ("macro_encoder", "stock_encoder", "graph", "gat", "head",
                "stock_id_emb", "residual_norm")


# --------------------------------------------------------------------------
# weight EMA, my fix for the "best is always epoch 1" mess
# --------------------------------------------------------------------------
class WeightEMA:
    """exp moving average of the params.

    update every optimizer step: shadow = decay*shadow + (1-decay)*live

    eval + best-checkpoint run on the EMA weights via the applied(model)
    context manager (swaps live<->shadow). inference just reads best.pt, which
    stores the EMA state under model_state, so predict/walk_forward/finetune
    don't need to know any of this. # WORKING

    decay=0.995 is roughly a 200-step window. ~26 steps/epoch (TBPTT=10 over
    ~262 train days), so call it ~8 epochs of smoothing.
    """
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        # only snapshot float params/buffers. int buffers are things like
        # positional ids and you do NOT want to average those.
        self.shadow: dict = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        """swap EMA weights into model for the with-block, then put the live
        weights back. use this for eval."""
        live = {k: model.state_dict()[k].detach().clone() for k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
        try:
            yield
        finally:
            model.load_state_dict({**model.state_dict(), **live}, strict=False)


def _module_grad_norms(model: torch.nn.Module) -> dict:
    norms = {}
    for name in _GRAD_GROUPS:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        sq = 0.0; has_any = False
        for p in mod.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
                has_any = True
        if has_any:
            norms[name] = sq ** 0.5
    return norms


# --------------------------------------------------------------------------
# ListMLE, listwise ranking loss for the cross-section
# --------------------------------------------------------------------------
def listmle_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ListMLE: neg log-likelihood of the true ranking given my scores.

        L = - sum_i [ s_pi(i) - logsumexp(s_pi(i), ..., s_pi(n)) ]

    pi sorts the targets high->low. the logcumsumexp-on-reversed trick keeps it
    numerically stable.

    scores  [N]  my predicted scores (any reals)
    targets [N]  the truth that defines the ranking (higher = better)

    returns a scalar; lower means my scores agree with the target order.
    """
    if scores.numel() < 2:
        return torch.zeros((), device=scores.device)
    # sort by descending target -> pi
    _, perm = torch.sort(targets, descending=True)
    s_sorted = scores[perm]
    # need logsumexp(s_sorted[i:]) per position. that's a reverse cum-logsumexp:
    # flip, cum-logsumexp, flip back.
    rev = s_sorted.flip(0)
    rev_cum = torch.logcumsumexp(rev, dim=0)
    log_denom = rev_cum.flip(0)
    loss = -(s_sorted - log_denom).sum()
    # divide by N so the term weighs the same on days with different N.
    return loss / float(scores.numel())


# --------------------------------------------------------------------------
# per-day cross-sectional IC + Rank-IC
# --------------------------------------------------------------------------
def _log_ljung_box(cfg: Config, loaders: dict) -> None:
    """Ljung-Box on the target's training-period RAW returns.

    prints the stat + p-value at lags {1,5,10,20} for three transforms:
        raw   ret_t      -> autocorr in the mean
        abs   |ret_t|    -> autocorr in magnitude
        sq    ret_t^2    -> autocorr in variance (ARCH)

    what i expect on real stock data:
        raw -> p > 0.05 everywhere (mean isn't predictable, the EMH wall)
        abs / sq -> p < 0.001 everywhere (vol clusters)
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception as e:                          # pragma: no cover
        log.warning("ljung-box: statsmodels not installed (%s) — skipping", e)
        return
    import pandas as pd
    try:
        df = pd.read_parquet(cfg.stocks_parquet)
        tgt = cfg.target_ticker.upper()
        if "close_ret_raw" not in df.columns:
            log.warning("ljung-box: close_ret_raw not in parquet — skipping")
            return
        sub = df.xs(tgt, level="ticker")["close_ret_raw"].dropna()
        # train period only (chronological split, no peeking)
        D = len(sub)
        train_end = int(D * cfg.train_frac)
        train_series = sub.iloc[:train_end]
    except Exception as e:
        log.warning("ljung-box: target series load failed (%s) — skipping", e)
        return

    log.info("─" * 64)
    log.info("Ljung-Box on training-period %s close_ret_raw (n=%d)",
             tgt, len(train_series))
    log.info("Expect: p>>0.05 for raw (no mean predictability);")
    log.info("        p~0    for abs / sq (volatility clusters → ARCH).")
    log.info(f"{'transform':>6s}  {'lag':>4s}  {'lb_stat':>10s}  {'p_value':>10s}")
    for name, series in [
        ("raw", train_series),
        ("abs", train_series.abs()),
        ("sq",  train_series ** 2),
    ]:
        try:
            lb = acorr_ljungbox(series.dropna(), lags=[1, 5, 10, 20], return_df=True)
            for lag, row in lb.iterrows():
                log.info(f"{name:>6s}  {int(lag):>4d}  "
                         f"{float(row['lb_stat']):>10.3f}  {float(row['lb_pvalue']):>10.4g}")
        except Exception as e:
            log.warning("ljung-box %s failed: %s", name, e)
    log.info("─" * 64)


def _slope(vals: list) -> float:
    """slope of a linear fit over a short series. used for the trend lines."""
    n = len(vals)
    if n < 2:
        return 0.0
    xs = np.arange(n, dtype=np.float32)
    ys = np.array(vals, dtype=np.float32)
    return float(np.polyfit(xs, ys, 1)[0])


def _day_ic(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Optional[tuple]:
    """(pearson_ic, spearman_rank_ic) for one day's cross-section, or None if
    there just aren't enough valid stocks."""
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(y)
    if valid.sum() < 10:
        return None
    p = pred[valid]; t = y[valid]
    # if everything's equal the correlation is undefined, bail. # KNOW
    if p.std() < 1e-12 or t.std() < 1e-12:
        return None
    from scipy.stats import pearsonr, spearmanr
    ic, _ = pearsonr(p, t)
    ric, _ = spearmanr(p, t)
    if not np.isfinite(ic) or not np.isfinite(ric):
        return None
    return float(ic), float(ric)


@dataclass
class EvalResult:
    """regression eval. loss = MSE on the scaled target, rmse_pct = same thing
    but in raw return units so it's readable, dir_acc = direction accuracy."""
    loss: float                  # MSE in scaled units
    rmse_pct: float              # RMSE / scale (avg pct-point error)
    dir_acc: float               # sign agreement with y_reg
    pred_mean: float
    pred_std: float
    avg_position: float = 0.0
    correct_weighted: float = 0.0
    # cross-sectional IC (Pearson) + Rank-IC (Spearman), computed per day across
    # the universe then averaged. this is the metric quant people actually care
    # about, and what TAGC is really chasing.
    ic_mean: float = 0.0          # mean daily Pearson(pred, y) across stocks
    ic_std:  float = 0.0
    icir:    float = 0.0          # mean / std (Sharpe-of-IC)
    rank_ic_mean: float = 0.0     # mean daily Spearman(pred, y)
    rank_ic_std:  float = 0.0
    rank_icir:    float = 0.0
    rank_loss:    float = 0.0     # avg listwise ranking loss on this split


def _enable_head_mc_dropout(model: nn.Module) -> None:
    """eval() everything, then flip just the head's dropout back to train() so
    MC-dropout actually samples."""
    model.eval()
    for m in model.head.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _z_position(p_mean: float, p_std: float, z_clamp: float = 2.0) -> float:
    """position size from the regression mean+std.
        z = mean / std, clamp to +/- z_clamp, normalise to +/- 1.
    bigger |position| = stronger predicted return per unit of uncertainty."""
    z = p_mean / max(p_std, 1e-4)
    z = max(-z_clamp, min(z_clamp, z))
    return z / z_clamp


def evaluate(
    model: TAGC,
    ds: TAGCWindows,
    cfg: Config,
    *,
    mc: bool = False,
    graph_logger: Optional[GraphLogger] = None,
    split_name: str = "val",
    predictions_csv: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> EvalResult:
    device = torch.device(cfg.device)
    if mc:
        _enable_head_mc_dropout(model)
    else:
        model.eval()

    tgt = cfg.target_idx
    sq_err_sum, n_seen = 0.0, 0
    correct_dir = 0
    pred_sum, pred_sqsum = 0.0, 0.0
    pos_size_acc, correct_weighted_acc = 0.0, 0.0

    # since v3.6 the predictions CSV is universe-wide: one row per (date, stock)
    # with the score, its rank, the predicted direction, and the raw forward
    # return. that's exactly what scripts/backtest.py eats to build the
    # long-short book + full-history Rank-IC.
    # FIX brittle if backtest.py ever renames a column, nothing here validates it.
    csv_writer = None
    csv_fh = None
    if predictions_csv is not None:
        csv_fh = open(predictions_csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "date", "ticker",
            "pred_score",        # the head output (cross-sectional score)
            "pred_rank",         # percentile rank in [0,1], 1 = top
            "pred_direction",    # 1 = predicted up (above the cross-sectional median), 0 = down
            "true_fwd_return",   # the REAL raw H-day forward return for this (date, ticker)
            "true_rank",         # the REAL return's percentile rank that day [0,1], compare vs pred_rank
            "direction_correct", # 1 if pred_direction matched the real return (vs that day's median)
        ])
        if tickers is None:
            tickers = [f"S{i}" for i in range(cfg.n_stocks)]

    # accumulators for daily IC / Rank-IC / ranking-loss
    daily_ic:      list = []
    daily_rank_ic: list = []
    daily_rank_loss: list = []

    h = model.init_hidden(device)
    A_prev = model.init_adjacency(device)
    with torch.no_grad():
        for i in range(len(ds)):
            X_s, X_m, _y_cls, y_reg, m, _, date_str, raw_fwd = ds[i]
            X_s = X_s.to(device); X_m = X_m.to(device)
            y_reg = y_reg.to(device); m = m.to(device)

            out = model(X_s, X_m, h, A_prev=A_prev)
            h = out["h"].detach()
            A_prev = out["A"].detach()
            target_emb = out["embeds"][tgt]

            # need the whole universe's preds for the cross-sectional IC
            pred_all = out["logits"]               # [N]

            if mc:
                samples = torch.stack([
                    model.head(target_emb.unsqueeze(0)).squeeze(0)
                    for _ in range(cfg.mc_samples)
                ])
                p_mean = float(samples.mean().item())
                p_std  = float(samples.std(unbiased=False).item())
                pred = p_mean
            else:
                pred = float(out["logits"][tgt].item())
                p_std = 0.0

            # per-day IC + Rank-IC across the full cross-section
            ic_pair = _day_ic(pred_all.detach().cpu().numpy(),
                              y_reg.detach().cpu().numpy(),
                              m.detach().cpu().numpy())
            if ic_pair is not None:
                daily_ic.append(ic_pair[0])
                daily_rank_ic.append(ic_pair[1])

            # per-day ranking loss, just for monitoring (no backprop here)
            valid_mask = m.bool() & torch.isfinite(y_reg)
            if valid_mask.sum() >= 2:
                rl = listmle_loss(pred_all[valid_mask], y_reg[valid_mask])
                if torch.isfinite(rl):
                    daily_rank_loss.append(float(rl.item()))

            if graph_logger is not None:
                graph_logger.maybe_log(date=date_str, split=split_name,
                                        A=out["A"], eps=out["eps"], embeds=out["embeds"])

            # ---- dump universe-wide preds (feeds the long-short backtest) ----
            if csv_writer is not None:
                vmask = m.detach().cpu().numpy().astype(bool)
                scores = pred_all.detach().cpu().numpy()
                raw_np = raw_fwd.detach().cpu().numpy()
                vi = np.where(vmask)[0]
                ranks = np.full(scores.shape[0], np.nan, dtype=np.float64)
                truer = np.full(scores.shape[0], np.nan, dtype=np.float64)   # actual rank of the real return
                real_med = 0.0
                if vi.size >= 2:
                    order = scores[vi].argsort().argsort().astype(np.float64)
                    ranks[vi] = order / (vi.size - 1)
                    torder = raw_np[vi].argsort().argsort().astype(np.float64)
                    truer[vi] = torder / (vi.size - 1)
                    real_med = float(np.median(raw_np[vi]))
                for j in vi:
                    dir_correct = int((scores[j] > 0.0) == (raw_np[j] > real_med))
                    csv_writer.writerow([
                        date_str, tickers[j],
                        f"{scores[j]:.6f}", f"{ranks[j]:.6f}",
                        int(scores[j] > 0.0), f"{raw_np[j]:.6f}",
                        f"{truer[j]:.6f}", dir_correct,
                    ])

            if float(m[tgt].item()) == 0.0:
                continue

            actual = float(y_reg[tgt].item())

            # both pred and actual are centered (loader subtracted the
            # per-ticker train mean from y_reg). keep them centered for
            # loss/MSE/pred_std so the units stay consistent, but un-center for
            # direction accuracy + the CSV so it reads as real returns.
            center_vals = getattr(cfg, "target_center_values", []) or []
            center = float(center_vals[tgt]) if tgt < len(center_vals) else 0.0
            pred_uncentered   = pred   + center
            actual_uncentered = actual + center

            err = pred - actual
            sq_err_sum += err * err
            n_seen += 1
            pred_sum += pred
            pred_sqsum += pred * pred
            # direction accuracy on the un-centered values, that's the real
            # "did the stock actually go up" question.
            sign_match = float((pred_uncentered > 0) == (actual_uncentered > 0))
            correct_dir += sign_match

            # position sizing has to use the un-centered pred too, same basis as
            # the direction call. i had centered `pred` here at first and it was
            # a real bug: mixed centered (position) with un-centered (direction)
            # units and trashed every Sharpe/PnL before the fix (the +2.86 in
            # RESULTS.md R2 was bogus). # KNOW
            pos = _z_position(pred_uncentered, p_std) if mc else max(-1.0, min(1.0, pred_uncentered / 2.0))
            pos_size_acc += abs(pos)
            correct_weighted_acc += abs(pos) * sign_match

    if csv_fh is not None:
        csv_fh.close()

    n = max(n_seen, 1)
    mse = sq_err_sum / n
    pmean = pred_sum / n
    pvar = max(pred_sqsum / n - pmean * pmean, 0.0)

    # roll up IC / Rank-IC over all the valid days
    if daily_ic:
        ic_mean = float(np.mean(daily_ic));  ic_std = float(np.std(daily_ic) or 0.0)
        icir    = ic_mean / max(ic_std, 1e-9)
    else:
        ic_mean = ic_std = icir = 0.0
    if daily_rank_ic:
        ric_mean = float(np.mean(daily_rank_ic));  ric_std = float(np.std(daily_rank_ic) or 0.0)
        ricir    = ric_mean / max(ric_std, 1e-9)
    else:
        ric_mean = ric_std = ricir = 0.0
    rank_loss = float(np.mean(daily_rank_loss)) if daily_rank_loss else 0.0

    return EvalResult(
        loss=mse,
        rmse_pct=math.sqrt(mse) / cfg.regression_scale,
        dir_acc=correct_dir / n,
        pred_mean=pmean / cfg.regression_scale,
        pred_std=math.sqrt(pvar) / cfg.regression_scale,
        avg_position=pos_size_acc / n,
        correct_weighted=correct_weighted_acc / max(pos_size_acc, 1e-9),
        ic_mean=ic_mean, ic_std=ic_std, icir=icir,
        rank_ic_mean=ric_mean, rank_ic_std=ric_std, rank_icir=ricir,
        rank_loss=rank_loss,
    )


class _Preempt:
    def __init__(self) -> None:
        self.stop = False
        self._prev = {}

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            try:
                self._prev[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass

    def _handle(self, signum, frame):
        log.warning("received signal %s — will checkpoint and exit at next epoch boundary", signum)
        self.stop = True


def train(cfg: Config, out_dir: Path, resume: bool = True) -> Dict[str, EvalResult]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    loaders = build_loaders(cfg)
    train_ds, val_ds, test_ds = loaders["train"], loaders["val"], loaders["test"]
    log.info("target=%s (idx %d)  n_stocks=%d  n_macro=%d  train=%d  val=%d  test=%d",
             cfg.target_ticker.upper(), cfg.target_idx,
             cfg.n_stocks, cfg.n_macro, len(train_ds), len(val_ds), len(test_ds))
    log.info("v5 regression: target=%s (FORWARD %d-day cumulative)  scale=%.1f",
             cfg.regression_target_col, getattr(cfg, "target_horizon", 5),
             cfg.regression_scale)

    # Ljung-Box diagnostic: quick check on whether the target's mean is
    # predictable from its own past. it isn't (EMH wall), but |ret| and ret^2
    # are (ARCH). just one log line at startup. # KNOW
    _log_ljung_box(cfg, loaders)

    save_config(out_dir, cfg)

    model = TAGC(cfg).to(device)
    # hand the model the canonical ticker order. the static_graph variant needs
    # it, no-op for the rest.
    model.set_universe_tickers(loaders["tickers"])
    log.info("model_variant = %s   loss_kind = %s   huber_weight = %.2f   "
             "aux_rank_weight = %.2f   l1_edge_weight = %.0e",
             cfg.model_variant, cfg.loss_kind, cfg.huber_weight,
             cfg.aux_rank_weight, cfg.l1_edge_weight)

    # 6 AdamW groups: {main, graph, macro} x {decay, no_decay}.
    # model.graph is None for the no_graph/static_graph variants, so guard it.
    graph_ids = {id(p) for p in model.graph.parameters()} if model.graph is not None else set()
    macro_ids = {id(p) for p in model.macro_encoder.parameters()}
    buckets = {"macro": ([], []), "graph": ([], []), "main": ([], [])}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = p.ndim <= 1 or name.endswith(".bias")
        bucket = "macro" if id(p) in macro_ids else "graph" if id(p) in graph_ids else "main"
        buckets[bucket][1 if is_no_decay else 0].append(p)
    graph_lr = cfg.lr * cfg.graph_lr_scale
    macro_lr = cfg.lr * cfg.macro_lr_scale
    opt = torch.optim.AdamW(
        [
            {"params": buckets["main"][0],  "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": buckets["main"][1],  "weight_decay": 0.0,              "lr": cfg.lr},
            {"params": buckets["graph"][0], "weight_decay": cfg.weight_decay, "lr": graph_lr},
            {"params": buckets["graph"][1], "weight_decay": 0.0,              "lr": graph_lr},
            {"params": buckets["macro"][0], "weight_decay": cfg.weight_decay, "lr": macro_lr},
            {"params": buckets["macro"][1], "weight_decay": 0.0,              "lr": macro_lr},
        ],
        lr=cfg.lr,
    )
    log.info("optimizer: base lr=%.2g  graph lr=%.2g (×%.2f)  macro lr=%.2g (×%.2f)",
             cfg.lr, graph_lr, cfg.graph_lr_scale, macro_lr, cfg.macro_lr_scale)

    # size the cosine schedule for max_epochs (longest possible run) so the LR
    # doesn't hit zero before the stopping criterion even fires.
    # KNOW: if early-stop fires well before max_epochs the LR never fully decays,
    # which is fine, the EMA weights are what we keep anyway.
    total_updates = max(1, (len(train_ds) // cfg.tbptt_steps) * cfg.max_epochs)
    warmup_steps = max(1, int(total_updates * cfg.warmup_frac))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    # weight EMA: splits "current model state" from "what i eval/save/serve".
    # the shadow moves slow enough that one lucky epoch can't grab "best", since
    # the EMA literally can't catch a sudden spike. # WORKING
    ema = WeightEMA(model, decay=cfg.ema_decay) if cfg.use_weight_ema else None
    if ema is not None:
        # window is roughly 1/(1-decay) steps (where the weight drops to 1/e)
        eff = int(round(1.0 / (1.0 - cfg.ema_decay)))
        log.info("weight EMA enabled  decay=%.4f  effective window ≈ %d optimizer steps",
                 cfg.ema_decay, eff)
    log.info("optimizer: %d updates total, %d warmup", total_updates, warmup_steps)

    start_epoch = 0
    # best_val = max Rank-IC so far (used to be min val_mse pre-v3). start at
    # -inf so literally any real value beats it on epoch 1.
    best_val: float = float("-inf")
    if resume:
        ckpt = maybe_resume(out_dir, model=model, optimizer=opt, cfg=cfg)
        if ckpt is not None:
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt["best_val"])
            log.info("resumed from epoch %d (best_rank_ic=%.4f)", start_epoch - 1, best_val)

    # rolling history of per-epoch val_rank_ic. feeds both the smoothed
    # early-stop metric and the slope diagnostic.
    rank_ic_history: List[float] = []
    train_rank_history: List[float] = []

    preempt = _Preempt(); preempt.install()

    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                # KNOW: train_mse historically stores the raw ListMLE rank loss
                # (== train_rank_loss); kept for backward-compat. train_mag_loss
                # is the REAL Huber/magnitude term — added so figures can plot it.
                "train_mse",       "train_rank_loss",   "train_mag_loss",
                "val_mse",         "val_rank_loss",
                "val_rmse_pct",    "val_dir_acc",
                "val_pred_mean",   "val_pred_std",
                "val_ic",          "val_ic_std",       "val_icir",
                "val_rank_ic",     "val_rank_ic_std",  "val_rank_icir",
                "best_rank_ic_so_far",
            ])

    last_val: Optional[EvalResult] = None
    tgt = cfg.target_idx
    # early-stop bookkeeping. no_improve counts epochs that didn't beat best_val
    # by at least early_stop_delta. only allowed to actually stop once
    # min_epochs has passed.
    no_improve = 0
    log.info("training: min_epochs=%d  max_epochs=%d  patience=%d (after min)  delta=%.1e",
             cfg.min_epochs, cfg.max_epochs, cfg.patience, cfg.early_stop_delta)

    epoch = start_epoch
    while epoch < cfg.max_epochs:
        model.train()
        # ask the graph builder to log its sim/g/A_new stats once this epoch.
        # it clears the flag itself after the first forward.
        if getattr(model, "graph", None) is not None:
            model.graph._log_sim = True
        h = model.init_hidden(device)
        A_prev = model.init_adjacency(device)
        running_loss, running_n = 0.0, 0
        rank_loss_running, rank_loss_n = 0.0, 0
        dir_loss_running, dir_loss_n = 0.0, 0
        mag_loss_running, mag_loss_n = 0.0, 0
        chunk_losses = []
        grad_norm_running: dict = {name: 0.0 for name in _GRAD_GROUPS}
        grad_norm_steps = 0

        def _flush_grads():
            nonlocal h, A_prev, grad_norm_steps
            if chunk_losses:
                opt.zero_grad()
                torch.stack(chunk_losses).sum().backward()
                if cfg.log_grad_norms:
                    norms = _module_grad_norms(model)
                    for k, v in norms.items():
                        grad_norm_running[k] = grad_norm_running.get(k, 0.0) + v
                    grad_norm_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                # update the EMA shadow right after the optimizer step
                if ema is not None:
                    ema.update(model)
                chunk_losses.clear()
            # always detach the carried state here, even on empty/all-masked
            # chunks. forgot this at first and the autograd graph just grew
            # forever across chunks. # WORKING
            h = h.detach()
            A_prev = A_prev.detach()

        for i in range(len(train_ds)):
            X_s, X_m, _y_cls, y_reg, m, _, _date, _raw = train_ds[i]
            X_s = X_s.to(device); X_m = X_m.to(device)
            y_reg = y_reg.to(device); m = m.to(device)

            out = model(X_s, X_m, h, A_prev=A_prev)
            pred, h, A = out["logits"], out["h"], out["A"]
            A_prev = A

            # ---- loss, switched on cfg.loss_kind ----------------------------
            # rank_dir (v3.6 default): universe-wide ListMLE + direction-BCE, a
            #   pure cross-sectional ranker. no single target row needed so it
            #   trains on every valid day.
            # combined/huber/listmle/learnable: legacy single-target Huber
            #   (+ universe ListMLE). note these still train the universe ranking
            #   on days where the target itself is masked.
            loss_kind = getattr(cfg, "loss_kind", "rank_dir")
            valid_mask = m.bool() & torch.isfinite(y_reg)
            tgt_valid = float(m[tgt].item()) != 0.0

            # ListMLE across the universe = the cross-sectional ranking term
            rl = None
            if valid_mask.sum() >= 2:
                _rl = listmle_loss(pred[valid_mask], y_reg[valid_mask])
                if torch.isfinite(_rl):
                    rl = _rl
                    rank_loss_running += float(_rl.item()); rank_loss_n += 1

            loss = None
            if loss_kind == "rank_mag":
                # v5 default. combined magnitude+ranking loss, both terms
                # universe-wide per day on the cross-sectional z-score target:
                #     loss = rank_weight*ListMLE(universe) + mag_weight*Huber(universe)
                # ListMLE (Xia et al. ICML 2008) gets the cross-sectional order;
                # the universe Huber (Huber 1964) pins the *scale* to the
                # standardized target. a constant mean can't minimise the
                # universe Huber the way it could the old single-target one, and
                # pure listwise leaves the levels totally uncalibrated
                # (Kwiatkowski & Chudziak CIKM '25). basically the regression+rank
                # setup from Feng et al. RSR (TOIS 2019). on purpose there's NO
                # direction/BCE term: the listwise order already covers sign, so
                # BCE would just be one more weight to babysit.
                if rl is not None:
                    loss = cfg.rank_weight * rl
                    running_loss += float(rl.item()); running_n += 1
                if cfg.mag_weight > 0 and valid_mask.any():
                    p_ = pred[valid_mask]; y_ = y_reg[valid_mask]
                    e_ = (p_ - y_).abs(); d_ = cfg.huber_delta
                    hub = torch.where(e_ < d_, 0.5 * (p_ - y_).pow(2),
                                      d_ * (e_ - 0.5 * d_)).mean()
                    if torch.isfinite(hub):
                        loss = cfg.mag_weight * hub if loss is None \
                            else loss + cfg.mag_weight * hub
                        mag_loss_running += float(hub.item()); mag_loss_n += 1
            elif loss_kind == "rank_dir":
                if rl is not None:
                    loss = rl                                          # ranking is the main term
                    if cfg.direction_bce_weight > 0 and valid_mask.any():
                        dir_label = (y_reg[valid_mask] > 0).float()
                        dl = F.binary_cross_entropy_with_logits(pred[valid_mask], dir_label)
                        if torch.isfinite(dl):
                            loss = loss + cfg.direction_bce_weight * dl
                            dir_loss_running += float(dl.item()); dir_loss_n += 1
                    if cfg.huber_weight > 0 and tgt_valid:            # optional little point term
                        err = pred[tgt] - y_reg[tgt]; a, d = err.abs(), cfg.huber_delta
                        loss = loss + cfg.huber_weight * torch.where(
                            a < d, 0.5 * err.pow(2), d * (a - 0.5 * d))
                    running_loss += float(rl.item()); running_n += 1
            elif tgt_valid:
                err = pred[tgt] - y_reg[tgt]; a, d = err.abs(), cfg.huber_delta
                target_loss = torch.where(a < d, 0.5 * err.pow(2), d * (a - 0.5 * d))
                if loss_kind == "huber":
                    loss = target_loss
                elif loss_kind == "listmle":
                    loss = rl if rl is not None else target_loss
                elif loss_kind == "learnable":
                    s_h = model.log_sigma_huber
                    loss = torch.exp(-s_h) * target_loss + s_h
                    if rl is not None:
                        s_r = model.log_sigma_rank
                        loss = loss + torch.exp(-s_r) * rl + s_r
                else:                                                # 'combined'
                    loss = cfg.huber_weight * target_loss
                    if rl is not None and cfg.aux_rank_weight > 0:
                        loss = loss + cfg.aux_rank_weight * rl
                running_loss += float(target_loss.item()); running_n += 1
            elif rl is not None and loss_kind in ("combined", "listmle"):
                # target's masked today, but still train the universe ranking
                loss = (cfg.aux_rank_weight if loss_kind == "combined" else 1.0) * rl

            if loss is not None:
                loss = loss + cfg.l1_edge_weight * A.abs().mean()
                chunk_losses.append(loss)

            if (i + 1) % cfg.tbptt_steps == 0:
                _flush_grads()
        _flush_grads()

        train_loss = running_loss / max(running_n, 1)
        train_rank_loss = rank_loss_running / max(rank_loss_n, 1)
        train_mag_loss = mag_loss_running / max(mag_loss_n, 1)
        # when EMA is on, always eval on the EMA weights. live weights stay
        # untouched for the next training epoch.
        if ema is not None:
            with ema.applied(model):
                val = evaluate(model, val_ds, cfg, split_name="val")
        else:
            val = evaluate(model, val_ds, cfg, split_name="val")
        last_val = val
        log.info(
            "epoch %02d  train_rank=%.4f  train_mag=%.4f  val_mse=%.5f  "
            "val_ic=%+.4f  val_rank_ic=%+.4f  ricir=%+.3f  "
            "dir_acc=%.3f  pred_mean=%+.4f  pred_std=%.4f",
            epoch, train_rank_loss, train_mag_loss,
            val.loss, val.ic_mean, val.rank_ic_mean, val.rank_icir,
            val.dir_acc, val.pred_mean, val.pred_std,
        )
        if cfg.log_grad_norms and grad_norm_steps > 0:
            avg = {k: v / grad_norm_steps for k, v in grad_norm_running.items() if v > 0}
            ordered = [(k, avg[k]) for k in _GRAD_GROUPS if k in avg]
            log.info("  grad-norms  " + "  ".join(f"{k}={v:.4f}" for k, v in ordered))
        # for the learnable-weight loss, log the effective term weights so i can
        # see how the optimiser ended up balancing Huber vs rank.
        if cfg.loss_kind == "learnable" and getattr(model, "log_sigma_huber", None) is not None:
            wh = float(torch.exp(-model.log_sigma_huber.detach()).item())
            wr = float(torch.exp(-model.log_sigma_rank.detach()).item())
            ratio = (wr / wh) if wh > 1e-9 else float("nan")
            log.info("  learnable loss weights  w_huber=%.4f  w_rank=%.4f  "
                     "(huber:rank = 1:%.2f)", wh, wr, ratio)

        save_checkpoint(out_dir, model=model, optimizer=opt, cfg=cfg,
                        epoch=epoch, best_val=best_val, scaler=loaders["scaler"], name=CKPT_LAST)

        # smoothed early-stop metric: a rolling N-epoch mean of val_rank_ic.
        # single-epoch numbers are way too noisy on this small val set to pick a
        # checkpoint off of. a sustained climb is the real signal.
        rank_ic_history.append(val.rank_ic_mean)
        train_rank_history.append(train_rank_loss)
        smooth_w = cfg.smooth_window
        recent = rank_ic_history[-smooth_w:]
        smooth_rank_ic = float(sum(recent) / len(recent))
        if cfg.log_grad_norms:
            log.info("  smoothed val_rank_ic (%d-epoch mean) = %+.4f   raw = %+.4f",
                     len(recent), smooth_rank_ic, val.rank_ic_mean)

        # quick "is it still learning" diagnostic. once there's enough history,
        # fit a slope over the last few epochs of train_rank_loss + val_rank_ic.
        # train up + val down = overfitting, both flat = stalled, both up = still
        # going.
        if len(rank_ic_history) >= 5:
            tr_slope = _slope(train_rank_history[-5:])
            va_slope = _slope(rank_ic_history[-5:])
            log.info("  trend  Δ train_rank/ep=%+.4f   Δ val_rank_ic/ep=%+.4f",
                     tr_slope, va_slope)
            # only warn after epoch 10 so warmup noise doesn't trip it
            if epoch >= 10:
                if abs(tr_slope) < 1e-3 and abs(va_slope) < 1e-3:
                    log.warning("  diag: BOTH slopes ~0 — model has stopped learning")
                elif tr_slope < -1e-3 and va_slope > 1e-3:
                    pass  # loss down, IC up = exactly what i want
                elif tr_slope < -1e-3 and va_slope < -1e-3:
                    log.warning("  diag: train improving but val Rank-IC declining → "
                                "possibly OVERFITTING")

        # best-tracking, but with a burn-in. for the first best_burnin_epochs
        # the optimizer is still warming up and the val numbers are basically
        # noise, so i refuse to call anything "best" until we're past that. # KNOW
        in_burnin = epoch < cfg.best_burnin_epochs
        improved = (not in_burnin) and (smooth_rank_ic > best_val + cfg.early_stop_delta)
        if in_burnin:
            log.info("  (burn-in %d/%d — best-tracking disabled)",
                     epoch + 1, cfg.best_burnin_epochs)
        elif improved:
            best_val = smooth_rank_ic
            no_improve = 0
            # write the EMA-shadow weights as model_state in best.pt, so the
            # inference loaders (predict/walk_forward/finetune) just pick up the
            # smoothed model for free. live weights stay in last.pt for resume.
            if ema is not None:
                with ema.applied(model):
                    save_checkpoint(out_dir, model=model, optimizer=opt, cfg=cfg,
                                    epoch=epoch, best_val=best_val,
                                    scaler=loaders["scaler"], name=CKPT_BEST)
            else:
                save_checkpoint(out_dir, model=model, optimizer=opt, cfg=cfg,
                                epoch=epoch, best_val=best_val,
                                scaler=loaders["scaler"], name=CKPT_BEST)
            log.info("  new best smoothed val_rank_ic=%+.4f -> saved %s%s",
                     best_val, CKPT_BEST, "  (EMA weights)" if ema is not None else "")
        else:
            no_improve += 1

        with metrics_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_rank_loss:.6f}",
                f"{train_mag_loss:.6f}",
                f"{val.loss:.6f}",
                f"{val.rank_loss:.6f}",
                f"{val.rmse_pct:.6f}",
                f"{val.dir_acc:.6f}",
                f"{val.pred_mean:.6f}", f"{val.pred_std:.6f}",
                f"{val.ic_mean:.6f}", f"{val.ic_std:.6f}", f"{val.icir:.6f}",
                f"{val.rank_ic_mean:.6f}", f"{val.rank_ic_std:.6f}", f"{val.rank_icir:.6f}",
                f"{best_val:.6f}",
            ])

        # bump epoch BEFORE the break checks so it's always "epochs completed",
        # not "index of last one". keeps the dashboard's "epochs trained: N"
        # honest no matter how we exit (natural end, patience, or SIGTERM). # KNOW
        epoch += 1

        if preempt.stop:
            log.warning("stopping early after epoch %d due to signal", epoch - 1)
            break

        if epoch >= cfg.min_epochs and no_improve >= cfg.patience:
            log.info("early stop after epoch %d — no val improvement in %d epochs (min_epochs=%d met)",
                     epoch - 1, no_improve, cfg.min_epochs)
            break

    best_path = out_dir / CKPT_BEST
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        log.info("loaded best checkpoint from %s for final test", best_path)

    graph_logger = GraphLogger(out_dir / cfg.graphs_subdir, tickers=loaders["tickers"],
                                every=cfg.save_graphs_every)
    # write the universe-wide preds from the deterministic test pass (no
    # MC-dropout noise) so the backtest gets clean scores.
    test_plain = evaluate(model, test_ds, cfg, graph_logger=graph_logger, split_name="test",
                          predictions_csv=out_dir / "predictions_test.csv",
                          tickers=loaders["tickers"])
    test_mc    = evaluate(model, test_ds, cfg, mc=True, split_name="test_mc")
    log.info("TEST       mse=%.5f  rmse_pct=%.4f  dir_acc=%.3f  "
             "IC=%+.4f  Rank-IC=%+.4f  Rank-ICIR=%+.3f",
             test_plain.loss, test_plain.rmse_pct, test_plain.dir_acc,
             test_plain.ic_mean, test_plain.rank_ic_mean, test_plain.rank_icir)
    log.info("TEST (MC)  mse=%.5f  rmse_pct=%.4f  dir_acc=%.3f  avg|pos|=%.3f  dir_acc@|pos|=%.3f",
             test_mc.loss, test_mc.rmse_pct, test_mc.dir_acc,
             test_mc.avg_position, test_mc.correct_weighted)

    # long-short backtest + full-history Rank-IC, off the universe-wide CSV
    try:
        import sys as _sys
        _repo = Path(__file__).resolve().parent.parent          # REPO/model/train.py -> REPO
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from scripts.backtest import backtest
        bt = backtest(out_dir / "predictions_test.csv", horizon=cfg.target_horizon, out_dir=out_dir)
        log.info("BACKTEST   Rank-IC=%+.4f (ICIR %+.2f)  L/S Sharpe=%.2f  hit=%.3f  over %d periods",
                 bt.get("rank_ic_mean", float('nan')), bt.get("rank_icir", float('nan')),
                 bt.get("ls_sharpe_nonoverlap", float('nan')), bt.get("ls_hit_rate", float('nan')),
                 bt.get("n_periods_nonoverlap", 0))
    except Exception as _e:
        log.warning("backtest skipped: %s", _e)

    # auto-generate all figures + the interactive graph HTML right here, so no
    # separate command is needed after training. # WORKING each generator is
    # wrapped so a plotting failure never takes the run down.
    try:
        import sys as _sys
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from scripts import visualize as _viz, model_report as _rep, interactive_graph as _ig
        figs = _viz.generate(out_dir)
        qs = _rep.generate(out_dir)
        html = _ig.generate(out_dir, quiet=True)
        log.info("FIGURES    %d standard + %d question-figures -> %s/figures/",
                 len(figs), len(qs), out_dir)
        if html is not None:
            log.info("INTERACTIVE  %s", html)
    except Exception as _e:
        log.warning("figure generation skipped: %s", _e)

    # dashboard dump
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║                      TRAINING DASHBOARD                       ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║ target:  %-52s ║", cfg.target_ticker.upper())
    log.info("║ out_dir: %-52s ║", str(out_dir))
    log.info("║ epochs trained: %d  (min=%d, max=%d, patience=%d)            ║",
             epoch, cfg.min_epochs, cfg.max_epochs, cfg.patience)
    log.info("║ best val Rank-IC: %+.4f                                      ║", best_val)
    log.info("║ test MSE:         %.5f   rmse_pct: %.4f%%   dir_acc: %.3f   ║",
             test_mc.loss, test_mc.rmse_pct * 100, test_mc.dir_acc)
    log.info("║ test IC:          %+.4f  (ICIR %+.3f)                          ║",
             test_mc.ic_mean, test_mc.icir)
    log.info("║ test Rank-IC:     %+.4f  (Rank-ICIR %+.3f)                     ║",
             test_mc.rank_ic_mean, test_mc.rank_icir)
    log.info("║ avg|position|:    %.3f   dir_acc@|pos|: %.3f                  ║",
             test_mc.avg_position, test_mc.correct_weighted)
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║ files in %s/:" % str(out_dir).ljust(52))
    log.info("║   best.pt, last.pt                  trained weights           ║")
    log.info("║   metrics.csv                       per-epoch curves          ║")
    log.info("║   predictions_test.csv              per-day MC predictions    ║")
    log.info("║   graphs/                           full per-day adj + embeds ║")
    log.info("║   figures/*.png                     auto-generated this run   ║")
    log.info("║   figures/interactive_graph.html    auto-generated this run   ║")
    log.info("║   backtest_summary.csv              gross L/S Sharpe + Rank-IC║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║ figures + interactive graph were generated automatically.    ║")
    log.info("║ to re-render manually:                                        ║")
    log.info("║   python scripts/visualize.py <out_dir>                       ║")
    log.info("║   python scripts/model_report.py <out_dir>                    ║")
    log.info("║   python scripts/interactive_graph.py <out_dir> --open        ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    return {"val": last_val, "test": test_plain, "test_mc": test_mc}
