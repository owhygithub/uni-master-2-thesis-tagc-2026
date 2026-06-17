"""reads the parquets and hands back one training batch per day for TAGC.

quick map of what happens here:
  1. load data/stocks_dataset.parquet + data/macro_signals.parquet
  2. pivot into dense [date, ticker, feature] cubes
  3. build the regression target (sum of close_ret_raw over the next 5 days)
  4. center the target per-ticker so it can't just predict the mean
  5. wrap it in a torch Dataset where item `t` = the 90-day window ending at
     t-1 plus the target at t (and the 4 days after)

what __getitem__ spits out:
    X_stock  [98, 90, F_stock]   every stock's features, window up to t-1
    X_macro  [10, 90, F_macro]   same window, macros
    y_cls    [98]                up/down label. legacy, only keep it for metrics
    y_reg    [98]                the 5d-forward target * scale
    mask     [98]                1 = good row, 0 = skip (IPO gap, dead tail, ...)
    t        scalar              day index
    date     str                 'YYYY-MM-DD'

# KNOW the encoder never sees row t (that's the future). window is rows [t-W, t-1].
# we walk day by day in order, no shuffling, so the GRU's memory actually means something.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config

log = logging.getLogger("tagc.data")


@dataclass
class Splits:
    train_end: int   # exclusive idx into the aligned date array
    val_end: int
    total: int


def _panel_to_array(
    df: pd.DataFrame,
    tickers: List[str],
    dates: pd.DatetimeIndex,
    columns: List[str],
) -> np.ndarray:
    """pivot a (date, ticker)-indexed frame into a dense [D, N, F] array.

    # KNOW any missing (date, ticker) pair comes back as NaN. caller has to
    # mask / zero-fill it, this fn won't.
    """
    D, N, F = len(dates), len(tickers), len(columns)
    out = np.full((D, N, F), np.nan, dtype=np.float32)
    if df.empty:
        return out
    # unstack ticker -> wide frame indexed by date, cols are (feature, ticker)
    wide = df[columns].unstack("ticker")
    wide = wide.reindex(dates)
    for fi, col in enumerate(columns):
        sub = wide[col].reindex(columns=tickers)     # [D, N]
        out[:, :, fi] = sub.to_numpy(dtype=np.float32)
    return out


def _load_panels(cfg: Config):
    """returns (stocks_panel, macro_panel, labels, mask, dates, tickers, macro_tickers)."""
    stocks = pd.read_parquet(cfg.stocks_parquet)
    macro  = pd.read_parquet(cfg.macro_parquet)

    if not isinstance(stocks.index, pd.MultiIndex) or list(stocks.index.names) != ["date", "ticker"]:
        raise ValueError(f"stocks parquet must have MultiIndex (date, ticker); got {stocks.index.names}")
    if not isinstance(macro.index, pd.MultiIndex) or list(macro.index.names) != ["date", "ticker"]:
        raise ValueError(f"macro parquet must have MultiIndex (date, ticker); got {macro.index.names}")

    # intersect the date axes so both panels line up
    stock_dates = stocks.index.get_level_values("date").unique()
    macro_dates = macro.index.get_level_values("date").unique()
    dates = pd.DatetimeIndex(sorted(set(stock_dates) & set(macro_dates)))

    # optional calendar slice (e.g. just 2012-2019). runs before last_n_days so you
    # can stack them, but usually i only use one.
    start_date = getattr(cfg, "start_date", None)
    end_date   = getattr(cfg, "end_date", None)
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]

    if cfg.last_n_days is not None:
        dates = dates[-cfg.last_n_days:]

    tickers = sorted(stocks.index.get_level_values("ticker").unique().tolist())
    macro_tickers = sorted(macro.index.get_level_values("ticker").unique().tolist())

    # drop everything outside the window we use, keeps memory down
    stocks = stocks.loc[stocks.index.get_level_values("date").isin(dates)]
    macro  = macro.loc[macro.index.get_level_values("date").isin(dates)]

    stocks_panel = _panel_to_array(stocks, tickers,       dates, cfg.stock_feature_columns)
    macro_panel  = _panel_to_array(macro,  macro_tickers, dates, cfg.macro_feature_columns)

    # labels: grab binary label_up if it's there. newer datasets dropped it, so
    # fall back to close_ret > 0 (same thing).
    if cfg.label_col in stocks.columns:
        label_wide = stocks[cfg.label_col].unstack("ticker").reindex(index=dates, columns=tickers)
        labels = label_wide.to_numpy(dtype=np.float32)
    else:
        # derive from close_ret on the fly (it has to be in the panel)
        if "close_ret" not in cfg.stock_feature_columns:
            raise ValueError(
                f"label_col={cfg.label_col!r} missing AND close_ret not in stock_feature_columns; "
                f"cannot derive labels.")
        cr_idx = cfg.stock_feature_columns.index("close_ret")
        labels = (stocks_panel[:, :, cr_idx] > 0).astype(np.float32)

    # mask: also optional now (strict alignment means no holes anyway)
    if cfg.mask_col in stocks.columns:
        miss_wide = stocks[cfg.mask_col].unstack("ticker").reindex(index=dates, columns=tickers)
        miss = miss_wide.to_numpy(dtype=np.float32)
        mask = (1.0 - np.nan_to_num(miss, nan=1.0)).astype(np.float32)
    else:
        mask = np.ones((len(dates), len(tickers)), dtype=np.float32)
    mask = mask * (~np.isnan(labels)).astype(np.float32)

    # zero-fill the feature NaNs so torch doesn't choke. mask still flags them anyway.
    stocks_panel = np.nan_to_num(stocks_panel, nan=0.0, posinf=0.0, neginf=0.0)
    macro_panel  = np.nan_to_num(macro_panel,  nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0)

    return stocks_panel, macro_panel, labels, mask, dates, tickers, macro_tickers


def _fit_scaler(panel: np.ndarray, mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D, N, F = panel.shape
    flat = panel.reshape(D * N, F)
    m = mask_2d.reshape(D * N).astype(bool)
    valid = flat[m]
    if valid.size == 0:
        return np.zeros(F, dtype=np.float32), np.ones(F, dtype=np.float32)
    mean = valid.mean(axis=0)
    std = np.where(valid.std(axis=0) < 1e-6, 1.0, valid.std(axis=0))
    return mean.astype(np.float32), std.astype(np.float32)


def prepare(cfg: Config):
    stocks_panel, macro_panel, labels, mask, dates, tickers, macro_tickers = _load_panels(cfg)

    # push ticker counts back into cfg so the model can size itself off it
    # TODO double-check this still lines up when last_n_days drops a thin tail.
    cfg.n_stocks = len(tickers)
    cfg.n_macro  = len(macro_tickers)
    # target ticker -> int index. hard error if it's not in the universe, no point
    # predicting a ticker we don't have.
    target = cfg.target_ticker.upper()
    if target not in tickers:
        raise ValueError(
            f"target_ticker={target!r} not in universe; first 10 available: {tickers[:10]}"
        )
    cfg.target_idx = tickers.index(target)

    # KNOW the regression target comes from close_ret_raw read straight off the
    # parquet, NOT from the feature panel. otherwise it'd leak in as an input.
    # FIX brittle if the parquet schema renames the return column, this just raises.
    target_col = cfg.regression_target_col
    stocks_df_raw = pd.read_parquet(cfg.stocks_parquet)
    if target_col not in stocks_df_raw.columns:
        raise ValueError(
            f"regression_target_col={target_col!r} not in parquet; columns: "
            f"{list(stocks_df_raw.columns)}"
        )
    target_panel = _panel_to_array(
        stocks_df_raw.loc[stocks_df_raw.index.get_level_values('date').isin(dates)],
        tickers, dates, [target_col]
    )[:, :, 0]  # [D, N], raw close_ret (not z-scored)
    target_panel = np.nan_to_num(target_panel, nan=0.0, posinf=0.0, neginf=0.0)

    D = len(dates)
    train_end = int(D * cfg.train_frac)
    val_end   = int(D * (cfg.train_frac + cfg.val_frac))
    splits = Splits(train_end=train_end, val_end=val_end, total=D)

    if cfg.refit_zscore:
        mean_s, std_s = _fit_scaler(stocks_panel[:train_end], mask[:train_end])
        stocks_panel = (stocks_panel - mean_s) / std_s
        # macro has no per-cell mask, so just pass ones
        mean_m, std_m = _fit_scaler(macro_panel[:train_end],
                                    np.ones(macro_panel.shape[:2], dtype=np.float32))
        macro_panel = (macro_panel - mean_m) / std_m
        scaler = (mean_s, std_s)
    else:
        F_s = stocks_panel.shape[-1]
        scaler = (np.zeros(F_s, dtype=np.float32), np.ones(F_s, dtype=np.float32))

    # ---- build the supervision target out of the forward-return panel -----
    # the build pipeline wrote return_5d / return_30d / return_60d as columns,
    # and target_panel [D,N] is the raw forward return. keep the raw panel around
    # (the long-short backtest needs it for PnL) and also build reg_targets, which
    # is either the legacy scaled+centered return or a per-day cross-sectional
    # transform (the ranking target).
    H = int(getattr(cfg, "target_horizon", 5))
    mask = mask.copy()
    raw_fwd = target_panel.astype(np.float32)               # raw forward returns [D,N]

    xst = getattr(cfg, "cross_sectional_target", "none")
    if xst != "none":
        # ---- cross-sectional per-day ranking target ----
        # each day, turn the forward returns across the valid stocks into a
        # zero-mean ~unit-var score. this kills the market common factor (big, but
        # you can't predict it cross-sectionally) so a constant prediction is
        # useless and all that's left is the relative ordering, which is exactly
        # what the graph/GAT is for. # WORKING
        # KNOW the per-day standardize self-normalizes the bigger 30d spread, so
        # 5d vs 30d both end up unit-ish. that's why regression_scale gets forced to 1.
        #   'zscore'          -> (r - mean) / std across stocks that day
        #   'rank'            -> percentile rank [0,1], recentered to [-.5,.5]
        #   'rank_and_zscore' -> gaussian-ized rank (van der Waerden). robust +
        #                        unit-var. this is the one i use.
        from scipy.stats import norm as _norm
        cfg.regression_scale = 1.0
        reg_targets = np.zeros_like(raw_fwd)
        MIN_XS = 10
        for d in range(len(dates)):
            valid = mask[d].astype(bool) & np.isfinite(raw_fwd[d])
            K = int(valid.sum())
            if K < MIN_XS:
                mask[d, :] = 0.0                            # cross-section too thin, drop the whole day
                continue
            r = raw_fwd[d, valid]
            if xst == "zscore":
                z = (r - r.mean()) / max(float(r.std()), 1e-8)
            elif xst == "rank":
                order = r.argsort().argsort().astype(np.float32)
                z = order / max(K - 1, 1) - 0.5
            else:  # rank_and_zscore
                order = r.argsort().argsort().astype(np.float32)
                z = _norm.ppf((order + 0.5) / K).astype(np.float32)
            row = np.zeros(reg_targets.shape[1], dtype=np.float32)
            row[valid] = z.astype(np.float32)
            reg_targets[d] = row
            mask[d] = valid.astype(np.float32)              # tighten mask down to finite+valid
        cfg.target_center_values = []
    else:
        # ---- legacy path: raw scaled return + per-ticker / cross-section centering
        reg_targets = (raw_fwd * float(cfg.regression_scale)).astype(np.float32)
        nan_target = ~np.isfinite(reg_targets)
        if nan_target.any():
            mask = mask * (~nan_target).astype(np.float32)
            reg_targets = np.nan_to_num(reg_targets, nan=0.0)
        target_center = getattr(cfg, "target_center", "train_mean")
        if target_center == "train_mean":
            train_slice = reg_targets[:train_end]
            train_mask  = mask[:train_end].astype(bool)
            ticker_means = np.zeros(reg_targets.shape[1], dtype=np.float32)
            for j in range(reg_targets.shape[1]):
                vals = train_slice[:, j][train_mask[:, j]]
                vals = vals[np.isfinite(vals)]
                ticker_means[j] = float(vals.mean()) if vals.size > 0 else 0.0
            reg_targets = reg_targets - ticker_means[np.newaxis, :]
            cfg.target_center_values = ticker_means.tolist()
        elif target_center == "cross_section":
            means_per_day = np.zeros(reg_targets.shape[0], dtype=np.float32)
            for d in range(reg_targets.shape[0]):
                row_mask = mask[d].astype(bool)
                if row_mask.any():
                    means_per_day[d] = float(reg_targets[d, row_mask].mean())
            reg_targets = reg_targets - means_per_day[:, np.newaxis]
            cfg.target_center_values = []
        else:
            cfg.target_center_values = []

    reg_targets = np.nan_to_num(reg_targets, nan=0.0, posinf=0.0, neginf=0.0)
    raw_fwd     = np.nan_to_num(raw_fwd,     nan=0.0, posinf=0.0, neginf=0.0)

    # sanity log: print the scaled-target std so a regression_scale vs parquet
    # mismatch jumps out right away. healthy scaled std is ~1, and the implied raw
    # std (scaled_std / regression_scale) should match the parquet return column.
    _valid = reg_targets[mask.astype(bool)]
    if _valid.size:
        _sstd = float(np.std(_valid))
        log.info("target check: %s  regression_scale=%.1f → scaled std=%.3f (raw≈%.4f)",
                 cfg.regression_target_col, cfg.regression_scale, _sstd,
                 _sstd / max(cfg.regression_scale, 1e-9))

    return {
        "stocks": stocks_panel,
        "macro": macro_panel,
        "labels": labels,
        "reg_targets": reg_targets,
        "raw_fwd": raw_fwd,                # raw forward returns [D,N], backtest PnL uses this
        "mask": mask,
        "dates": dates,
        "tickers": tickers,
        "macro_tickers": macro_tickers,
        "splits": splits,
        "scaler": scaler,
        "target_center_values": cfg.target_center_values,
    }


class TAGCWindows(Dataset):
    """one item per valid day t. now carries the regression target too.

    gives back:
        X_stock [N, W, F_stock]
        X_macro [M, W, F_macro]
        y_cls   [N]        up/down direction (label_up at row t-1)
        y_reg   [N]        scaled return (close_ret at row t)
        mask    [N]
        day_idx
        date_str
    """

    def __init__(
        self,
        stocks:      np.ndarray,    # [D, N, F_stock]
        macro:       np.ndarray,    # [D, M, F_macro]
        labels:      np.ndarray,    # [D, N]   binary
        reg_targets: np.ndarray,    # [D, N]   training target (xs score or scaled return)
        raw_fwd:     np.ndarray,    # [D, N]   raw forward returns, backtest PnL uses this
        mask:        np.ndarray,    # [D, N]
        dates:       pd.DatetimeIndex,
        day_indices: np.ndarray,
        window:      int,
    ):
        self.stocks = stocks
        self.macro  = macro
        self.labels = labels
        self.reg_targets = reg_targets
        self.raw_fwd = raw_fwd
        self.mask   = mask
        self.dates  = dates
        self.day_indices = day_indices
        self.W = window

    def __len__(self) -> int:
        return len(self.day_indices)

    def __getitem__(self, i: int):
        t = int(self.day_indices[i])
        X_s = np.transpose(self.stocks[t - self.W : t], (1, 0, 2))   # [N, W, F_stock]
        X_m = np.transpose(self.macro[t - self.W : t],  (1, 0, 2))   # [M, W, F_macro]
        y_cls = self.labels[t - 1]                                    # direction label, row t-1
        y_reg = self.reg_targets[t]                                   # training target at t
        raw   = self.raw_fwd[t]                                       # raw fwd return at t (backtest)
        m = self.mask[t]                                              # KNOW mask lines up with the target at t, not t-1
        return (
            torch.from_numpy(X_s),
            torch.from_numpy(X_m),
            torch.from_numpy(y_cls),
            torch.from_numpy(y_reg),
            torch.from_numpy(m),
            torch.tensor(t, dtype=torch.long),
            str(self.dates[t - 1].date()),
            torch.from_numpy(raw),
        )


def build_loaders(cfg: Config):
    payload = prepare(cfg)
    W = cfg.window
    D = payload["splits"].total

    # first t we can use is W, otherwise [t-W, t-1] doesn't exist yet
    all_t = np.arange(W, D)
    train_t = all_t[all_t < payload["splits"].train_end]
    val_t   = all_t[(all_t >= payload["splits"].train_end) & (all_t < payload["splits"].val_end)]
    test_t  = all_t[all_t >= payload["splits"].val_end]

    def _ds(idx):
        return TAGCWindows(
            stocks=payload["stocks"], macro=payload["macro"],
            labels=payload["labels"], reg_targets=payload["reg_targets"],
            raw_fwd=payload["raw_fwd"], mask=payload["mask"],
            dates=payload["dates"], day_indices=idx, window=W,
        )

    return {
        "train": _ds(train_t),
        "val":   _ds(val_t),
        "test":  _ds(test_t),
        "dates": payload["dates"],
        "tickers": payload["tickers"],
        "macro_tickers": payload["macro_tickers"],
        "scaler": payload["scaler"],
        "splits": payload["splits"],
    }
