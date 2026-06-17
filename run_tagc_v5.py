"""Entry point. train TAGC v5 end-to-end on the prepared parquet dataset.

v5 supervises with MSE on the next-day return instead of BCE on the next-day
direction. All the scaffolding (TBPTT chunks, warmup+cosine LR, per-module LR
scaling, per-module grad-norm logging, atomic checkpoints, SIGTERM-safe) is
identical to v1.

Usage:
    python run_tagc_v5.py                                  # 20 epochs, 2y, target=AAPL
    python run_tagc_v5.py --fresh                          # wipe out_dir and retrain
    python run_tagc_v5.py --last-n-days 0 --out-dir runs/v2_full10y
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from model.config import Config
from model.train import train


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "train.log")],
        force=True,
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass


def _set_determinism(seed: int, strict: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stocks-parquet", default=os.environ.get("TAGC_STOCKS_PARQUET", None),
                   help="explicit path to stocks parquet; defaults to Config.use_news routing")
    p.add_argument("--macro-parquet",  default=os.environ.get("TAGC_MACRO_PARQUET",  "data/macro_signals.parquet"))
    p.add_argument("--no-news", dest="use_news", action="store_false", default=True,
                   help="(legacy) use the fundamentals (no-news) dataset variant")
    p.add_argument("--feature-set", default=None,
                   choices=["ohlcv", "fundamentals", "news", "final"],
                   help="pick which dataset to load. ohlcv/fundamentals/news = the "
                        "v3.4 builds; final = the v3.5 data/final_data/ panel (97 "
                        "tickers, 38 features). Default: news.")
    p.add_argument("--horizon", type=int, default=None, choices=[5, 30, 60],
                   help="prediction horizon in trading days (5/30; +60 for 'final').")
    p.add_argument("--variant", default=None,
                   choices=["tagc", "no_gru", "no_gat", "static_graph", "no_graph", "random_graph"],
                   help="model variant (default: tagc / the full pipeline)")
    p.add_argument("--loss", default=None,
                   choices=["combined", "huber", "listmle"],
                   help="loss kind. combined = Huber+ListMLE (default). "
                        "huber = predict-value task only. listmle = ranking only.")
    p.add_argument("--out-dir", default=os.environ.get("TAGC_OUT_DIR", "runs/v2_local"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--last-n-days", type=int, default=None,
                   help="0 = use full history; otherwise last N trading days")
    p.add_argument("--target-ticker", default=None,
                   help="ticker to train on (overrides Config default AAPL)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--fresh", action="store_true",
                   help="WIPE out_dir before training")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--stock-id-in-residual", default=None, choices=["true", "false"],
                   help="route the per-stock ID embedding into the head's residual "
                        "path (default true). Set false to remove the per-stock "
                        "constant shortcut, the anti mean-collapse diagnostic.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        out_resolved = out_dir.resolve()
        safe_roots = [Path("runs").resolve(), Path("/tmp").resolve(), Path("/var/scratch").resolve()]
        if not any(str(out_resolved).startswith(str(r)) for r in safe_roots):
            sys.exit(f"--fresh refused: {out_resolved} is not under runs/, /tmp/, or /var/scratch/")
        shutil.rmtree(out_dir)
    _setup_logging(out_dir)
    log = logging.getLogger("tagc.run")
    if args.fresh:
        log.info("--fresh: wiped %s before starting", out_dir)
        args.resume = False

    # v3.4: feature_set + horizon route to one of 6 parquets.
    # --feature-set and --horizon override use_news/horizon defaults.
    # KNOW the 6-parquet routing only holds for the legacy builds, 'final' is separate
    cfg_kwargs = dict(use_news=args.use_news)
    if args.feature_set is not None:
        cfg_kwargs['feature_set'] = args.feature_set
    if args.horizon is not None:
        cfg_kwargs['target_horizon'] = args.horizon
    cfg = Config(**cfg_kwargs)
    if args.stocks_parquet is not None:
        cfg.stocks_parquet = Path(args.stocks_parquet)
    # Only override the macro parquet when the user explicitly passed one, OR for
    # the legacy datasets. For feature_set='final', Config.__post_init__ already
    # points macro at data/final_data/macro.parquet, so don't clobber it with the
    # old default (whose columns wouldn't match the final macro schema).
    if args.macro_parquet != "data/macro_signals.parquet" or cfg.feature_set != "final":
        cfg.macro_parquet = Path(args.macro_parquet)
    if args.epochs is not None:
        # User-supplied --epochs forces a fixed N-epoch run (overrides the
        # min/max/patience defaults). Convenient for quick smoke tests.
        # KNOW this disables early stopping entirely, so don't use it for real runs
        cfg.min_epochs = args.epochs
        cfg.max_epochs = args.epochs
        cfg.patience = max(1, args.epochs)   # effectively disable early stop
    if args.last_n_days is not None:
        cfg.last_n_days = None if args.last_n_days == 0 else args.last_n_days
    if args.target_ticker is not None:
        cfg.target_ticker = args.target_ticker.upper()
    if args.loss is not None:
        cfg.loss_kind = args.loss
    if args.stock_id_in_residual is not None:
        cfg.stock_id_in_residual = (args.stock_id_in_residual == "true")
        log.info("override: stock_id_in_residual=%s", cfg.stock_id_in_residual)
    if args.variant is not None:
        cfg.model_variant = args.variant
        if args.variant == "static_graph":
            # Static-graph needs the pre-built sector adjacency.
            cfg.static_graph_path = REPO / "data" / "sectors_static.parquet"
            if not cfg.static_graph_path.exists():
                sys.exit(f"--variant static_graph: {cfg.static_graph_path} missing, "
                         f"build it via data-preprocessing/build_sectors.ipynb")

    _set_determinism(cfg.seed, strict=args.deterministic)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("v5 (regression)  device=%s  out_dir=%s  stocks=%s  macro=%s  last_n_days=%s",
             cfg.device, out_dir, cfg.stocks_parquet, cfg.macro_parquet, cfg.last_n_days)
    if cfg.device == "cuda":
        log.info("gpu=%s (count=%d)", torch.cuda.get_device_name(0), torch.cuda.device_count())

    train(cfg, out_dir=out_dir, resume=args.resume)


if __name__ == "__main__":
    main()
