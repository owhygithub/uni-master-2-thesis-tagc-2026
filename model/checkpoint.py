"""save/load checkpoints.

a checkpoint is just a torch.save bundle: model + optimizer state, the feature
scaler (mean/std), the Config, and bookkeeping (epoch, best val, rng states).
so a run can resume after getting killed, and inference is fully self-contained.

# TODO the rng states are saved but resume mid-epoch isn't truly bit-exact,
# we only resume on epoch boundaries, so close enough for the thesis.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .config import Config


CKPT_LAST = "last.pt"
CKPT_BEST = "best.pt"
CONFIG_JSON = "config.json"


def save_config(out_dir: Path, cfg: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(cfg)
    payload["stocks_parquet"] = str(cfg.stocks_parquet)
    payload["macro_parquet"]  = str(cfg.macro_parquet)
    with (out_dir / CONFIG_JSON).open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_checkpoint(
    out_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    epoch: int,
    best_val: float,
    scaler: Tuple[np.ndarray, np.ndarray],
    name: str = CKPT_LAST,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = scaler
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "epoch": epoch,
        "best_val": best_val,
        "scaler_mean": mean,
        "scaler_std": std,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }
    # atomic write: save to .tmp then rename, so a kill mid-save can't leave a
    # half-written checkpoint. # WORKING
    tmp = out_dir / (name + ".tmp")
    final = out_dir / name
    torch.save(payload, tmp)
    tmp.replace(final)
    return final


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def maybe_resume(
    out_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Optional[Config] = None,
) -> Optional[Dict[str, Any]]:
    """if last.pt is in out_dir, load it into model+optimizer and return the payload.

    hard-fails (on purpose) with a clear message if the saved architecture doesn't
    match the current model, that's almost always a config change you want to know about.

    cfg is the LIVE run config; we compare the saved horizon/target against it.
    # KNOW this used to compare against a fresh Config() default, so a resume with a
    # non-default horizon got checked against the wrong reference, now fixed.
    """
    last = out_dir / CKPT_LAST
    if not last.exists():
        return None
    ckpt = load_checkpoint(last)

    # ── horizon check ────────────────────────────────────────────────────
    # a 1d model and a 5d model predict totally different things and are NOT
    # interchangeable. the weights look compatible (same shapes) but the head is
    # calibrated to a different target distribution. # KNOW easy trap.
    saved_cfg = ckpt.get("cfg", {})
    saved_h = int(saved_cfg.get("target_horizon", 1))
    saved_col = saved_cfg.get("regression_target_col", "close_ret")
    # Compare against the LIVE run config (falls back to defaults only if the
    # caller didn't pass one, for backward compatibility).
    current = cfg if cfg is not None else Config()
    cur_h   = int(getattr(current, "target_horizon", 1))
    cur_col = getattr(current, "regression_target_col", "close_ret")
    if saved_h != cur_h or saved_col != cur_col:
        raise RuntimeError(
            f"\nCheckpoint target mismatch in {last}\n"
            f"  saved  target_horizon={saved_h}  regression_target_col={saved_col!r}\n"
            f"  current target_horizon={cur_h}   regression_target_col={cur_col!r}\n\n"
            f"These predict DIFFERENT THINGS and cannot be transferred.\n"
            f"To retrain in this directory from scratch:\n"
            f"  python tagc.py train <TICKER> --fresh\n"
            f"Or pick a different --out-dir for the new run.\n"
        )

    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        raise RuntimeError(
            f"\nCheckpoint architecture mismatch in {last}\n"
            f"  → {msg}\n\n"
            f"This means models/config.py changed since the checkpoint was saved\n"
            f"(e.g. n_layers_*, d_model, use_residual, use_stock_id_emb, etc.).\n\n"
            f"To retrain in this directory from scratch:\n"
            f"  python run_tagc.py --fresh --out-dir {out_dir}\n\n"
            f"To keep the old checkpoint and start a new run elsewhere:\n"
            f"  python run_tagc.py --out-dir runs/<new-name>\n"
        ) from None
    try:
        optimizer.load_state_dict(ckpt["optim_state"])
    except (KeyError, ValueError):
        # optimizer groups changed but the weights loaded fine, so just start the
        # optimizer fresh instead of crashing. # KNOW resume keeps the model but
        # drops the optimizer momentum here, fine for a short interruption.
        pass
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    return ckpt
