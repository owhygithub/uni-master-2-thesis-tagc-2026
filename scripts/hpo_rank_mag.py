"""
Hyperparameter tuning for the rank_mag TAGC (Optuna TPE).
=========================================================

Tunes the model/optimizer hyperparameters that matter most, on the 30-day
horizon (where cross-sectional signal is strongest). The objective is the best
SMOOTHED VALIDATION Rank-IC (`best_rank_ic_so_far` in metrics.csv) — we tune on
VAL only, never on the test split.

Key design choices for a CPU-feasible, honest search:
  * data loaders are built ONCE and reused across trials (the architecture HPs
    don't change the data), via a cache shim on build_loaders.
  * reduced per-trial budget (small last_n_days, few epochs) — enough to rank
    configs, validated afterwards with a full run.
  * search includes the over-smoothing levers the AE/M3 audit flagged:
    n_layers_gat and topk.

Outputs -> hyperparam/:
  best_config.json   the winning hyperparameters (+ val Rank-IC)
  hpo_trials.csv     every trial (params + value)
  study.log          human-readable trial log

Run:  thesis-venv/bin/python scripts/hpo_rank_mag.py  [N_TRIALS]  [LAST_N_DAYS]  [MAX_EPOCHS]
"""
import sys, os, json, logging
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
import optuna

from model.config import Config
import model.data as D
import model.train as T

N_TRIALS    = int(sys.argv[1]) if len(sys.argv) > 1 else 16
LAST_N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
MAX_EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
HORIZON     = 30

HP = Path("hyperparam"); HP.mkdir(exist_ok=True)
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("hpo")
slog = open(HP / "study.log", "w")
def say(m):
    print(m, flush=True); slog.write(m + "\n"); slog.flush()

# --- cache loaders across trials (data config is constant) ------------------ #
_orig_build = D.build_loaders
_CACHE = {}
def cached_build_loaders(cfg):
    key = (cfg.target_horizon, cfg.last_n_days, cfg.feature_set, cfg.cross_sectional_target,
           cfg.target_ticker)
    if key not in _CACHE:
        loaders = _orig_build(cfg)
        _CACHE[key] = (loaders, cfg.n_stocks, cfg.n_macro, cfg.target_idx)
    loaders, ns, nm, ti = _CACHE[key]
    cfg.n_stocks, cfg.n_macro, cfg.target_idx = ns, nm, ti
    return loaders
T.build_loaders = cached_build_loaders                    # patch the name train.py uses

def base_cfg():
    c = Config(feature_set="final", target_horizon=HORIZON)
    c.device = "cpu"; c.last_n_days = LAST_N_DAYS
    c.min_epochs = c.max_epochs = MAX_EPOCHS; c.patience = MAX_EPOCHS
    c.best_burnin_epochs = 1; c.smooth_window = 3
    c.save_graphs_every = 0; c.log_grad_norms = False
    return c

def objective(trial: optuna.Trial) -> float:
    c = base_cfg()
    c.lr           = trial.suggest_float("lr", 4e-4, 2e-3, log=True)
    c.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    c.n_layers_gat = trial.suggest_categorical("n_layers_gat", [1, 2, 3])
    c.topk         = trial.suggest_categorical("topk", [3, 5, 8])
    c.dropout_gat  = trial.suggest_float("dropout_gat", 0.0, 0.3)
    c.head_dropout = trial.suggest_float("head_dropout", 0.1, 0.4)
    rw             = trial.suggest_categorical("rank_weight", [0.5, 0.6, 0.7])
    c.rank_weight  = rw; c.mag_weight = round(1.0 - rw, 2)
    out = Path(f"runs_final/hpo/trial_{trial.number:02d}")
    out.mkdir(parents=True, exist_ok=True)
    # isolate this trial's training log
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(out / "train.log", mode="w")], force=True)
    try:
        torch.manual_seed(0); np.random.seed(0)
        T.train(c, out_dir=out, resume=False)
        m = pd.read_csv(out / "metrics.csv")
        val = float(m["best_rank_ic_so_far"].max())
    except Exception as e:
        say(f"  trial {trial.number} FAILED: {e}")
        return -1.0
    say(f"  trial {trial.number:02d}  val_rankIC={val:+.4f}  | "
        f"lr={c.lr:.1e} wd={c.weight_decay:.1e} gat_layers={c.n_layers_gat} topk={c.topk} "
        f"drop_gat={c.dropout_gat:.2f} drop_head={c.head_dropout:.2f} rank_w={rw}")
    return val

say(f"HPO start: {N_TRIALS} trials | horizon={HORIZON} last_n_days={LAST_N_DAYS} "
    f"max_epochs={MAX_EPOCHS} | objective=best smoothed VAL Rank-IC")
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

# --- save results ----------------------------------------------------------- #
df = study.trials_dataframe()
df.to_csv(HP / "hpo_trials.csv", index=False)
best = study.best_trial
best_params = dict(best.params)
best_params["mag_weight"] = round(1.0 - best_params["rank_weight"], 2)
json.dump({"objective": "best_smoothed_val_rank_ic",
           "best_value": float(best.value),
           "best_params": best_params,
           "search": {"n_trials": N_TRIALS, "horizon": HORIZON,
                      "last_n_days": LAST_N_DAYS, "max_epochs": MAX_EPOCHS}},
          open(HP / "best_config.json", "w"), indent=2)
say("\n==== HPO DONE ====")
say(f"best val Rank-IC = {best.value:+.4f}")
say(f"best params = {json.dumps(best_params, indent=2)}")
say("saved -> hyperparam/best_config.json, hpo_trials.csv, study.log")
print("HPO_ALL_DONE", flush=True)
