"""One-shot experiment runner — trains the whole ablation/objective matrix and
auto-fills the results table.

## Overview
Define the run matrix once (`RUNS`, matching `results_master.csv`), then `run_all()`
trains every row in sequence. Each `train()` already auto-runs the backtest + figures,
so when a run finishes its metrics are sitting in `<out_dir>/backtest_summary.csv`.
`collect()` then scrapes every run, computes the Diebold–Mariano p-value vs the full-TAGC
reference (per period), and rewrites BOTH `experiments/results_master.csv` and the
paste-ready `manuals/04_results/EXPERIMENTS_TABLE.md`. Idempotent + resumable: a run whose
`backtest_summary.csv` already exists is skipped, so you can re-run the notebook any time.

Durability: each run's outputs are written to disk by `train()` as it finishes, and the
aggregate table is re-saved after EVERY run — so a crash in a later run never loses the
earlier ones (and a failing run is logged + skipped, the rest continue).

Use from the notebook (`RUN_ALL_EXPERIMENTS.ipynb`) or as a script:
    python experiments/experiment_runner.py                 # full run (>=12 epochs, early-stopped)
    python experiments/experiment_runner.py --smoke         # fast 5-epoch pipeline check
    python experiments/experiment_runner.py --collect       # just re-collect (no training)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "experiments" / "runs_ablation"
CSV_PATH = REPO / "experiments" / "results_master.csv"
MD_PATH = REPO / "manuals" / "04_results" / "EXPERIMENTS_TABLE.md"

# ── the run matrix (one dict per row of results_master.csv) ──────────────────
RUNS: List[dict] = [
    dict(run_id="h30_tagc_post",     variant="tagc",         loss="rank_mag", horizon=30, seed=42, period="post2020", notes="reference"),
    dict(run_id="h30_tagc_pre",      variant="tagc",         loss="rank_mag", horizon=30, seed=42, period="pre2020",  notes="reference"),
    dict(run_id="h30_no_graph",      variant="no_graph",     loss="rank_mag", horizon=30, seed=42, period="post2020", notes="RQ1"),
    dict(run_id="h30_static_graph",  variant="static_graph", loss="rank_mag", horizon=30, seed=42, period="post2020", notes="RQ2"),
    dict(run_id="h30_no_gru",        variant="no_gru",       loss="rank_mag", horizon=30, seed=42, period="post2020", notes="RQ3"),
    dict(run_id="h30_random_graph",  variant="random_graph", loss="rank_mag", horizon=30, seed=42, period="post2020", notes="RQ4"),
    dict(run_id="h30_no_gat",        variant="no_gat",       loss="rank_mag", horizon=30, seed=42, period="post2020", notes="support"),
    dict(run_id="obj_huber",         variant="tagc",         loss="huber",    horizon=30, seed=42, period="post2020", notes="support"),
    dict(run_id="obj_listmle",       variant="tagc",         loss="listmle",  horizon=30, seed=42, period="post2020", notes="support"),
]

# which run is the "full" reference each variant/objective is compared against
REFERENCE = {"post2020": "h30_tagc_post", "pre2020": "h30_tagc_pre"}


# ── config builder ──────────────────────────────────────────────────────────
def build_cfg(spec: dict, device: str = "cpu", target: str = "AAPL",
              min_epochs: Optional[int] = None, max_epochs: Optional[int] = None):
    """Translate one matrix row into a Config. Period sets the data window;
    variant/loss set the ablation; everything else stays at the locked v5 values.

    Epochs: leave both None to use the config defaults (min 30 / max 80, early-stopped).
    Set `min_epochs` to enforce a floor (e.g. 12) while keeping early stopping; set both
    equal for a fixed-length smoke run."""
    from model.config import Config
    cfg = Config(feature_set="final_v5", target_horizon=spec["horizon"])
    cfg.device = device
    cfg.seed = spec["seed"]
    cfg.target_ticker = target
    cfg.model_variant = spec["variant"]
    cfg.loss_kind = spec["loss"]
    if spec["variant"] == "static_graph":
        cfg.static_graph_path = REPO / "data" / "sectors_static.parquet"
    if spec["period"] == "pre2020":
        cfg.start_date, cfg.end_date = "2013-01-01", "2019-12-31"
    else:  # post2020 == full history; the test tail lands post-COVID
        cfg.start_date = cfg.end_date = None
        cfg.last_n_days = None
    if min_epochs is not None:
        cfg.min_epochs = min_epochs
    if max_epochs is not None:
        cfg.max_epochs = max_epochs
    if cfg.min_epochs > cfg.max_epochs:          # keep the cap >= the floor
        cfg.max_epochs = cfg.min_epochs
    return cfg


# ── train every row ─────────────────────────────────────────────────────────
def run_all(device: str = "cpu", target: str = "AAPL",
            min_epochs: Optional[int] = 12, max_epochs: Optional[int] = None,
            skip_done: bool = True, only: Optional[List[str]] = None,
            stop_on_error: bool = False) -> pd.DataFrame:
    """Train each run in sequence and SAVE AFTER EACH ONE.

    Durability: every `train()` writes its own outputs (best.pt, metrics.csv,
    predictions_test.csv, backtest_summary.csv, figures/) to its out_dir as it finishes,
    and we re-`collect()` the aggregate table immediately after each run. So if run #4
    crashes, runs #1–#3 are already fully saved AND in the results table. By default a
    failing run is logged and the rest continue (`stop_on_error=True` to halt instead).
    Resumable: a run that already has a backtest_summary.csv is skipped."""
    from model.train import train
    failures = []
    for spec in RUNS:
        if only and spec["run_id"] not in only:
            continue
        out = OUT_ROOT / spec["run_id"]
        if skip_done and (out / "backtest_summary.csv").exists():
            print(f"[skip] {spec['run_id']} — already done")
            continue
        cfg = build_cfg(spec, device=device, target=target,
                        min_epochs=min_epochs, max_epochs=max_epochs)
        print(f"\n=== TRAIN {spec['run_id']}  (variant={spec['variant']}, loss={spec['loss']}, "
              f"period={spec['period']}, epochs {cfg.min_epochs}..{cfg.max_epochs}) ===")
        try:
            train(cfg, out_dir=out, resume=False)
        except Exception as e:                    # one bad run must not lose the others
            failures.append((spec["run_id"], repr(e)))
            print(f"[FAIL] {spec['run_id']}: {e}")
            collect()                             # persist whatever finished before this
            if stop_on_error:
                break
            continue
        collect()                                 # save the aggregate table after each run
        print(f"[saved] results table refreshed after {spec['run_id']}")
    df = collect()
    if failures:
        print("\n=== FAILED runs (re-run the notebook to retry; finished ones are skipped) ===")
        for rid, err in failures:
            print(f"  - {rid}: {err}")
    return df


# ── Diebold–Mariano p-value on the per-day Rank-IC series ────────────────────
def _dm_pvalue(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Two-sided DM-style test that variant daily Rank-IC (a) differs from the
    reference (b), aligned on date. HAC-free, large-sample normal approx."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 8:
        return None
    d = (j["a"] - j["b"]).values
    sd = d.std(ddof=1)
    if sd == 0:
        return None
    t = d.mean() / (sd / math.sqrt(len(d)))
    # two-sided p from standard normal
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def _daily_ic(run_id: str) -> Optional[pd.Series]:
    p = OUT_ROOT / run_id / "backtest_daily.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" not in df or "rank_ic" not in df:
        return None
    return df.set_index("date")["rank_ic"]


# ── scrape every run + (re)write the csv and markdown ────────────────────────
def collect() -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        out = OUT_ROOT / spec["run_id"]
        rec = dict(run_id=spec["run_id"], variant=spec["variant"], loss=spec["loss"],
                   horizon=spec["horizon"], seed=spec["seed"], period=spec["period"],
                   rank_ic="", icir="", sharpe_gross="", hit_rate="",
                   dm_p_vs_full="", notes=spec["notes"])
        summ = out / "backtest_summary.csv"
        if summ.exists():
            s = pd.read_csv(summ).iloc[0]
            rec["rank_ic"] = round(float(s.get("rank_ic_mean", "nan")), 4)
            rec["icir"] = round(float(s.get("rank_icir", "nan")), 3)
            rec["sharpe_gross"] = round(float(s.get("ls_sharpe_nonoverlap", "nan")), 2)
            rec["hit_rate"] = round(float(s.get("ls_hit_rate", "nan")), 3)
        rows.append(rec)

    # DM vs the full-TAGC reference of the same period (reference rows stay blank)
    for rec in rows:
        ref_id = REFERENCE.get(rec["period"])
        if ref_id is None or rec["run_id"] == ref_id:
            continue
        a, b = _daily_ic(rec["run_id"]), _daily_ic(ref_id)
        if a is not None and b is not None:
            p = _dm_pvalue(a, b)
            if p is not None:
                rec["dm_p_vs_full"] = round(p, 3)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    _write_markdown(df)
    print(f"collected {df['rank_ic'].astype(bool).sum()}/{len(df)} runs -> "
          f"{CSV_PATH.name} + {MD_PATH.name}")
    return df


_MD_HEADER = """# Master experiment table (ablations + objective study)

## Overview
**One row per run.** Auto-filled by `experiments/experiment_runner.py` as jobs finish —
paste the table below straight into the results chapter. All runs: horizon 30, seed 42,
`feature_set='final_v5'`, lr 3e-4, rank/mag 0.6/0.4. `period`: **post2020** = full-data test
tail (data-ceiling regime) · **pre2020** = pre-COVID window (where signal exists).

Metric sources (per run's `backtest_summary.csv`): `rank_ic`←`rank_ic_mean`,
`icir`←`rank_icir`, `sharpe (gross)`←`ls_sharpe_nonoverlap`, `hit rate`←`ls_hit_rate`;
`DM p vs full` = Diebold–Mariano on per-day Rank-IC vs the period's `tagc` reference.

## Table
"""


def _cell(v) -> str:
    return "" if v == "" or v is None else str(v)


def _write_markdown(df: pd.DataFrame) -> None:
    cols = ["run_id", "variant", "loss", "horizon", "seed", "period",
            "rank_ic", "icir", "sharpe_gross", "hit_rate", "dm_p_vs_full", "notes"]
    head = ("| run id | variant | loss | horizon | seed | period | rank_ic | icir | "
            "sharpe (gross) | hit rate | DM p vs full | notes |")
    sep = "|" + "---|" * len(cols)
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(r[c]) for c in cols) + " |")
    MD_PATH.write_text(_MD_HEADER + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="only re-collect, don't train")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target", default="AAPL")
    ap.add_argument("--min-epochs", type=int, default=12, help="epoch floor (early stop above it)")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap (default: config's 80)")
    ap.add_argument("--smoke", action="store_true", help="fixed 5-epoch pass to test the pipeline")
    args = ap.parse_args()
    if args.collect:
        collect()
    elif args.smoke:
        run_all(device=args.device, target=args.target, min_epochs=5, max_epochs=5)
    else:
        run_all(device=args.device, target=args.target,
                min_epochs=args.min_epochs, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
