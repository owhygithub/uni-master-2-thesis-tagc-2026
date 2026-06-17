# TAGC final model (uni-master-tagc-2026)

## Overview
This is the final TAGC model (Temporal Attention Graph Constructor), a temporal-graph model
that ranks the S&P 500 cross-sectionally each day. I've slimmed it down to just what you need
to run and understand it: the model code, the manuals, the experiment setup code, and only the
final data the model actually uses. Start with `MASTER.ipynb` for an end-to-end run, or
`manuals/03_user_guides/SCRIPTS.md` for every command.

The model in one line: PatchTST encoder (M1), then a graph constructor (M2, top-5, edge-EMA),
then a 2-layer GAT (M3), then an MC-dropout head (M4), trained with `rank_mag` (0.6·ListMLE +
0.4·Huber, universe-wide on the cross-sectional z-score of forward returns). lr=3e-4, wd=1e-4,
`feature_set='final_v5'`.

## Getting started (fresh clone)
```bash
git clone <repo-url> && cd uni-master-tagc-2026
python3 -m venv thesis-venv                       # any Python 3.11+
thesis-venv/bin/pip install -r requirements.txt
```
The data is NOT in the repo. It's gitignored (~1.8 GB, and the parquet files exceed
GitHub's 100 MB per-file limit). To get it, either:
- rebuild it from `data-preprocessing/build_dataset_v5.ipynb` (yfinance + SEC), then run
  `experiments/2026-06-14_input_feature_screen/build_v5_features.py` for the v5 features, or
- ask me for the parquets: `data/feature_experiments/stocks_{5,30,60}d_v5.parquet`,
  `data/final_data/macro.parquet`, `data/sectors_static_labels.csv`, then drop them under `data/`.

Then run as below.

## Run it
```bash
cd /path/to/uni-master-tagc-2026

# end-to-end in the notebook (recommended):
#   open MASTER.ipynb in VS Code / Cursor and pick the kernel thesis-venv/bin/python, then Run All.
#   (or install jupyter first:  thesis-venv/bin/pip install jupyterlab && thesis-venv/bin/jupyter lab)

# or a single training run from the CLI
thesis-venv/bin/python run_tagc_v5.py --feature-set final_v5 --horizon 30 --out-dir runs_final/run30

# sanity-check the modules
thesis-venv/bin/python test/m3m4_integrity_test.py
```

## Layout
```
uni-master-tagc-2026/
├── MASTER.ipynb            # run the whole pipeline end-to-end (start here)
├── model/                  # THE final model package (M1/M2/M3/M4, train, data, config)
├── scripts/                # evaluation / experiment setup code (backtest, visualize, hpo, tune, baselines)
├── test/                   # module-validity setup code (M1/M2/M3/M4 integrity, autoencoder probe)
├── experiments/            # experiment SETUP CODE only (input-feature screen, AE probe, v5 build), no old results
├── data-preprocessing/     # the build pipeline (yfinance + SEC) to (re)create the dataset
├── hyperparam/             # the Optuna-tuned config (best_config.json) + study
├── manuals/                # the docs
│   ├── 01_research_log/        THESIS_LOG.md (full chronology) + WHAT_WE_FOUND.md
│   ├── 02_architecture_spec/   ARCHITECTURE.md (final spec) + diagram
│   ├── 03_user_guides/         SCRIPTS.md (commands), GUIDE.md, DATA.md
│   └── 04_results/             RESULTS.md, HYPERPARAMETERS.md
├── data/                   # ONLY the final data used: feature_experiments/stocks_{5,30,60}d_v5.parquet
│                           # + final_data/macro.parquet + sectors  (gitignored, ~1.8 GB)
├── run_tagc_v5.py          # CLI entry point (single training run)
├── requirements.txt, stocks.txt, macro.txt
└── thesis-venv/            # symlink to the Python environment
```

## Notes
- Only `feature_set='final_v5'` data is shipped (the model the project finalized). The legacy
  `'final'` (33-feature) parquets are not included, so rebuild via `data-preprocessing/` if you
  need them.
- `data/` and `thesis-venv/` are gitignored (large or machine-specific). The local venv is a
  symlink to my machine and is not in the repo, so create your own from `requirements.txt`
  (see Getting started above).
- Experiment results and figures from past runs were removed on purpose; only the runnable
  setup code is kept. Re-run anything from `experiments/`, `scripts/`, or `test/`.
- Honest expectation: a small, regime-dependent edge. Recent-regime Rank-IC is roughly 0 (the
  documented data ceiling), and the model reaches a higher ceiling where signal exists (pre-2020).
