"""every knob for TAGC training lives here so I don't have to go hunting.

defaults are the thesis-depth setup. to try something else just pass overrides:

    cfg = Config(target_ticker="NVDA", last_n_days=504, epochs=20)

or poke at it after:

    cfg = Config()
    cfg.lr = 5e-5  # fine-tuning

which parquet gets loaded is decided by use_news (auto-routed in __post_init__),
and the feature column list comes from the same place.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# two feature lists: base (no news) and full (+ news). the build writes both:
#     data/stocks_dataset.parquet         (no news columns)
#     data/stocks_dataset_w_news.parquet  (+ 4 news PCs + presence flag)
#
# everything's already causally z-scored per ticker (252-day rolling) and
# clipped to +/-5 in the build, so the model sees std~1 inputs with no crazy
# tails. AdamW likes that.
# KNOW: the clip happens in the build, not here, so if a new feature isn't
# clipped upstream it'll sneak in with fat tails.
# split the lists by category so I can compose them per dataset.

# OHLCV + technicals + options + calendar (always there)
_OHLCV_FEATURES: List[str] = [
    # price level + return-shape
    "open_ret", "high_ret", "low_ret", "close_ret",
    "intraday_range", "body",
    "log_price_z",                    # causal 252d z of log(close)
    "price_percentile",               # [0,1], trailing 1y rank
    "log_volume_z",
    # technicals
    "rsi_norm", "macd_rel", "ema_20_rel",
    # engineered: cum returns + momentum + vol + 52w distance
    "ret_5d_cum", "ret_20d_cum",
    "mom_12_1",                       # 12-1 momentum (classic anomaly)
    "vol_20d",                        # 20d realised vol
    "dist_52w_high",                  # % below 52w high, in [-1,0]. # TODO double-check the sign convention holds on the 60d build
    # seasonality (cyclical + raw int)
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_month_end",
    "dow", "dom", "month", "quarter", "week_of_year",   # raw int versions
    # options proxy
    "iv_proxy",
]

# quarterly fundamentals (z-scored cross-sectionally per day)
_FUNDAMENTAL_FEATURES: List[str] = [
    "revenue", "net_income", "total_debt", "total_equity",
    "profit_margin", "debt_to_equity", "roe",
]

# news PCs: 4 PCA components + a presence flag
_NEWS_FEATURES: List[str] = [
    "news_pc1", "news_pc2", "news_pc3", "news_pc4",
    "news_present",
]

# old alias, kept around in case something external still imports it. # KNOW
_BASE_FEATURES: List[str] = _OHLCV_FEATURES + _FUNDAMENTAL_FEATURES

# this is just the default (everything incl news). the dataclass actually
# picks the real list from feature_set in __post_init__.
STOCK_FEATURE_COLUMNS: List[str] = _OHLCV_FEATURES + _FUNDAMENTAL_FEATURES + _NEWS_FEATURES

# macro panel: same kind of features but no fundamentals/news/options. I keep
# seasonality here so the macro encoder can pick up calendar effects in regime.
# dropped sma_50_rel like the stocks (redundant with macd_rel + rsi_norm).
MACRO_FEATURE_COLUMNS: List[str] = [
    "open_ret", "high_ret", "low_ret", "close_ret",
    "intraday_range", "body",
    "log_price_z", "price_percentile", "log_volume_z",
    "rsi_norm", "macd_rel", "ema_20_rel",
    # engineered + seasonality (same as stocks)
    "ret_5d_cum", "ret_20d_cum", "mom_12_1", "vol_20d", "dist_52w_high",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_month_end",
]


# ---------------------------------------------------------------------------
# the FINAL dataset (data/final_data/).
# fresh rebuild with a richer, already-standardised schema (std~1, clipped +/-5),
# 445 tickers, 2013-2025. forward-return labels ship as RAW returns
# (label_5d/label_30d/label_60d), then prepare_final.py renames the chosen
# horizon to return_<h>d and DROPS the others so there's no look-ahead leak.
# feature_set='final' points Config at these parquets and the lists below.
#
# left out on purpose:
#   - raw price levels (open/high/low/close/volume/vwap, sma_*, ema_*) are
#     non-stationary, so I keep the stationary versions instead (log_price_z,
#     price_to_sma*, *_ret).
#   - macd_line/macd_signal/bb_upper/bb_lower: redundant with macd_hist/bb_width/bb_pct.
#   - log_ret/log_ret_5d/log_ret_21d: basically dupes of norm_ret_*.
#   - year: a trend that just can't generalise to a future test split.
#   - missing_mask: used as the row mask, not a feature.
_FINAL_STOCK_FEATURES: List[str] = [
    # dropped the calendar features (day_of_week, day_of_month, month, quarter,
    # week_of_year) from the per-stock encoder. they're identical across every
    # stock on a given day (market-wide) AND big raw ints (mean |val| ~9.7 vs
    # ~0.76 for the real features), so they carry zero cross-sectional info but
    # dominate the input norm, which pushed every stock's embedding into the
    # same direction (input cosine 0.97 down to 0.32 once dropped). if I want
    # seasonality it goes in the macro/regime path, not here. # WORKING
    # returns / candle shape
    "open_ret", "high_ret", "low_ret", "close_ret", "intraday_range", "body",
    "norm_ret_1d", "norm_ret_5d", "norm_ret_21d",
    "alr_1w", "alr_2w", "alr_1m", "alr_2m",
    # price / volume position
    "log_price_z", "price_percentile", "log_volume_z",
    # technicals
    "rsi_14", "macd_hist", "bb_width", "bb_pct",
    "price_to_sma20", "price_to_sma50", "price_to_sma200",
    "mom_5", "mom_10", "mom_21",
    # options proxy
    "iv_proxy", "iv_percentile",
    # quarterly fundamentals
    "revenue", "net_income", "profit_margin", "debt_to_equity", "roe",
]

# v5 set = the 33 final features + stuff the input-feature screen said actually
# helps (experiments/2026-06-14_input_feature_screen):
#   MOM_252_21   12-1 momentum, the only trending/positive signal, and I didn't have it
#   CNTP_21      fraction of up-days over 21d
#   sector_mom21 sector-mean 21d momentum, strongest single feature I found
#                (Rank-IC -0.033), robust post-2020. it's the cross-sectional
#                "sector tilt" that's basically THE signal here. sector-neutralising
#                kills it, so I add sector info rather than remove it.
#   sector_ret5  sector-mean short return
#   sec_*        11 GICS sector one-hots (explicit membership)
# continuous adds are pre-standardised in the v5 build (refit_zscore=False) so they
# sit at the same scale as everything else (avoids the calendar-feature blowup).
# FIX brittle if a feature_experiments parquet ever ships these un-standardised.
_SECTORS_V5 = ['comms', 'discretionary', 'energy', 'financials', 'health', 'industrials',
               'materials', 'real_estate', 'staples', 'tech', 'utilities']
# default v5 = 37 features: the 33 final + the 4 continuous adds.
# the 11 sec_* one-hots are IN the parquet but off by default. the
# v3.7 vs v5-full vs v5-lean bake-off showed they overfit (full-48 had the worst
# test Sharpe and the biggest val-to-test gap). the time-varying sector aggregates
# (sector_mom21 / sector_ret5) already carry the rotation signal, sector_mom21
# being the strongest single predictor (Rank-IC -0.033).
# KNOW: these adds roughly double the in-sample Rank-IC but DON'T beat the recent
# out-of-sample ceiling, so the bottleneck is the regime/data, not the inputs.
_FINAL_V5_STOCK_FEATURES: List[str] = _FINAL_STOCK_FEATURES + [
    "MOM_252_21", "CNTP_21", "sector_mom21", "sector_ret5",
]
# here if you want to play with them: cfg.stock_feature_columns += these
_V5_SECTOR_ONEHOTS: List[str] = [f"sec_{s}" for s in _SECTORS_V5]

# macro panel for the final dataset, only columns that actually exist in
# macro.parquet (no calendar/options/fundamentals/labels in there).
_FINAL_MACRO_FEATURES: List[str] = [
    "open_ret", "high_ret", "low_ret", "close_ret", "intraday_range", "body",
    "norm_ret_1d", "norm_ret_5d", "norm_ret_21d",
    "alr_1w", "alr_2w", "alr_1m", "alr_2m",
    "log_price_z", "price_percentile", "log_volume_z",
    "rsi_14", "macd_hist", "bb_width", "bb_pct",
    "price_to_sma20", "price_to_sma50", "price_to_sma200",
    "mom_5", "mom_10", "mom_21",
]


@dataclass
class Config:
    # ---- Data ----------------------------------------------------------------
    # the dataset is picked by two knobs:
    #   feature_set in {'ohlcv', 'fundamentals', 'news'}
    #   target_horizon in {5, 30}
    # together they map to one of 6 parquets:
    #   data/stocks_<feature_set>_<horizon>d.parquet
    # e.g. feature_set='news', target_horizon=5 gives data/stocks_news_5d.parquet
    #
    # old code used use_news (True/False) to pick. still works for back-compat,
    # it just maps to ('news', 5) / ('fundamentals', 5).
    feature_set: str = "news"                # 'ohlcv' | 'fundamentals' | 'news'
    target_horizon: int = 5                  # 5 | 30
    use_news: bool = True                    # legacy alias, gets overridden by feature_set in __post_init__
    stocks_parquet: Optional[Path] = None    # resolved in __post_init__
    macro_parquet:  Path = Path("data/macro_signals.parquet")
    # used to be 504 (~2y) for fast iteration, now None = full ~13 years. this is
    # the single biggest variance-reduction lever on the val Rank-IC I select on,
    # because going from ~75 val days (504) to ~340 (full) roughly halves the SE
    # of the mean Rank-IC. pass --last-n-days 504 for a quick smoke test.
    last_n_days: Optional[int] = None
    # optional calendar slice (e.g. only train/test on 2012-2019). both are
    # 'YYYY-MM-DD' strings or None. applied before last_n_days.
    start_date: Optional[str] = None
    end_date:   Optional[str] = None
    label_col: str = "label_up"              # binary label, still loaded for the direction-acc metric. # KNOW newer datasets dropped it, data.py falls back to close_ret>0
    mask_col: str = "missing_mask"           # 1 == missing

    # ---- regression target, read straight from the parquet -------------------
    # the build now writes the forward-return label as a column per parquet:
    #   return_5d  for stocks_*_5d.parquet,  return_30d for stocks_*_30d.parquet
    # data.py just reads it. no more in-memory rolling. # WORKING
    # KNOW: this only matters once the parquet actually has the column, older
    # builds don't and data.py will raise.
    # regression_target_col is auto-resolved from target_horizon in __post_init__,
    # override only if you want a custom one.
    regression_target_col: Optional[str] = None
    target_mode: str = "regression"
    # 5d cumulative raw return has std ~0.04-0.05, so x20 brings it to ~1.
    # AdamW is happier there.
    regression_scale: float = 20.0

    # --- target centering (stops mean-collapse) ------------------------------
    # even after per-ticker z-scoring, a single stock's close_ret has a tiny but
    # nonzero train-period mean. without centering the model can just output that
    # constant and call it a day, the classic "mean collapse" (pred_mean drifts
    # to the train mean, pred_std goes to 0, dir_acc goes to 0.5).
    #
    #   "none"          no centering (original v2)
    #   "train_mean"    subtract per-ticker train-split mean [default]. preds are
    #                   in centered units but get un-centered back to real return
    #                   for dashboards/CSVs.
    #   "cross_section" per-day cross-sectional mean (predicts relative outperf vs
    #                   the universe that day). this changes what the target MEANS,
    #                   so only use it if that's what you actually want.
    target_center: str = "train_mean"

    # cross-sectional target, amplifies the signal and makes it a ranking problem.
    # when != 'none', data.prepare() swaps the raw scaled return for a per-day
    # cross-sectional transform across the valid stocks and forces
    # regression_scale=1.0 (transform's already O(1)). this strips out the market
    # common factor (huge but unpredictable cross-sectionally), makes a constant
    # prediction worthless, and self-normalises so 5d and 30d both land at std~1.
    #   'none'             legacy scaled+centered raw return
    #   'zscore' (DEFAULT) (r - mean)/std across stocks each day
    #   'rank'             percentile rank in [0,1], recentred to [-0.5,+0.5]
    #   'rank_and_zscore'  gaussianised rank (robust + unit-variance)
    cross_sectional_target: str = "zscore"

    # filled in by data.prepare(). per-ticker means I use to un-center preds for display.
    target_center_values: List[float] = field(default_factory=list)

    # --- loss type -----------------------------------------------------------
    #   "mse"   plain squared error (original v2)
    #   "huber" quadratic small / linear large, robust to outlier days, good
    #           default for noisy returns
    loss_type: str = "huber"
    huber_delta: float = 1.0     # the kink, in std-units (~1 sigma)

    # ---- model variant (for ablations) --------------------------------------
    # "tagc"          full pipeline (default, the thesis novelty)
    # "no_gru"        GraphConstructor but GRU step replaced with h = z.
    #                 asks: does the recurrent node memory actually help?
    # "static_graph"  same encoder + GAT + head but the [N,N] adjacency is FIXED
    #                 (sector-based, loaded from static_graph_path).
    #                 asks: does the LEARNED graph beat a hand-coded sector one?
    # "no_graph"      encoder then head only, no GraphConstructor/GAT.
    #                 asks: does the graph beat a vanilla transformer baseline?
    model_variant: str = "tagc"
    # needed when model_variant == "static_graph". a parquet with a square
    # ticker-by-ticker adjacency (indexed + columned by ticker).
    static_graph_path: Optional[Path] = None

    # ---- walk-forward online fine-tuning ------------------------------------
    online_lr: float = 1e-5                  # way smaller than training lr
    online_aux_weight: float = 0.3           # multi-target loss during walk-forward
    online_save_every: int = 0               # 0 = never save updated weights; N = every N days

    # the target stock. model predicts this ticker only, the other 97 are graph
    # context. loss + metrics + position sizing all run on this index.
    target_ticker: str = "AAPL"

    # filled in by __post_init__ from use_news. set directly for a custom subset.
    stock_feature_columns: Optional[List[str]] = None
    macro_feature_columns: List[str] = field(
        default_factory=lambda: list(MACRO_FEATURE_COLUMNS))

    # already normalised in the build, so don't refit the scaler.
    refit_zscore: bool = False

    # ---- splits (chronological, on the last_n_days slice) -------------------
    train_frac: float = 0.70
    val_frac:   float = 0.10                 # test = 1 − train − val = 0.20

    # ---- windowing -----------------------------------------------------------
    # bumped window 90 to 180 (~8.5 trading months of context). had to bump
    # patch_len too to keep n_patches sane: at W=180, patch_len=10 gives 18 patches
    # per stock (same 18 as 90/5). same token count, just each token covers 10
    # days now instead of 5, coarser, matches the 5d/30d targets better.
    # change window and patch_len together, n_patches has to come out integer.
    window: int = 180                        # was 90
    patch_len: int = 10                      # was 5
    patch_stride: int = 10                   # non-overlapping
    # the n_patches assertion is checked at runtime in PatchTSTEncoder.

    # ---- stock encoder (PatchTST) -------------------------------------------
    # thesis-depth sizes. roughly between published transformer SOTA on daily
    # stock prediction (DTML is d=256/8 layers, PatchTST ref is d=128/3-4), so I
    # sit in the middle.
    #
    # bumped these back up after v3.2 had shrunk them to fight overfit on a tiny
    # val window. now that I run full 13y history + EMA + burn-in + smoothed
    # early-stop, the bigger model settles fine without spiking at epoch 1.
    #
    # if val_rank_ic crashes: bump dropout_tst to 0.25 or weight_decay to 5e-3
    # before you start cutting depth. # TODO
    d_model: int = 128                       # was 64
    n_heads_tst: int = 8                     # 16-dim per head (was 4 x 16)
    n_layers_tst: int = 4                    # was 2
    ffn_mult: int = 2
    dropout_tst: float = 0.20                # was 0.15, for the extra depth
    # attention pool over patches (learned query) instead of plain mean-pool, so
    # the model can lean on recent days. basically what self-attention is for.
    # KNOW: only kicks in when use_attn_pool is True, else it's plain mean-pool.
    use_attn_pool: bool = True

    # ---- macro encoder (mini-PatchTST giving regime vector r_t) -------------
    # smaller than the stock encoder on purpose, only 10 macro tickers to sum
    # up, doesn't need the same firepower. d_macro stays separate from d_model
    # since they're different feature spaces.
    d_macro: int = 64
    n_heads_macro: int = 4
    n_layers_macro: int = 2
    dropout_macro: float = 0.1
    use_film: bool = True                    # FiLM the stock tokens on r_t. KEEP this,
                                              # lets the encoder shift its repr in
                                              # different macro regimes.
    # dropped the macro-conditioned eps threshold and macro-FiLM on Q/K. each was
    # a tiny zero-init nudge to the graph similarity surface that, with how noisy
    # the gradient is, just added variance and no signal. macro still feeds in via
    # FiLM on stock tokens (the strongest path), which I keep.
    use_macro_eps: bool = False
    use_macro_kq: bool = False

    # ---- graph constructor ---------------------------------------------------
    # h_t = GRUCell(z_t, h_{t-1})                       (the novelty, GRU memory)
    # g_t = graph_proj(h_t)                              (decoupled similarity space)
    # sim = (W_q g_t) . (W_k g_t)^T / sqrt(d_attn)       (Q/K bilinear similarity)
    # eps_t = softplus(eps_base + W_eps . r_t)           (macro-conditioned threshold)
    # A_new = sigmoid((sim - eps_t) / tau)               (soft eps-sphere)
    # A_t = alpha . A_{t-1} + (1 - alpha) . A_new        (edge memory, GRU idea on edges)
    # has to match d_model so the no-GRU variants (no_gru, static_graph) can feed
    # h=z (shape [N, d_model]) straight into the GAT's first layer.
    # if you change d_model, change this too. # KNOW
    gru_hidden: int = 128                    # was 64, tracks d_model
    # toggle the GRU step in the GraphConstructor. False = h is z (no temporal
    # memory in the constructor). used by the no_gru ablation.
    use_gru: bool = True
    graph_proj_dim: int = 64                 # separate similarity-space dim
    eps_init: float = 0.0                    # bilinear sim starts centred at 0
    tau: float = 1.0                         # soft-sigmoid temperature
    # dropped this 0.3 to 0.1.  A_t = alpha*A_{t-1} + (1-alpha)*A_new
    # at 0.3 the graph was basically frozen across the test split (Pearson corr
    # between first-day and last-day adjacency = 1.00 over 4 months). at 0.1 each
    # day only inherits 10% from yesterday so 90% is fresh.
    # set 0.0 to kill memory entirely.
    edge_ema: float = 0.1
    # 0 = pure soft eps-sphere (dense, sigmoid never hits zero so every pair has
    # some weight). >0 = also keep only top-k neighbours per row (applied AFTER
    # the edge-EMA so the smoothing still feeds the cut). default 5 means each
    # stock attends to ~5% of the universe. GAT still respects edge WEIGHTS
    # (alpha = softmax(attn)*A) so even inside the top-5 it's proportional to similarity.
    topk: int = 5
    # was 5e-4, bumped to 2e-3 to keep the graph sparser, because the richer input
    # set tempts the constructor into fitting spurious correlations.
    l1_edge_weight: float = 2e-3

    # save the constructed adjacencies so I can poke at them later.
    save_graphs_every: int = 1               # 1 = every test day, 0 = never
    graphs_subdir: str = "graphs"

    # ---- GAT -----------------------------------------------------------------
    # 4 heads x 32-dim each (d_model=128 / 4).
    # HPO knocked n_layers_gat 3 to 2: the autoencoder probe + M3 audit showed a
    # 3-layer GAT over-smooths (post-graph embedding cosine 0.02 up to 0.74, a 3-hop
    # field ~125 of 445 nodes) and washes out stock identity. Optuna agreed,
    # n_layers_gat=2 had the best mean AND single-best val Rank-IC. 2 hops keeps
    # relational context without collapsing identity. dropout landed at 0.28. # WORKING
    n_heads_gat: int = 4
    n_layers_gat: int = 2                    # HPO: 3 to 2 (over-smoothing fix)
    dropout_gat: float = 0.28                # HPO (was 0.20)

    # ---- prediction head -----------------------------------------------------
    # 2-layer MLP with MC dropout. wider than v3.2 (128 vs 64) to match the
    # encoder's d_model so the gradient path stays uniform.
    head_hidden: int = 128                   # was 64
    head_dropout: float = 0.34               # HPO (was 0.30); MC dropout at inference
    mc_samples: int = 30

    # ---- fixes for the "deep model that won't learn" failure mode -----------
    # residual/skip from the encoder output (z) to the head input, skipping the
    # GAT. without it AAPL's own 60-day signal gets diluted to ~9% in each
    # post-GAT embedding (1 self + 10 neighbours). with it the graph just adds
    # enrichment on top of the untouched encoder output. standard transformer/GNN
    # trick, I'd just been missing it. # WORKING
    # TODO double-check the ~9% dilution number still holds at topk=5, not 10.
    use_residual: bool = True

    # cross-sectional centering of the encoder output, the big one.
    # the per-stock embeddings were directionally collapsed (cosine ~0.99): they
    # split into a big SHARED vector (market common factor + leftover common-mode)
    # plus a small per-stock bit. cross-sectional models have to strip the common
    # factor. when True, TAGC.forward subtracts the cross-stock mean from z each
    # day (z = z - z.mean(dim=0)) so the graph, GAT and residual all see de-meaned,
    # comparable per-stock reps (drives embedding cosine ~0.99 down to ~0). this is
    # the representation-level half, and dropping the calendar features above is
    # the input-level half. # WORKING
    cross_sectional_norm: bool = True

    # learnable per-stock identity embedding (nn.Embedding(N, d_model)) added after
    # the encoder. PatchTST is permutation-invariant in stock identity, so without
    # this two stocks with similar 60-day windows give near-identical embeddings,
    # which is exactly why the graph clusters everything. N x d_model params is nothing.
    use_stock_id_emb: bool = True

    # True (default): ID embedding added BEFORE the residual so it flows through
    # both the GAT and straight to the head. that makes the ID a per-stock bias
    # that can short-circuit the encoder when the target's uncentered, model just
    # learns "for AAPL output train-mean". False adds the ID only to the encoder
    # INPUT, not the residual branch, good diagnostic when you suspect mean
    # collapse. pair with target_center='train_mean' for the cleanest fix.
    stock_id_in_residual: bool = True

    # ---- training ------------------------------------------------------------
    # chunked TBPTT: accumulate loss over tbptt_steps days, then one backward +
    # optimizer step + detach the GRU state. tbptt_steps=1 = pure online (super
    # noisy on small data). default 10 = effective batch size 10.
    # KNOW: this is the chunk length for TBPTT, not a true minibatch, the days
    # inside a chunk still go through the GRU in order.
    # training runs at least min_epochs, then keeps going while val_loss improves
    # and stops after patience epochs of no improvement. hard cap at max_epochs.
    # longer min + bigger patience because the larger feature space + smaller
    # model needs more steps to settle. shorter warmup so the optimizer gets real
    # gradients before patience can kick in.
    min_epochs: int = 30                     # was 20, give it time
    max_epochs: int = 80                     # was 60
    patience: int = 10                       # was 5, survive a noisy plateau
    # early-stop tracks Rank-IC now (used to be val_mse). Rank-IC sits around
    # 0.01-0.05 so a 1e-3 threshold is about the same relative sensitivity 1e-4
    # had against MSE. higher is better here (maximise). also smoothed over a
    # rolling window, see smooth_window.
    early_stop_delta: float = 1e-3
    # rolling-window length (epochs) for the smoothed val_rank_ic that drives
    # early-stop. single-epoch readings have SE ~0.05 on small val sets, which is
    # about the size of the signal I'm chasing.
    smooth_window: int = 5

    # for the first best_burnin_epochs, refuse to call anything a new "best" even
    # if val_rank_ic spikes, optimizer's still in warmup and the val reading is
    # mostly init noise.
    best_burnin_epochs: int = 5

    # -- weight EMA -----------------------------------------------------------
    # standard fix for the epoch-1-best pathology in noisy-signal land: keep a
    # shadow set of weights that lag the live ones via EMA. eval + best.pt use
    # the EMA copy. # WORKING
    use_weight_ema: bool = True
    # decay=0.995 gives an effective window ~200 steps ~8 epochs at last_n_days=504
    # / TBPTT=10. full-history runs have more steps/epoch so the same decay smooths
    # over fewer epochs, which is fine.
    ema_decay: float = 0.995

    # -- optimizer ------------------------------------------------------------
    # peak LR + warmup tuned for the noisy-ranking regime: smaller peak + longer
    # warmup so the optimizer can't take big destabilising steps early. (was
    # lr=3e-4 / warmup 0.05, fine for plain MSE on a single target but too hot
    # for ListMLE.)
    # lr 1e-4 to 3e-4. Optuna's cheap-proxy winner was ~1.7e-3 but that COLLAPSED
    # at full budget (pred_std went to 0, training flatlined), classic short-horizon
    # HPO trap where 5-epoch trials over-reward an aggressive LR. a full-data
    # lr-stability sweep showed 3e-4 is the highest LR that keeps a healthy spread
    # (pred_std ~0.01) AND the best val Rank-IC; 6e-4+ already collapse. 20% warmup
    # + cosine decay smooth it out. # WORKING
    lr: float = 3e-4                         # stability-validated (was 1e-4; HPO's 1.7e-3 collapsed)
    warmup_frac: float = 0.20                # was 0.05, let the optimizer settle
    # weight_decay 5e-4 to 1e-4. the lr-stability sweep paired stable lr=3e-4 with
    # wd=1e-4 (HPO's 3e-5 was too light and let preds collapse).
    weight_decay: float = 1e-4               # stability-validated (was 5e-4)
    # was 5.0 (loosened from 1.0 to keep directional gradient under the old MSE
    # loss). under ListMLE the head's gradient is naturally smaller so back to 1.0,
    # which stops big early steps locking the model into a lucky-noise region.
    grad_clip: float = 1.0
    tbptt_steps: int = 10

    # LR multiplier for GraphConstructor params. was 1.5 to rebalance the
    # encoder-vs-graph gradient asymmetry under old MSE. under ListMLE every
    # module gets gradient through the cross-sectional comparison so the asymmetry
    # is gone, flat at 1.0.
    graph_lr_scale: float = 1.0

    # LR multiplier for the MacroEncoder. was 3.0 to make up for the macro paths
    # being zero-init under old MSE. under ListMLE all paths get gradient, so drop
    # to 1.0 to avoid amplifying early instability.
    macro_lr_scale: float = 1.0

    # log L2 grad norm per top-level submodule each epoch. cheap check that the
    # graph constructor is actually getting gradient and not vanishing next to the
    # encoder/head. False to silence.
    log_grad_norms: bool = True

    # aux multi-target loss: also train the head on the other 97 stocks (mean
    # BCE), skipping the target so target_loss stays clean. those 97 extra losses
    # give the shared encoder + graph constructor a lot more gradient per day.
    # catch: on small data a big aux_loss_weight drags the head toward the
    # universe-average and collapses the target's decisions. rough guide:
    #   0.0           pure single-target (original spec, safe default)
    #   0.2 to 0.5    adds encoder signal without drowning target-loss; try with
    #                 --last-n-days 0 (10y) where more data absorbs the regularising
    #   >= 1.0        aux dominates, only if you actually want multi-target preds
    #                 (model becomes usable for any stock at inference)
    # dropped in the v3.2 cleanup. the listwise rank loss (aux_rank_weight)
    # already gives the encoder universe-wide gradient, so a per-stock Huber aux
    # on top was redundant. left at 0 for back-compat, the train loop ignores it. # KNOW
    aux_loss_weight: float = 0.0

    # which TYPE of loss to optimise.
    #   'combined' (default) = huber_weight*Huber(target) + aux_rank_weight*ListMLE(universe)
    #                          means "predict value AND rank well". ratio set by
    #                            (huber_weight, aux_rank_weight). see
    #                            experiments/experiment_loss_functions.
    #   'huber'              = Huber(target) only, "nail the target's exact return"
    #   'listmle'            = ListMLE(universe) only, "rank right, ignore magnitude"
    #   'learnable'          = Huber + ListMLE with LEARNABLE weights (Kendall et al.
    #                          2018 homoscedastic-uncertainty weighting). two learnable
    #                          log-variances s_h, s_r on the model;
    #                          loss = sum_k [exp(-s_k)*L_k + s_k], s_k = log sigma_k^2. the
    #                          optimiser tunes the balance itself, no hand-set ratio.
    #                          see model.TAGC.log_sigma_*.
    #   'rank_mag' (DEFAULT) = literature-backed combined magnitude+ranking loss, both
    #                          terms computed UNIVERSE-WIDE per day on the
    #                          cross-sectional z-score target (convex split):
    #                              loss = rank_weight*ListMLE(universe)   [ranking]
    #                                   + mag_weight *Huber(universe)     [magnitude]
    #                                   + l1_edge_weight*|A|
    #                          why: ListMLE fixes the cross-sectional ORDER but leaves
    #                          the SCALE under-determined (pure listwise = grossly
    #                          uncalibrated levels). the universe-wide Huber anchors
    #                          the scale to the standardized target (a pred reads as a
    #                          standardized expected return, not just a rank) and shrugs
    #                          off the fat tails. this is the regression+ranking template
    #                          from Feng et al. RSR (TOIS 2019) / Sawhney et al. STHAN-SR
    #                          (AAAI 2021), with ListMLE (Xia et al., ICML 2008), Huber
    #                          (1964), and the magnitude-anchor + ranking pattern of
    #                          Kwiatkowski & Chudziak (CIKM '25). no direction/BCE term,
    #                          because the listwise order already covers sign, so it'd be
    #                          a redundant extra weight. the big difference vs legacy
    #                          'combined': the Huber is over the WHOLE cross-section, not
    #                          one ticker, so a constant mean can't minimise it (the old
    #                          collapse). pair with cross_sectional_target != 'none'.
    #                          (optional upgrade: the 'learnable' mode, Kendall-Gal-
    #                          Cipolla CVPR 2018 uncertainty weighting.) # WORKING
    #   'rank_dir'           = the v3.6 cross-sectional ranking objective:
    #                          1.0·ListMLE(universe) + direction_bce_weight·BCE(up vs
    #                          down) + l1_edge_weight·|A|. no magnitude term. the head's
    #                          single score does double duty for ranking (its order) and
    #                          direction (its sign). pair with cross_sectional_target != 'none'.
    # train.py uses this to gate the loss terms. doesn't touch the architecture,
    # just the supervision (and for 'learnable', two scalar params).
    loss_kind: str = "rank_mag"

    # weight on the DIRECTION BCE aux in 'rank_dir'. label = (y_reg > 0) = "above
    # the cross-sectional median today" (the target's zero-mean per day). logit is
    # the head's score.
    direction_bce_weight: float = 0.3

    # weight on the Huber (value) term in 'combined'. ListMLE (ranking) is weighted
    # by aux_rank_weight below, together they set the Huber:ListMLE ratio. examples
    # from the loss experiment:
    #   huber_weight=1, aux_rank_weight=2  gives 1:2 (rank-leaning)
    #   huber_weight=2, aux_rank_weight=1  gives 2:1 (value-leaning)
    # ignored by 'huber'/'listmle'/'learnable'.
    huber_weight: float = 1.0

    # weights for the universe-wide 'rank_mag' loss, a convex split that leans a
    # bit toward ranking (the portfolio trades on ORDER, not level):
    #     loss = rank_weight*ListMLE + mag_weight*Huber       (rank+mag = 1.0)
    # both run over every valid stock each day on the cross-sectional z-score
    # target. default lambda=0.6 on ranking (grid {0.5, 0.7} per Kwiatkowski &
    # Chudziak CIKM '25; pick the final lambda on the PORTFOLIO metric you actually
    # trade, since max-IC != max-Sharpe). ListMLE's loss VALUE (~5) dwarfs Huber's
    # (~0.35) but their GRADIENTS are comparable (~0.049 vs ~0.032), so at
    # lambda=0.6 the Huber still drives ~30% of the gradient, a real magnitude
    # anchor. mag_weight=0 = pure ranking. separate from huber_weight/aux_rank_weight,
    # which only the legacy single-target modes read.
    rank_weight: float = 0.6
    mag_weight:  float = 0.4

    # ListMLE listwise-ranking loss over all valid stocks per day:
    #     L_rank = -sum (s_i - logsumexp(s_i, s_{i+1}, ..., s_n))
    # under the true order by y_reg. pushes the encoder + head to predict
    # cross-sectional ORDER, not absolute return. cheap (O(N log N)/day) and gives
    # gradient even on days where the point signal is ~0 (most days, honestly).
    # 0.3 ~ same magnitude as L_main on well-mixed days.
    aux_rank_weight: float = 0.3

    # --- asymmetric (direction-aware) MSE ------------------------------------
    # when sign(pred) != sign(y) the squared error gets x(1 + dir_loss_weight).
    # 0 = plain MSE; 1 = wrong-direction errors cost 2x the right-direction same
    # magnitude; 2 = 3x.
    # dir_hinge_weight adds a relu(-pred*y) term on top, which gives a clean
    # directional gradient even when |pred|,|y| are both tiny (where (pred-y)^2
    # alone barely moves). 0 to disable.
    # dropped in v3.2. with listwise ranking + Huber + EMA + burn-in these
    # asymmetric/hinge bits were just noise. kept at 0 for back-compat. # KNOW
    dir_loss_weight:  float = 0.0
    dir_hinge_weight: float = 0.0

    device: str = "cpu"
    seed: int = 42

    # ---- derived shapes (filled in by the data loader at runtime) ------------
    n_stocks: int = 0
    n_macro:  int = 0
    target_idx: int = -1                     # set once the tickers are known

    def __post_init__(self):
        # -- resolve dataset path from (feature_set, target_horizon) -------
        # legacy use_news still wired: True gives 'news', False 'fundamentals'
        if self.feature_set is None:
            self.feature_set = 'news' if self.use_news else 'fundamentals'
        assert self.feature_set in ('ohlcv', 'fundamentals', 'news', 'final', 'final_v5'), \
            f"feature_set must be one of ohlcv|fundamentals|news|final|final_v5, got {self.feature_set}"
        # validate the model variant, used to silently mis-wire on a typo. # KNOW
        _VARIANTS = ('tagc', 'no_gru', 'no_gat', 'static_graph', 'no_graph', 'random_graph')
        assert self.model_variant in _VARIANTS, \
            f"model_variant must be one of {_VARIANTS}, got {self.model_variant!r}"
        assert self.cross_sectional_target in ('none', 'zscore', 'rank', 'rank_and_zscore'), \
            f"cross_sectional_target invalid: {self.cross_sectional_target!r}"

        if self.feature_set in ('final', 'final_v5'):
            # the final dataset (plus the v5 augmented variant). 5/30/60-day horizons.
            assert self.target_horizon in (5, 30, 60), \
                f"target_horizon must be 5, 30 or 60 for 'final', got {self.target_horizon}"
            if self.stocks_parquet is None:
                if self.feature_set == 'final_v5':
                    self.stocks_parquet = Path(
                        f"data/feature_experiments/stocks_{self.target_horizon}d_v5.parquet")
                else:
                    self.stocks_parquet = Path(
                        f"data/final_data/stocks_{self.target_horizon}d.parquet")
            # point macro at the final panel unless the caller already set one
            if str(self.macro_parquet) in ("data/macro_signals.parquet", "."):
                self.macro_parquet = Path("data/final_data/macro.parquet")
            if self.stock_feature_columns is None:
                self.stock_feature_columns = list(
                    _FINAL_V5_STOCK_FEATURES if self.feature_set == 'final_v5'
                    else _FINAL_STOCK_FEATURES)
            # swap in the final macro list if it's still the legacy default
            if list(self.macro_feature_columns) == list(MACRO_FEATURE_COLUMNS):
                self.macro_feature_columns = list(_FINAL_MACRO_FEATURES)
            # auto-scale so the forward return lands at std ~1 (Huber likes that).
            # return_30d std~0.095 gives x10; return_60d std~0.134 gives x7.5;
            # return_5d std~0.040 gives x25. only if the default's untouched.
            if self.regression_scale == 20.0:
                self.regression_scale = {5: 25.0, 30: 10.0, 60: 7.5}[self.target_horizon]
        else:
            assert self.target_horizon in (5, 30), \
                f"target_horizon must be 5 or 30, got {self.target_horizon}"
            if self.stocks_parquet is None:
                self.stocks_parquet = Path(
                    f"data/stocks_{self.feature_set}_{self.target_horizon}d.parquet"
                )
            # -- feature list, match the dataset's actual columns ----
            # ohlcv:        OHLCV + technicals + options + calendar
            # fundamentals: above + quarterly fundamentals
            # news:         above + news PCs
            if self.stock_feature_columns is None:
                feats = list(_OHLCV_FEATURES)
                if self.feature_set in ('fundamentals', 'news'):
                    feats += _FUNDAMENTAL_FEATURES
                if self.feature_set == 'news':
                    feats += _NEWS_FEATURES
                self.stock_feature_columns = feats

        # -- regression target column ---------------------------------------
        # auto-picks return_5d / return_30d / return_60d from the parquet
        if self.regression_target_col is None:
            self.regression_target_col = f"return_{self.target_horizon}d"

        # coerce string paths so CLI overrides like --stocks-parquet ".." work
        self.stocks_parquet = Path(self.stocks_parquet)
        self.macro_parquet  = Path(self.macro_parquet)

    @property
    def n_features_stock(self) -> int:
        return len(self.stock_feature_columns)

    @property
    def n_features_macro(self) -> int:
        return len(self.macro_feature_columns)

    @property
    def n_patches(self) -> int:
        assert (self.window - self.patch_len) % self.patch_stride == 0, \
            f"window-patch_len ({self.window - self.patch_len}) must be divisible by stride ({self.patch_stride})"
        return (self.window - self.patch_len) // self.patch_stride + 1
