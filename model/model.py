"""TAGC model. four parts bolted together.

map of the file, top to bottom:

    1   PatchTSTEncoder   per-stock time series, one vector per stock
        MacroEncoder      10 macros, one "what's the regime today" vector
        (the macro vector FiLMs the stock encoder)

    2   GraphConstructor  builds a NxN adjacency that changes every day, with
                          GRU memory

    3   DenseGAT          graph attention over the adjacency from part 2

    4   PredictionHead    little MLP w/ dropout, one number per stock

the encoder runs independently per ticker, so no cross-stock leakage inside it.
cross-ticker info only flows through parts 2 + 3, that's basically the thesis.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

log = logging.getLogger("tagc.model")


# shared patch transformer, both the stock and macro encoders use it
class _PatchTransformer(nn.Module):
    """one PatchTST, reused for both the stock and macro encoders.

    idea's from the PatchTST paper (Nie et al. 2023). instead of one token per
    day (seq len 90, expensive), chop the window into patches of patch_len days
    and make each patch a token. same window, ~9x fewer tokens, so faster and
    overfits less.

    x: [R, W, F]   R = how many things we're encoding (N stocks, or 10 macros)
                   W = window length in days (90)
                   F = num features
    z: [R, d_model]   one vector per row, summary of its W-day window
    """

    def __init__(self, *, d_model: int, n_heads: int, n_layers: int,
                 ffn_mult: int, dropout: float,
                 patch_len: int, patch_stride: int, n_patches: int,
                 n_features: int, d_cond: int = 0,
                 use_attn_pool: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_stride
        self.n_patches = n_patches
        self.d_model = d_model
        self.d_cond = d_cond
        self.use_attn_pool = use_attn_pool

        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # FiLM near-identity at init. biases zero (so gamma=beta~0, tokens barely
        # change at start) but weights get a tiny non-zero init.
        # had a bug here: with zero weights, dloss/dr_t = 0, so the macro encoder
        # got literally no gradient at init. it only started learning once the
        # FiLM weights drifted off zero on their own, super slow. tiny std-1e-3
        # weight init opens the macro gradient path from step 1 and FiLM is still
        # basically identity. # WORKING
        if d_cond > 0:
            self.film_gamma = nn.Linear(d_cond, d_model)
            self.film_beta  = nn.Linear(d_cond, d_model)
            nn.init.normal_(self.film_gamma.weight, std=1e-3); nn.init.zeros_(self.film_gamma.bias)
            nn.init.normal_(self.film_beta.weight,  std=1e-3); nn.init.zeros_(self.film_beta.bias)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        # attention pool: a learnable query attends across the P patches.
        # beats dumb mean-pool since recent patches can get more weight than old
        # ones, which is the whole point of attention. # KNOW only active when
        # use_attn_pool is True, otherwise forward() falls back to mean-pool.
        if use_attn_pool:
            self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self._pool_scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        R, W, F_ = x.shape
        # unfold gives [R, n_patches, F, k], then reshape to [R, P, k*F]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(R, self.n_patches, self.patch_len * F_)
        tokens = self.patch_proj(patches) + self.pos_emb       # [R, P, d]

        if self.d_cond > 0 and cond is not None:
            # cond is [d_cond], same for every row, broadcast to [1, 1, d_model]
            gamma = self.film_gamma(cond).view(1, 1, -1)
            beta  = self.film_beta(cond).view(1, 1, -1)
            tokens = tokens * (1.0 + gamma) + beta             # identity at init

        z = self.encoder(tokens)                                # [R, P, d]

        if self.use_attn_pool:
            # learned-query pool: weights = softmax(q . k_i / sqrt(d))
            q = self.pool_query.expand(z.size(0), -1, -1)       # [R, 1, d]
            scores = (q @ z.transpose(-1, -2)) * self._pool_scale  # [R, 1, P]
            weights = F.softmax(scores, dim=-1)                  # [R, 1, P]
            pooled = (weights @ z).squeeze(1)                    # [R, d]
        else:
            pooled = z.mean(dim=1)                               # [R, d]

        return self.norm(pooled)                                 # [R, d_model]


# macro encoder -> regime vector r_t
class MacroEncoder(nn.Module):
    """squashes the 10 macro tickers into one regime vector r_t.

    the 10 macros are totally different signals (SPY = equity beta, VIX = stress,
    IRX = short rates, GLD/USO = commodities, UUP = USD...). mean-pooling them
    would be dumb, you'd average a vol spike with a rate cut and lose the lot.
    so instead attention-pool with a learned query and let the model pick which
    macros matter each day (VIX in stress, IRX in rate shocks, etc).

    out is one d_macro vector that goes into the stock encoder via FiLM, so the
    stock encoder can shift depending on the regime.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = _PatchTransformer(
            d_model=cfg.d_macro,
            n_heads=cfg.n_heads_macro,
            n_layers=cfg.n_layers_macro,
            ffn_mult=cfg.ffn_mult,
            dropout=cfg.dropout_macro,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            n_patches=cfg.n_patches,
            n_features=cfg.n_features_macro,
            d_cond=0,
            use_attn_pool=cfg.use_attn_pool,
        )
        # attention pool across the macro tickers
        self.ticker_query = nn.Parameter(torch.zeros(1, cfg.d_macro))
        nn.init.trunc_normal_(self.ticker_query, std=0.02)
        self._ticker_scale = 1.0 / math.sqrt(cfg.d_macro)

    def forward(self, x_macro: torch.Tensor) -> torch.Tensor:
        per_ticker = self.backbone(x_macro)               # [M, d_macro]
        # pool across the M=10 macro tickers
        scores = (self.ticker_query @ per_ticker.t()) * self._ticker_scale  # [1, M]
        weights = F.softmax(scores, dim=-1)               # [1, M]
        return (weights @ per_ticker).squeeze(0)          # [d_macro]


# stock encoder (PatchTST + FiLM on the macro vector)
class PatchTSTEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_cond = cfg.d_macro if cfg.use_film else 0
        self.backbone = _PatchTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads_tst,
            n_layers=cfg.n_layers_tst,
            ffn_mult=cfg.ffn_mult,
            dropout=cfg.dropout_tst,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            n_patches=cfg.n_patches,
            n_features=cfg.n_features_stock,
            d_cond=d_cond,
            use_attn_pool=cfg.use_attn_pool,
        )

    def forward(self, x: torch.Tensor, r_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.backbone(x, cond=r_t)          # [N, d_model]


# graph constructor: GRU memory + bilinear similarity + edge EMA
class GraphConstructor(nn.Module):
    """the actual thesis contribution.

    builds the NxN stock adjacency that changes every day.

    each day:
      1. take the N stock vectors z from the encoder
      2. update a per-stock memory h with a GRU (so we keep recent context)
      3. project h into a smaller similarity space g
      4. pairwise sims  sim_ij = (W_q g_i) . (W_k g_j) / sqrt(d)
      5. soft threshold  A_new = sigmoid((sim - eps) / tau)
      6. smooth w/ yesterday  A = alpha*A_prev + (1-alpha)*A_new
      7. keep top-k edges per row (sparse)

    why each bit's here:
    - GRU memory: graph doesn't reset daily, keeps continuity of "which stocks
      have been moving together lately"
    - separate similarity space (graph_proj): keeps the GRU's job (memory)
      apart from the graph's job (cross-stock similarity)
    - Q/K bilinear: attention-style sim, holds up in high dims where plain L2
      distance gets mushy
    - edge EMA: kills day-to-day jitter, real cross-stock relationships move
      slowly anyway
    - top-k: sparse trains faster and overfits less than dense

    no_gru variant skips step 2 (h = z). static_graph skips this whole thing
    and uses a fixed sector-based A.

    # KNOW the topk mask is applied AFTER the EMA, so a dropped edge can still
    # leak a bit through A_prev for a day or two. probably fine.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # when True, forward() logs sim/g/A_new stats once then flips itself off.
        # train.py sets it at the start of each epoch so I get one "sim stats"
        # line per epoch. lets me check the similarity gate is alive (sim.std > 0)
        # not dead (~0).
        self._log_sim = False
        # GRU's part of the novelty, but skip its params entirely when use_gru is
        # False so the no_gru ablation has a clean lower param count. graph_proj's
        # input dim doesn't change, we always feed it d_model features (z is
        # d_model, GRU output is gru_hidden which == d_model by default).
        if cfg.use_gru:
            self.gru = nn.GRUCell(cfg.d_model, cfg.gru_hidden)
        else:
            self.gru = None

        # separate similarity space
        self.graph_proj = nn.Linear(cfg.gru_hidden, cfg.graph_proj_dim)
        self.q_proj = nn.Linear(cfg.graph_proj_dim, cfg.graph_proj_dim, bias=False)
        self.k_proj = nn.Linear(cfg.graph_proj_dim, cfg.graph_proj_dim, bias=False)
        self.sim_scale = 1.0 / math.sqrt(cfg.graph_proj_dim)

        # eps baseline, learnable additive bias on the sim logits
        self.eps_base = nn.Parameter(torch.tensor(float(cfg.eps_init)))
        self.tau = cfg.tau
        self.topk = cfg.topk

        # macro-conditioned eps. zero-init so eps starts at eps_base
        if cfg.use_macro_eps:
            self.eps_mlp = nn.Linear(cfg.d_macro, 1)
            nn.init.zeros_(self.eps_mlp.weight)
            nn.init.zeros_(self.eps_mlp.bias)

        # macro-FiLM on the Q/K projections. lets the similarity surface bend
        # with the regime: risk-off everyone correlates more (flatter), calm
        # everyone's more idiosyncratic (sharper). zero-init so gamma=beta=0 at
        # start (same as before), then it learns.
        # TODO check if this actually helps or just adds params
        if cfg.use_macro_kq:
            self.gamma_q = nn.Linear(cfg.d_macro, cfg.graph_proj_dim)
            self.beta_q  = nn.Linear(cfg.d_macro, cfg.graph_proj_dim)
            self.gamma_k = nn.Linear(cfg.d_macro, cfg.graph_proj_dim)
            self.beta_k  = nn.Linear(cfg.d_macro, cfg.graph_proj_dim)
            for layer in (self.gamma_q, self.beta_q, self.gamma_k, self.beta_k):
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

        # learnable edge-EMA coeff alpha, sigmoid so it stays in (0, 1)
        if cfg.edge_ema > 0:
            a = min(max(cfg.edge_ema, 0.05), 0.95)
            self.ema_logit = nn.Parameter(torch.tensor(math.log(a / (1.0 - a))))

    def init_hidden(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.cfg.gru_hidden, device=device)

    def init_adjacency(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, n, device=device)

    @property
    def edge_alpha(self) -> Optional[torch.Tensor]:
        if self.cfg.edge_ema > 0:
            return torch.sigmoid(self.ema_logit)
        return None

    def _eps(self, r_t: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.use_macro_eps and r_t is not None:
            return self.eps_base + self.eps_mlp(r_t).squeeze(-1)
        return self.eps_base

    def forward(
        self,
        z: torch.Tensor,
        h_prev: torch.Tensor,
        r_t: Optional[torch.Tensor] = None,
        A_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # GRU's optional: use_gru=False swaps it for the identity, which is what
        # the no_gru ablation uses to isolate what the GRU buys us. just use the
        # encoder output z directly then.
        if self.cfg.use_gru:
            h = self.gru(z, h_prev)
        else:
            h = z   # encoder output is the node feature for the GAT

        g = F.gelu(self.graph_proj(h))
        # drop the shared component across stocks, otherwise it dominates the sims
        g = g - g.mean(dim=0, keepdim=True)   # WORKING

        Q = self.q_proj(g)                                      # [N, d_proj]
        K = self.k_proj(g)                                      # [N, d_proj]
        if self.cfg.use_macro_kq and r_t is not None:
            # gamma/beta are shared across stocks but change day-to-day with r_t
            gQ = self.gamma_q(r_t).unsqueeze(0)                 # [1, d_proj]
            bQ = self.beta_q (r_t).unsqueeze(0)
            gK = self.gamma_k(r_t).unsqueeze(0)
            bK = self.beta_k (r_t).unsqueeze(0)
            Q = Q * (1.0 + gQ) + bQ
            K = K * (1.0 + gK) + bK
        sim = (Q @ K.transpose(0, 1)) * self.sim_scale
        # grab the raw spread BEFORE standardising, this is the real "did
        # centering actually expose pairwise structure?" number. the line below
        # forces sim.std -> ~1 so the post-std value tells you nothing.
        sim_raw_std = float(sim.std().detach())
        # standardise so the tau=1.0 gate has real resolution
        sim = (sim - sim.mean()) / (sim.std() + 1e-6)

        eps_t = self._eps(r_t)
        A_new = torch.sigmoid((sim - eps_t) / self.tau)

        eye = torch.eye(A_new.size(0), device=A_new.device, dtype=A_new.dtype)
        A_new = A_new * (1.0 - eye)

        # gate health, once per epoch. raw std is the real diagnostic (post-std
        # is ~1 by construction); A_new std shows the gate actually resolves edges
        if self._log_sim:
            log.info("  sim stats: RAW std=%.5f (real signal) | post-std=%.3f | "
                     "g std=%.4f | A_new std=%.4f | eps=%+.4f",
                     sim_raw_std, float(sim.std().detach()), float(g.std().detach()),
                     float(A_new.std().detach()),
                     float((eps_t.mean() if eps_t.ndim else eps_t).detach()))
            self._log_sim = False

        if self.cfg.edge_ema > 0 and A_prev is not None:
            alpha = self.edge_alpha
            A = alpha * A_prev + (1.0 - alpha) * A_new
        else:
            A = A_new

        if self.topk and self.topk < A.size(0) - 1:
            _, topi = torch.topk(A, k=self.topk, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(1, topi, 1.0)
            A = A * mask

        return h, A, eps_t


# dense multi-head GAT, attention weighted by the adjacency
class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        Wh = self.W(h).view(N, self.heads, self.head_dim)
        e_src = (Wh * self.a_src).sum(dim=-1)                 # [N, H]
        e_dst = (Wh * self.a_dst).sum(dim=-1)
        e = self.leaky(e_src.unsqueeze(1) + e_dst.unsqueeze(0))   # [N, N, H]

        eye = torch.eye(N, device=A.device, dtype=A.dtype)
        A_full = A + eye
        mask = (A_full > 0).float()
        e = e.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        alpha = F.softmax(e, dim=1)
        alpha = alpha * A_full.unsqueeze(-1)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-9)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)
        return out.reshape(N, self.heads * self.head_dim)


class DenseGAT(nn.Module):
    """graph attention network.

    takes the per-stock features h and the adjacency A from part 2 and lets each
    stock attend to its neighbours. attention is gated by A, so edges not in the
    graph contribute nothing. n_layers_gat hops of propagation (a stock can get
    influenced by friends-of-friends), default 2 since 3 over-smooths.
    # KNOW: this docstring used to say "3 layers", but HPO dropped it to 2.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        # first layer input dim = gru_hidden (== d_model here). later layers
        # output d_model and feed forward.
        dim = cfg.gru_hidden
        for _ in range(cfg.n_layers_gat):
            layers.append(DenseGATLayer(dim, cfg.d_model, cfg.n_heads_gat, cfg.dropout_gat))
            dim = cfg.d_model
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        out = h
        for i, layer in enumerate(self.layers):
            out = layer(out, A)
            if i < len(self.layers) - 1:
                out = F.elu(out)
        return self.norm(out)


# prediction head with MC dropout
class PredictionHead(nn.Module):
    """little MLP that turns each stock's embedding into a number.

    dropout stays on at inference too, we sample mc_samples times, mean is the
    prediction and std is the uncertainty. uncertainty drives position sizing,
    bet bigger when we're more confident.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# full model
class TAGC(nn.Module):
    """router. cfg.model_variant picks which sub-pipeline runs:

        "tagc"          full   encoder, GraphConstructor (GRU+bilinear+EMA), GAT, head
        "no_gru"        ablate encoder, GraphConstructor without GRU (h = z), GAT, head
        "static_graph"  ablate encoder, A_static (sector-based), GAT, head
        "no_graph"      ablate encoder, head

    they all share the same encoder + (maybe) GAT + head, so the only thing that
    changes is what produces the adjacency / node features fed into the GAT.
    keeps the ablation comparison honest.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # no_gru turns off the GRU sub-module. do it BEFORE building submodules
        # so its params never get allocated.
        if cfg.model_variant == "no_gru":
            cfg.use_gru = False
        self.cfg = cfg
        self.macro_encoder = MacroEncoder(cfg)
        self.stock_encoder = PatchTSTEncoder(cfg)
        # constructor (part 2) only for variants that learn a graph: tagc,
        # no_gru, no_gat. (no_gat keeps the constructor but drops the GAT, to
        # test whether the message-passing actually matters.)
        _USES_CONSTRUCTOR = ("tagc", "no_gru", "no_gat")
        self.graph = GraphConstructor(cfg) if cfg.model_variant in _USES_CONSTRUCTOR else None
        # GAT (part 3) runs for everything except no_graph + no_gat
        _NO_GAT = ("no_graph", "no_gat")
        self.gat = DenseGAT(cfg) if cfg.model_variant not in _NO_GAT else None
        self.head = PredictionHead(cfg)

        # per-stock learnable identity, used by every variant
        if cfg.use_stock_id_emb:
            self.stock_id_emb = nn.Embedding(cfg.n_stocks, cfg.d_model)
            nn.init.normal_(self.stock_id_emb.weight, std=0.02)
        else:
            self.stock_id_emb = None

        # layernorm for the residual sum (z + post-GAT embeds)
        self.residual_norm = nn.LayerNorm(cfg.d_model) if cfg.use_residual else None

        # static graph: load once, register as a buffer so .to(device) works
        if cfg.model_variant == "static_graph":
            if cfg.static_graph_path is None:
                raise ValueError("model_variant='static_graph' requires cfg.static_graph_path")
            import pandas as _pd  # local import so it's not a hard dep at import time
            df = _pd.read_parquet(cfg.static_graph_path)
            # reorder rows/cols to match the ticker list (data.prepare fills it in).
            # if n_stocks isn't set yet we lazy-load on first forward.
            self._static_adj_df = df
            self.register_buffer("_static_A", torch.zeros(1, 1), persistent=False)
            self._static_A_ready = False

        # random graph: a FIXED random adjacency, top-k random neighbours per
        # node with arbitrary weights. built lazily in set_universe_tickers()
        # once n_stocks is known. control for "does the learned graph actually
        # beat an arbitrary fixed one?".
        if cfg.model_variant == "random_graph":
            self.register_buffer("_random_A", torch.zeros(1, 1), persistent=False)
            self._random_A_ready = False

        # learnable loss weighting (Kendall et al. 2018). only for
        # loss_kind='learnable'. two log-variances; training loss becomes
        # sum_k [ exp(-s_k)*L_k + s_k ] with s_k = log sigma_k^2. init 0 so both
        # terms start at unit weight, then the optimiser tunes the Huber-vs-ListMLE
        # balance itself. train.py only reads these when loss_kind=='learnable'.
        if getattr(cfg, "loss_kind", "combined") == "learnable":
            self.log_sigma_huber = nn.Parameter(torch.zeros(()))
            self.log_sigma_rank = nn.Parameter(torch.zeros(()))
        else:
            self.log_sigma_huber = None
            self.log_sigma_rank = None

    def set_universe_tickers(self, tickers: list) -> None:
        """trainer calls this once after build_loaders.
        - static_graph: reindex the loaded sector matrix to the ticker order.
        - random_graph: build a fixed random top-k adjacency.
        no-op for the other variants."""
        if self.cfg.model_variant == "static_graph":
            df = self._static_adj_df.reindex(index=tickers, columns=tickers)
            # FIX brittle if the static_graph parquet uses lowercase tickers, the
            # reindex would silently NaN every row and trip the check below.
            if df.isna().any().any():
                missing = df.index[df.isna().any(axis=1)].tolist()
                raise ValueError(f"static_graph_path missing tickers: {missing[:5]}...")
            # .copy() because pandas .to_numpy() can hand back a read-only view
            A = torch.from_numpy(df.to_numpy().copy()).to(dtype=torch.float32)
            self._static_A = A
            self._static_A_ready = True

        elif self.cfg.model_variant == "random_graph":
            # fixed random graph: each node gets topk random neighbours with
            # random weights in (0, 1]. seeded so it's reproducible.
            n = len(tickers)
            k = max(1, int(self.cfg.topk))
            g = torch.Generator().manual_seed(self.cfg.seed)
            A = torch.zeros(n, n)
            for i in range(n):
                # k random distinct neighbours, not i itself
                choices = torch.randperm(n, generator=g)
                choices = choices[choices != i][:k]
                w = torch.rand(k, generator=g)        # arbitrary weights
                A[i, choices] = w
            A = torch.maximum(A, A.t())               # symmetrise
            self._random_A = A
            self._random_A_ready = True

    def init_hidden(self, device) -> torch.Tensor:
        if self.graph is None:
            return torch.zeros(self.cfg.n_stocks, self.cfg.gru_hidden, device=device)
        return self.graph.init_hidden(self.cfg.n_stocks, device)

    def init_adjacency(self, device) -> torch.Tensor:
        if self.graph is None:
            return torch.zeros(self.cfg.n_stocks, self.cfg.n_stocks, device=device)
        return self.graph.init_adjacency(self.cfg.n_stocks, device)

    def forward(self, X_stock: torch.Tensor, X_macro: torch.Tensor,
                h_prev: torch.Tensor, A_prev: Optional[torch.Tensor] = None):
        N = X_stock.size(0)
        device = X_stock.device

        r_t = self.macro_encoder(X_macro)
        z = self.stock_encoder(X_stock, r_t=r_t)

        # center embeddings across stocks each day, otherwise they're nearly
        # identical (cosine ~0.99), the market-wide common move swamps the
        # per-stock signal. subtracting the cross-stock mean kills that common
        # factor so the graph / GAT / residual all see comparable de-meaned
        # per-stock reps, and the real signal is first-order instead of a tiny
        # wiggle. (goes with dropping the market-wide calendar features from the
        # encoder input, see config._FINAL_STOCK_FEATURES.) # WORKING
        if getattr(self.cfg, "cross_sectional_norm", False):
            z = z - z.mean(dim=0, keepdim=True)

        # z plays two roles:
        #   z (into GAT)  always carries the ID, gives graph-level identity so
        #                 it's not permutation-invariant
        #   z_residual    added to gat_out in the residual path. if
        #                 stock_id_in_residual is False we strip the ID so it
        #                 can't sneak in as a per-stock bias to the head.
        z_id_added = z
        if self.stock_id_emb is not None:
            stock_ids = torch.arange(z.size(0), device=device)
            id_vec = self.stock_id_emb(stock_ids)
            z_id_added = z + id_vec
        if getattr(self.cfg, "stock_id_in_residual", True):
            z_residual = z_id_added
        else:
            z_residual = z         # encoder-only, no ID bias in the residual
        z = z_id_added             # everything else (GAT, graph) uses z+ID

        # variant routing
        if self.cfg.model_variant == "no_graph":
            # no graph, no GAT, straight encoder to head. use z_residual so the
            # stock_id_in_residual flag still applies.
            embeds = z_residual
            logits = self.head(embeds)
            zero_A = torch.zeros(N, N, device=device)
            return {"logits": logits, "embeds": embeds,
                    "h": z, "A": zero_A, "eps": torch.tensor(0.0, device=device), "r_t": r_t}

        if self.cfg.model_variant == "static_graph":
            if not self._static_A_ready:
                raise RuntimeError(
                    "static_graph variant requires model.set_universe_tickers(loaders['tickers']) "
                    "to be called once after build_loaders()."
                )
            A = self._static_A.to(device)
            h = z                       # no GRU, node features are the encoder output
            eps = torch.tensor(0.0, device=device)
        elif self.cfg.model_variant == "random_graph":
            # fixed random adjacency, the arbitrary-graph control
            if not self._random_A_ready:
                raise RuntimeError(
                    "random_graph variant requires model.set_universe_tickers(...) first.")
            A = self._random_A.to(device)
            h = z
            eps = torch.tensor(0.0, device=device)
        elif self.cfg.model_variant in ("no_gru", "no_gat", "tagc"):
            # learn the graph with the constructor (part 2)
            h, A, eps = self.graph(z, h_prev, r_t=r_t, A_prev=A_prev)
        else:
            raise ValueError(f"unknown model_variant: {self.cfg.model_variant}")

        # GAT (part 3), skipped for no_gat. when skipped the constructor's node
        # memory h goes straight to residual + head.
        if self.gat is not None:
            node_out = self.gat(h, A)
        else:
            node_out = h                # no_gat: just use the GRU memory

        if self.residual_norm is not None:
            embeds = self.residual_norm(node_out + z_residual)
        else:
            embeds = node_out

        logits = self.head(embeds)
        return {"logits": logits, "embeds": embeds,
                "h": h, "A": A, "eps": eps, "r_t": r_t}
