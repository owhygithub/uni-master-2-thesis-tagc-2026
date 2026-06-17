"""dump the full graph for a day to a .npz so i can inspect it / reuse it later.

one file per (date, split) — the whole universe, not just a target-centric slice.
what's in each file:

    adj      [N, N]       the adjacency (after top-k), rows = source, cols = target
    embeds   [N, d_model] pre-head relation-aware embedding for every stock, so i can
                          grab any stock's prediction later via head(embeds[i])
                          without re-running the encoder/graph/GAT  # WORKING
    eps      ()           macro-conditioned eps_t for that day
    tickers  [N]          symbols, same order as the rows/cols/embeds
    date     str          YYYY-MM-DD
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class GraphLogger:
    def __init__(self, out_dir: Path, tickers: List[str], every: int = 1):
        self.dir = Path(out_dir)
        self.every = int(every)
        self.tickers = np.array(tickers, dtype="U16")
        self._counter = 0
        if self.every > 0:
            self.dir.mkdir(parents=True, exist_ok=True)

    def maybe_log(
        self,
        *,
        date: str,
        split: str,
        A: torch.Tensor,
        eps: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
    ) -> None:
        if self.every <= 0:
            return
        self._counter += 1
        if self._counter % self.every != 0:
            return

        payload = {
            "adj": A.detach().cpu().numpy().astype(np.float32),
            "eps": (float(eps.detach().cpu().item()) if eps.dim() == 0
                    else eps.detach().cpu().numpy().astype(np.float32)),
            "tickers": self.tickers,
            "date": np.array(date),
        }
        if embeds is not None:
            payload["embeds"] = embeds.detach().cpu().numpy().astype(np.float32)

        np.savez_compressed(self.dir / f"{split}_{date}.npz", **payload)
